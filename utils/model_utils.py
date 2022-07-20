import tensorflow as tf
from tensorflow import keras
from jiwer import wer
from .data_utils import *
import numpy as np
from tqdm import tqdm
import os
import time
import cv2
import PIL
import math

import torch
import torch.backends.cudnn as cudnn

from collections import OrderedDict

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from basenet.craft import CRAFT

from utils import im_utils

# unwarp corodinates
def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def getContours(textmap, linkmap, text_threshold, link_threshold, low_text):

    linkmap = linkmap.copy()
    textmap = textmap.copy()

    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    # gộp mask của region và link
    # TODO thử cách quét qua từng link score, rồi xem những region score nào dính với nó
    text_score_comb = np.clip(text_score + link_score, 0, 1)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8),
                                                                         connectivity=4)
    contours = []
    for k in range(1, nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # tạo segmentation map CHỈ chứa các region thuộc cc thứ k
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        segmap_copy = segmap.copy()
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # nhằm loại bỏ nhiễu

        # tính tọa độ của 4 góc  (chừa ra một ít)
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h

        inner = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x1, y1, ww, hh = cv2.boundingRect(inner)
        mask = np.zeros_like(segmap_copy)
        mask[y1:y1 + hh, x1:x1 + ww] = 255
        segmap_copy = np.logical_and(segmap_copy, mask).astype(np.uint8) * 255

        # dilate các segment của từng chữ, TỐI ĐA trong vùng của 4 góc vừa tính
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel, iterations=3)
        segmap = np.logical_or(segmap, segmap_copy).astype(np.uint8) * 255

        [cnt], hierarchy = cv2.findContours(segmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contours.append(cnt.reshape(-1, 2).astype(np.float32))

    return contours


def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


def arrange_corners(corners):
    """
    Arrange corner in the order: top-left, bottom-left, bottom-right, top-right
    :param corners: numpy array of shape (4, 1, 2)
    :return: numpy array of shape (4, 1, 2)
    """
    shape = corners.shape
    corners = np.squeeze(corners).tolist()
    corners = sorted(corners, key=lambda x: x[0])
    corners = sorted(corners[:2], key=lambda x: x[1]) + sorted(corners[2:], key=lambda x: x[1], reverse=True)
    return np.array(corners).reshape(shape)


def decode_batch(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=False, beam_width=4)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(NUM_TO_CHAR(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


class CallbackEval(keras.callbacks.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        predictions = []
        targets = []
        for batch in tqdm(self.dataset):
            y_pred = self.model.predict(batch)
            y_pred = decode_batch(y_pred)
            predictions.extend(y_pred)
            for label in batch['y_true']:
                label = tf.strings.reduce_join(NUM_TO_CHAR(label)).numpy().decode('utf-8')
                targets.append(label)
        wer_score = wer(targets, predictions)
        print(f'WER: {wer_score:.4f}')
        for i in np.random.randint(0, len(predictions), 24):
            print(f'True: {targets[i]}')
            print(f'Pred: {predictions[i]}')


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_weights(detect_pth, refine_pth, recog_pth, cuda):

    # detection model
    detect_net = CRAFT()

    print('[INFO] Loading weights of detection model from checkpoint (' + detect_pth + ')')
    if cuda:
        detect_net.load_state_dict(copyStateDict(torch.load(detect_pth)))
    else:
        detect_net.load_state_dict(copyStateDict(torch.load(detect_pth, map_location='cpu')))

    if cuda:
        detect_net = detect_net.cuda()
        detect_net = torch.nn.DataParallel(detect_net)
        cudnn.benchmark = False
    print('[INFO] Done!')

    detect_net.eval()

    refine_net = None
    if refine_pth:
        from basenet.refinenet import RefineNet

        refine_net = RefineNet()
        print('[INFO] Loading weights of refiner from checkpoint (' + refine_pth + ')')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_pth)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refine_pth, map_location='cpu')))
        print('[INFO] Done!')

        refine_net.eval()

    # recognition model
    config = Cfg.load_config_from_name('vgg_transformer')

    if os.path.exists(recog_pth):
        print('[INFO] Loading weights of recognition model from checkpoint (' + recog_pth + ')')
        config['weights'] = recog_pth
    else:
        # config['weights'] = 'https://drive.google.com/uc?id=1--0gOdyQXIhQArom-bcDE0ZMuUeVvcUj'
        config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained'] = True
    if cuda:
        config['device'] = 'cuda:0'
    else:
        config['device'] = 'cpu'

    recog_net = Predictor(config)
    print('[INFO] Done')

    return detect_net, refine_net, recog_net


def predict_one_image(image, detect_net, refine_net, recog_net, config):

    img_resized, target_ratio, size_heatmap = im_utils.resize_aspect_ratio(
        image,
        config['canvas_size'],
        mag_ratio=config['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    x = im_utils.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]

    if config['cuda']:
        x = x.cuda()

    t0 = time.time()

    with torch.no_grad():
        y, feature = detect_net(x)
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().numpy()

    print('[INFO] Time for detection:', time.time() - t0)

    t0 = time.time()

    contours = getContours(score_text, score_link, config['text_threshold'], config['link_threshold'], config['low_text'])
    contours = adjustResultCoordinates(contours, ratio_w, ratio_h)

    # recognition
    H = 118
    patches = []
    max_W = 0
    quads = []

    for cnt in contours:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [cnt.astype(np.int32)], -1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_8)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        loc = np.roll(np.where(mask == 255), 1, axis=0).transpose().reshape(-1, 2)
        quad = cv2.minAreaRect(loc)
        quad = cv2.boxPoints(quad)
        quad = arrange_corners(quad).astype(np.float32)
        quads.append(quad.astype(np.int32))
        rect = cv2.boundingRect(quad.astype(np.int32))
        x, y, w, h = rect
        if h < 3 * w:
            rect = np.array(
                [
                    [0, 0],
                    [0, h],
                    [w, h],
                    [w, 0]
                ],
                dtype=np.float32
            )
        else:
            w, h = h, w
            rect = np.array(
                [
                    [w, 0],
                    [0, 0],
                    [0, h],
                    [w, h]
                ],
                dtype=np.float32
            )

        if config['mask_contour']:
            cc_image = np.full_like(image, 255)
            cc_image[mask == 255] = image[mask == 255]
        else:
            cc_image = image

        mat = cv2.getPerspectiveTransform(quad, rect)
        pat = cv2.warpPerspective(cc_image, mat, (w, h))
        W = int(H / pat.shape[0] * pat.shape[1])
        pat = cv2.resize(pat, (W, H))
        patches.append(pat)
        if W > max_W:
            max_W = W

    for i in range(len(patches)):
        patches[i] = np.pad(patches[i], ((0, 0), (0, max_W - patches[i].shape[1]), (0, 0)), mode='constant',
                            constant_values=255)

    # CHÚ Ý: patches đang ở dạng uint8, [0, 255]
    patches = [PIL.Image.fromarray(p) for p in patches]

    print('[INFO] Time for mid-process:', time.time() - t0)

    t0 = time.time()

    texts, probs = recog_net.predict_batch(patches, return_prob=True)

    print('[INFO] Time for reognition:', time.time() - t0)

    return quads, texts, probs
