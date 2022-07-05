import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import cv2
import numpy as np

import file_utils
import craft_utils2 as craft_utils
import imgproc
from craft import CRAFT

from collections import OrderedDict
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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
# TODO đang để thresh cụ thể cho 1 tờ giấy khai sinh
parser.add_argument('--low_text', default=0.5, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--test_folder', default='data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, type=str2bool, help='enable link refiner')  # đã xóa action='store_true', thêm cái type
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

image_list, _, _ = file_utils.get_files(args.test_folder)

cache_folder = './cache/'
if not os.path.isdir(cache_folder):
    os.mkdir(cache_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, refine_net=None):

    # NOTE: Xử lý ảnh trước khi đưa vào mô hình:
    # 1. resize ảnh giữ nguyên tỉ lệ, mặc định là gấp lên 1.5 lần nhưng không quá 1280
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio        # ratio_h, ratio_w là tỉ lệ cạnh cũ/mới

    # NOTE: Xử lý ảnh trước khi đưa vào mô hình:
    # 2. normalize về mean khoảng độ 0.5, var = 0.22
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    contours = craft_utils.getContours(score_text, score_link, text_threshold, link_threshold, low_text)

    # coordinate adjustment
    contours = craft_utils.adjustResultCoordinates(contours, ratio_w, ratio_h)

    return contours

if __name__ == '__main__':
    net = CRAFT()

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if args.refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()

    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        filename, file_ext = os.path.splitext(os.path.basename(image_path))

        # Load ảnh RGB, shape (H, W, 3)
        image = imgproc.loadImage(image_path)
        image = imgproc.enhance_image(image, BGR=False)     # BGR order
        cv2.imwrite(os.path.join(cache_folder, filename + '.jpg'), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # RGB order

        contours = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, refine_net)

        file_utils.saveResult(image_path, image[:,:,::-1], contours, dirname=cache_folder)
