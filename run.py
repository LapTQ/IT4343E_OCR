import argparse
import os
import json
import yaml
import time

import PIL.Image
import torch

import cv2
import numpy as np


from utils import im_utils, file_utils, model_utils


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--input', type=str, default='1.jpg') # TODO demo default
    ap.add_argument('--output', type=str, default='output')

    opt = vars(ap.parse_args())

    return opt


def main(opt):

    if os.path.isdir(opt['output']):
        os.makedirs(opt['output'], exist_ok=True)

    filename = os.path.basename(opt['input'])

    img = cv2.imread(opt['input'])

    proc_img = im_utils.fix_blur(img)
    corners = im_utils.get_corners(proc_img)
    proc_img = im_utils.align(proc_img, corners)
    proc_img = im_utils.clean(proc_img, BGR=True)
    proc_img = im_utils.deskew(proc_img)

    out_path = os.path.join('cache', os.path.splitext(filename)[0] + '.jpg')
    cv2.imwrite(out_path, proc_img)
    print('[INFO] Cache saved in', out_path)

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cuda = torch.cuda.is_available()
    config['cuda'] = cuda
    print(f"[INFO] Cuda is {'' if cuda else 'not '}available")

    detect_net, refine_net, recog_net = model_utils.load_weights(
        config['detect_pth'],
        config['refine_pth'],
        config['recog_pth'],
        config['cuda']
    )

    t0 = time.time()

    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB)  # RGB order

    quads, texts, probs = model_utils.predict_one_image(proc_img, detect_net, refine_net, recog_net, config)
    json_dict = file_utils.json_from_prediction(filename, proc_img, quads, texts, probs)

    # save output using 4 corner instead of all contours
    out_path = os.path.join(opt['output'], os.path.splitext(filename)[0] + '.json')
    with open(out_path, 'w', encoding='utf8') as f:
        json.dump(json_dict, f, indent=4, ensure_ascii=False)
    f.close()
    print('[INFO] JSON saved in', out_path)

    # demo
    out_path = os.path.join(opt['output'], os.path.splitext(filename)[0] + '.jpg')
    box_drawed = file_utils.draw_box(proc_img, json_dict, draw_text=False)
    text_drawed = file_utils.draw_text(proc_img, json_dict)
    collage = np.concatenate([box_drawed, text_drawed], axis=1)
    cv2.imwrite(out_path, collage)
    print('[INFO] Demo saved in', out_path)


    print('[INFO] Total time:', time.time() - t0)

    cv2.namedWindow('Final result', cv2.WINDOW_NORMAL)
    cv2.imshow('Final result', collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





if __name__ == '__main__':

    opt = parse_opt()

    main(opt)