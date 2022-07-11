import argparse
import os


from utils.im_utils import *


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--input', type=str, default='1.jpg')
    ap.add_argument('--output', type=str, default='output')

    opt = vars(ap.parse_args())

    return opt


def main(opt):

    if os.path.isdir(opt['output']):
        os.mkdir(opt['output'])

    img = cv2.imread(opt['input'])

    img = fix_blur(img)

    corners = get_corners(img)
    img = align(img, corners)

    img = clean(img, BGR=True)
    img = deskew(img)





if __name__ == '__main__':

    opt = parse_opt()

    main(opt)