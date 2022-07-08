import numpy as np
import cv2
import imutils
from tqdm import tqdm

import matplotlib.pyplot as plt


def query(msg):
    return not bool(input('[QUERY] ' + msg + ' [<ENTER> for yes] ').strip())


def info(msg):
    print('[INFO] ' + msg)


def warn(msg):
    print('[WARNING] ' + msg)

def binary_segment(img):

    H, W = img.shape[:2]

    img = cv2.medianBlur(img, 21)

    Z = img.reshape(-1, 3).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # # intermediate visualization
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res = res.reshape(img.shape)

    seg = label.reshape(H, W).astype('uint8') * 255

    kernel = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])
    seg = cv2.dilate(seg, kernel=kernel, iterations=3)

    return seg


def detect_corner(img):
    """
    Auto detect 4 corners.
    :param img: np.array shape (H, W, 1), binary image
    :return: np.array shape (4, 1, 2)
    """
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)
    k = sorted([(k, stats[k, cv2.CC_STAT_AREA]) for k in range(1, ret)],
               key=lambda x: x[1],
               reverse=True)[0][0]

    loc = np.roll(np.where(labels == k), 1, axis=0).transpose().reshape(-1, 1, 2)
    hull = cv2.convexHull(loc)

    eps = 0.1 * cv2.arcLength(hull, True)
    pts = cv2.approxPolyDP(hull, eps, True)

    if pts.shape[0] != 4:
        return None

    return pts

    # demo = cv2.cvtColor((labels == k).astype('uint8') * 255, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(demo, [pts.astype('int32')], -1, color=(0, 255, 0))
    #
    # return demo


def select_corners():
    """
    Let user manually select corners of document.
    :return: np.array
    """
    global img_copy
    global buffer

    def on_mouse(event, x, y, flags, param):
        global buffer
        global img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_copy, (x, y), radius=4, color=(0, 0, 255), thickness=8)
            if len(buffer) > 0:
                cv2.line(img_copy, (x, y), buffer[-1], color=(0, 255, 0), thickness=2)
            if len(buffer) == 3:
                cv2.line(img_copy, (x, y), buffer[0], color=(0, 255, 0), thickness=2)
            buffer.append([x, y])
            cv2.imshow('Select corners', img_copy)

    cv2.namedWindow('Select corners', cv2.WINDOW_NORMAL)
    img_copy = img.copy()
    cv2.imshow('Select corners', img_copy)

    corner_selected = False

    while not corner_selected:
        cv2.namedWindow('Select corners', cv2.WINDOW_NORMAL)
        img_copy = img.copy()
        buffer = []
        cv2.imshow('Select corners', img_copy)
        cv2.setMouseCallback('Select corners', on_mouse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        corner_selected = query('Are you sure with your choice?')

    pts = np.array(buffer).reshape(4, 1, 2)

    return pts


def good_corners(img, pts):
    pts = arrange_corners(pts)
    demo = img.copy()
    cv2.drawContours(
        image=demo,
        contours=[pts],
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=2
    )
    cv2.namedWindow('Corners preview', cv2.WINDOW_NORMAL)
    cv2.imshow('Corners preview', demo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return query('Auto detect corners. Is this okay?')


def arrange_corners(pts):
    """
    Arrange corner in the order: top-left, bottom-left, bottom-right, top-right
    :param corners: numpy array of shape (4, 1, 2)
    :return: numpy array of shape (4, 1, 2)
    """
    shape = pts.shape
    pts = np.squeeze(pts).tolist()
    pts = sorted(pts, key=lambda x: x[0])
    pts = sorted(pts[:2], key=lambda x: x[1]) + sorted(pts[2:], key=lambda x: x[1], reverse=True)
    return np.array(pts).reshape(shape)


def get_corners(img):
    """
    Return upper left, bottom left, bottom right, upper right corner of document.
    :param img: np.array
    :return: np.array
    """

    seg = binary_segment(img)
    pts = detect_corner(seg)
    good = False
    if pts is not None:
        good = good_corners(img, pts)

    info(f"Proposed corners are {'' if good else 'not '}good.")
    if not good:
        pts = select_corners()

    pts = arrange_corners(pts)

    return pts


def align(img, pts):

    pts = pts.reshape(4, 1, 2)
    rect = cv2.boundingRect(pts)
    x1, y1, w, h = rect

    target_pts = np.array([[0, 0], [0, h], [w, h], [w, 0]]).reshape(4, 1, 2)

    M = cv2.getPerspectiveTransform(pts.astype('float32'), target_pts.astype('float32'))
    # homography, mask = cv2.findHomography(pts, target_pts, cv2.RANSAC)

    img = cv2.warpPerspective(img, M, (w, h))

    return img


def clean(img, BGR):
    assert BGR is True or BGR is False, f'Invalid argument: BGR must be True or False, got {BGR}'
    if BGR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = img.astype(np.float32)

    val, cnt = np.unique(img, return_counts=True)
    med = cnt[np.argmin(np.abs(cnt - int(np.median(cnt))))]
    alpha = np.max(val[np.where(cnt == med)])

    thresh = (255 * 255 + 1.2 * alpha * alpha - 255 * alpha) / (255 - (1 - 1.2) * alpha) - alpha
    beta = 255 / thresh

    img = (img - alpha) * beta

    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def local_max(spec, size, thresh=float('-inf'), skip_center=False):
    pos = []
    h, w = spec.shape[:2]
    i = 0
    while i < h - size:
        j = 0
        while j < w - size:
            p = np.argmax(spec[i: i + size, j:j + size])
            p = (i + p // size, j + p % size)
            if spec[p] > thresh:
                if p != (h // 2, w // 2) or not skip_center:
                    pos.append(p)
            j += size
        i += size

    return pos


def calc_rot(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    mag = 20 * np.log(np.abs(fft_shift))

    pos = local_max(mag, 15, skip_center=True)
    p = sorted([(p, mag[p]) for p in pos], key=lambda x: x[1], reverse=True)[0][0]

    h, w = mag.shape[:2]
    rot = np.arctan2(p[0] - h//2, p[1] - w//2)
    rot = np.degrees(rot) % 90

    return rot


def deskew(img):
    rot = calc_rot(img)
    img = imutils.rotate_bound(img, -rot)

    return img


from scipy.signal import convolve2d


def motion_blur(img, kernel, noise=None):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    out = convolve2d(img, kernel, mode='valid')

    if noise is not None:
        noise = np.random.normal(0, noise, out.shape)
        out += noise

    out = np.clip(out, 0, 255).astype('uint8')
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    return out


def wiener_filter(img, kernel=None):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if kernel is None:
        kernel = np.eye(3) / 3
    g = np.fft.fft2(img)
    h = np.fft.fft2(kernel, s=img.shape)

    f_ = g / (h * (1 + 0.002/np.abs(h) ** 2))
    f = np.abs(np.fft.ifft2(f_))

    rec = cv2.cvtColor(f.astype('uint8'), cv2.COLOR_GRAY2BGR)

    return rec


def get_psf(size, theta):

    h = np.zeros([max(size) * 2] * 2, dtype='float32')
    h = cv2.ellipse(h,
                    (h.shape[1] // 2, h.shape[0] // 2),
                    size,
                    90 - theta,
                    0, 360,
                    255, -1)
    h /= np.sum(h)
    return h


def wiener(h, nsr, shape):
    h_fft = np.fft.fft2(h, s=shape)
    HW = np.conj(h_fft) / (np.abs(h_fft) ** 2 + nsr)

    return HW


def deblur(img, kernel, nsr):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    HW = wiener(kernel, nsr, img.shape)
    G = np.fft.fft2(img)
    F = G * HW
    rec = np.abs(np.fft.ifft2(F))
    return cv2.cvtColor(rec.astype('uint8'), cv2.COLOR_GRAY2BGR)


def check_blur(img, size=60, thresh=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    cx, cy = int(w/2), int(h/2)

    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)

    # for visualization
    # mag = 20 * np.log(np.abs(fft_shift))

    fft_shift[cy - size: cy + size, cx - size:cx + size] = 0
    fft_rec = np.fft.ifftshift(fft_shift)
    rec = np.fft.ifft2(fft_rec)

    mag = 20 * np.log(np.abs(rec))
    mean = np.mean(mag)

    if thresh is None:
        return mean
    else:
        return mean < thresh


def edge_taper(img, gamma, beta):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    w1 = np.zeros((w,), dtype='float32')
    w2 = np.zeros((h,), dtype='float32')

    dx = 2 * 3.14 / w
    x = -3.14
    for i in range(w):
        w1[i] = 0.5 * (np.tanh((x + gamma/2) / beta) - (np.tanh((x - gamma/2) / beta)))
        x += dx

    dy = 2 * 3.14 / h
    y = -3.14
    for i in range(h):
        w2[i] = 0.5 * (np.tanh((y + gamma/2) / beta) - (np.tanh((y - gamma/2) / beta)))
        y += dy

    w1 = w1.reshape(1, w)
    w2 = w2.reshape(h, 1)

    img = img * np.matmul(w2, w1)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)



def fix_blur(img):

    if not check_blur(img, thresh=15):
        info('Image is not blurry.')
        return img
    elif query('Image seems to be blurry. Continue?'):
        warn('Skipping blurry image. This might affect accuracy.')
        return img

    # TODO fix blur here
    kernel = get_psf((1, 6), 0)
    # img = edge_taper(img, 5, 0.2)
    rec = deblur(img, kernel, 0.05)
    return rec


if __name__ == '__main__':
    path = '/media/tran/003D94E1B568C6D11/Workingspace/handwritten_text_recognition/8.jpg'
    img  = cv2.imread(path)

    # img = motion_blur(img, kernel, noise=3)

    img = fix_blur(img)

    # corners = get_corners(img)
    # img = align(img, corners)

    img = clean(img, BGR=True)
    img = deskew(img)

    plt.figure(figsize=(20, 20)); plt.imshow(img[:, :, ::-1] if len(img.shape) == 3 else img); plt.show()





