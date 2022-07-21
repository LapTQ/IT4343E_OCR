import numpy as np
import cv2
import imutils
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt

from utils.generals import *


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


def select_corners(img):
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
        pts = select_corners(img)

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


def auto_clean(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


def clean(img, BGR):
    assert BGR is True or BGR is False, f'Invalid argument: BGR must be True or False, got {BGR}'
    if not BGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    val, cnt = np.unique(img, return_counts=True)
    med = cnt[np.argmin(np.abs(cnt - int(np.median(cnt))))]
    alpha = np.max(val[np.where(cnt == med)])

    thresh = (255 * 255 + 1.2 * alpha * alpha - 255 * alpha) / (255 - (1 - 1.2) * alpha) - alpha
    beta0 = 255 / thresh
    img = img - alpha

    root = tk.Tk()
    root.title('Select filter level co-efficients')
    root.geometry('400x540+50+50')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)

    def get_tk_img(photo):
        mag = min(550 / photo.shape[0], 1)
        h, w = int(mag * photo.shape[0]), int(mag * photo.shape[1])
        photo = cv2.resize(np.clip(photo, 0, 255).astype('uint8'), (w, h))
        photo = cv2.cvtColor(photo, cv2.COLOR_GRAY2RGB)
        data = f'P6 {w} {h} 255 '.encode() + photo.tobytes()
        return tk.PhotoImage(width=w, height=h, data=data, format='PPM')

    def img_change():
        img2 = img * beta.get()
        img_tk = get_tk_img(img2)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

    img_tk = get_tk_img(img * beta0)

    img_label = ttk.Label(root, image=img_tk)
    img_label.grid(column=0, row=0, columnspan=2, padx=5, pady=5)

    beta = tk.DoubleVar(value=beta0)

    def beta_value():
        return 'level: %.2f' % (beta.get())

    def beta_change(event):
        beta_label.config(text=beta_value())
        img_change()

    beta_label = ttk.Label(root, text=beta_value())
    beta_label.grid(column=0, row=1, sticky='w', padx=5, pady=5)
    beta_slider = ttk.Scale(root, from_=1, to=10, orient='horizontal', variable=beta, command=beta_change)
    beta_slider.grid(column=1, row=1, sticky='we', padx=5, pady=5)

    root.mainloop()

    img = img * beta.get()

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
    img_rot = imutils.rotate_bound(img, -rot)

    root = tk.Tk()
    root.title('Select deskew angle')
    root.geometry('400x540+50+50')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)

    def get_tk_img(photo):
        mag = min(550 / photo.shape[0], 1)
        h, w = int(mag * photo.shape[0]), int(mag * photo.shape[1])
        photo = cv2.resize(photo, (w, h))
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        data = f'P6 {w} {h} 255 '.encode() + photo.tobytes()
        return tk.PhotoImage(width=w, height=h, data=data, format='PPM')

    def img_change():
        img2 = imutils.rotate_bound(img, -angle.get())
        img_tk = get_tk_img(img2)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

    img_tk = get_tk_img(imutils.rotate_bound(img, -rot))

    img_label = ttk.Label(root, image=img_tk)
    img_label.grid(column=0, row=0, columnspan=2, padx=5, pady=5)

    angle = tk.DoubleVar(value=rot)

    def angle_value():
        return 'angle: %.2f degree' % (angle.get())

    def angle_change(event):
        angle_label.config(text=angle_value())
        img_change()

    angle_label = ttk.Label(root, text=angle_value())
    angle_label.grid(column=0, row=1, sticky='w', padx=5, pady=5)
    angle_slider = ttk.Scale(root, from_=-90, to=90, orient='horizontal', variable=angle, command=angle_change)
    angle_slider.grid(column=1, row=1, sticky='we', padx=5, pady=5)

    root.mainloop()

    return imutils.rotate_bound(img, -angle.get())



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

    kernel, nsr = select_deblur_coef(img)
    # img = edge_taper(img, 10, 0.2)
    rec = deblur(img, kernel, nsr)
    return rec


def select_deblur_coef(img):
    root = tk.Tk()
    root.title('Select deblur co-efficients')
    root.geometry('400x540+50+50')

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=3)

    def get_tk_img(photo):
        mag = min(550 / photo.shape[0], 1)
        h, w = int(mag * photo.shape[0]), int(mag * photo.shape[1])
        photo = cv2.resize(photo, (w, h))
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        data = f'P6 {w} {h} 255 '.encode() + photo.tobytes()
        return tk.PhotoImage(width=w, height=h, data=data, format='PPM')

    def img_change():
        kernel = get_psf((y.get(), x.get()), angle.get())
        img2 = deblur(img, kernel, 2 ** nsr_log2.get())
        img_tk = get_tk_img(img2)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

    img_tk = get_tk_img(img)

    img_label = ttk.Label(root, image=img_tk)
    img_label.grid(column=0, row=0, columnspan=2, padx=5, pady=5)

    angle = tk.DoubleVar(value=0)

    def angle_value():
        return 'angle: %.2f degree' % (angle.get())

    def angle_change(event):
        angle_label.config(text=angle_value())
        img_change()


    angle_label = ttk.Label(root, text=angle_value())
    angle_label.grid(column=0, row=1, sticky='w', padx=5, pady=5)
    angle_slider = ttk.Scale(root, from_=-90, to=90, orient='horizontal', variable=angle, command=angle_change)
    angle_slider.grid(column=1, row=1, sticky='we', padx=5, pady=5)

    x = tk.IntVar(value=1)

    def x_value():
        return 'x: %d pixels' % (x.get())

    def x_change(event):
        x_label.config(text=x_value())
        img_change()

    x_label = ttk.Label(root, text=x_value())
    x_label.grid(column=0, row=2, sticky='w', padx=5, pady=5)
    x_slider = ttk.Scale(root, from_=1, to=20, orient='horizontal', variable=x, command=x_change)
    x_slider.grid(column=1, row=2, sticky='we', padx=5, pady=5)

    y = tk.IntVar(value=1)

    def y_value():
        return 'y: %d pixels' % (y.get())

    def y_change(event):
        y_label.config(text=y_value())
        img_change()

    y_label = ttk.Label(root, text=y_value())
    y_label.grid(column=0, row=3, sticky='w', padx=5, pady=5)
    y_slider = ttk.Scale(root, from_=1, to=20, orient='horizontal', variable=y, command=y_change)
    y_slider.grid(column=1, row=3, sticky='we', padx=5, pady=5)

    nsr_log2 = tk.DoubleVar(value=-9.9)

    def nsr_value():
        return 'nsr: %.3f' % (2 ** nsr_log2.get())

    def nsr_change(event):
        nsr_label.config(text=nsr_value())
        img_change()

    nsr_label = ttk.Label(root, text=nsr_value())
    nsr_label.grid(column=0, row=4, sticky='w', padx=5, pady=5)
    nsr_slider = ttk.Scale(root, from_=-10, to=0, orient='horizontal', variable=nsr_log2, command=nsr_change)
    nsr_slider.grid(column=1, row=4, sticky='we', padx=5, pady=5)

    root.mainloop()

    return get_psf((y.get(), x.get()), angle.get()), 2 ** nsr_log2.get()


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation=cv2.INTER_LINEAR, mag_ratio=1):
    """
    NOTE: ảnh đầu vào được resize giữ nguyên tỉ số
    Bước 1: Kích thước cạnh được nhân (giữ nguyên tỉ lệ) lên tối đa là mag_ratio lần kích thước ban đầu, nhưng không quá 1280
    Bước 2: Pad thêm 0 vào cạnh dưới và cạnh phải, sao cho 2 cạnh đều chia hết cho 32.
    :param img: ảnh RGB, shape (H, W, 3)
    :param square_size: kích thước (~ tối đa khi resize)
    :param interpolation:
    :param mag_ratio: nhân ảnh lên gấp bao nhiêu lần (không vượt quá cái ngưỡng square_size). Mặc định là giữ nguyên.
    :return: ảnh mới đã resize, tỉ lệ cạnh mới/cũ, kích thước của heatmap (1/2 kích thước ảnh đã resize)
    """
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


if __name__ == '__main__':
    path = '/media/tran/003D94E1B568C6D11/Workingspace/handwritten_text_recognition/input/007.jpg'
    img0  = cv2.imread(path)

    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)

    val, cnt = np.unique(img, return_counts=True)
    med = cnt[np.argmin(np.abs(cnt - int(np.median(cnt))))]
    alpha = np.max(val[np.where(cnt == med)])

    plt.figure(figsize=(10, 15))

    plt.subplot(3, 1, 1)
    plt.hist(img.ravel(), bins=int(np.max(img) - np.min(img)), range=[np.min(img), np.max(img)])
    plt.plot([alpha, alpha], plt.ylim())
    plt.xlim([-600, 600])

    thresh = (255 * 255 + 1.2 * alpha * alpha - 255 * alpha) / (255 - (1 - 1.2) * alpha) - alpha
    beta = 255 / thresh
    img = img - alpha
    plt.subplot(3, 1, 2)
    plt.hist(img.ravel(), bins=int(np.max(img) - np.min(img)), range=[np.min(img), np.max(img)])
    plt.plot([thresh, thresh], plt.ylim(), color='lightgreen')
    plt.xlim([-600, 600])


    img = img * beta

    plt.subplot(3, 1, 3)
    plt.hist(img.ravel(), bins=int(np.max(img) - np.min(img)), range=[np.min(img), np.max(img)])
    plt.xlim([-600, 600])
    plt.savefig('hist.jpg')

    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    cv2.imwrite('cleaned.jpg', np.concatenate([img0, img], axis=1))


    # plt.figure(figsize=(20, 20)); plt.imshow(img[:, :, ::-1] if len(img.shape) == 3 else img); plt.show()

