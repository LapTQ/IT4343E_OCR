import numpy as np
import cv2


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

        corner_selected = input('[QUERY] Are you sure with your choice? [y/n] ').strip()[0].lower() == 'y'

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

    return input('[QUERY] Auto detect corners. Is this okay? [y/n] ').strip()[0].lower() == 'y'


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

    print(f"[INFO] Proposed corners are {'' if good else 'not '}good")
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


if __name__ == '__main__':
    path = '/media/tran/003D94E1B568C6D11/Workingspace/handwritten_text_recognition/1.jpg'
    img  = cv2.imread(path)

    pts = get_corners(img)
    img = align(img, pts)
    img = clean(img, BGR=True)

    import matplotlib.pyplot as plt

    plt.imshow(img[:, :, ::-1])
    plt.show()

