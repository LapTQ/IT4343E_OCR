# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# RGB
PURPLE = (77, 0, 75)
GREEN = (62, 160, 85)
RED = (32, 44, 216)


def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def json_from_prediction(filename, img, boxes, texts, probs):
    img = np.array(img)

    json_dict = {"filename": filename, "width": img.shape[1], "height": img.shape[0], "textboxes": []}
    for id, (box, text, prob) in enumerate(zip(boxes, texts, probs)):
        poly = np.array(box).astype(np.int32).reshape((-1))
        json_dict["textboxes"].append({
            "id": id,
            "bndbox": [{"x": int(poly[i]), "y": int(poly[i + 1])} for i in range(0, len(poly), 2)],
            "confidence": prob,
            "text": text,
        })

    return json_dict


def fit_fontsize(font_path, target_h):
    s = 1
    while True:
        font = ImageFont.truetype(font_path, s)
        w, h = font.getsize('dummy text')
        if h > target_h:
            break
        s += 1
    return ImageFont.truetype(font_path, s - 1)


def draw_box(img, json_dict, draw_text=False):

    color = RED

    ids = [tb['id'] for tb in json_dict['textboxes']]
    boxes = [np.array([[pt['x'], pt['y']] for pt in tb['bndbox']], dtype='int32')
             for tb in json_dict['textboxes']]
    texts = [tb['text'] for tb in json_dict['textboxes']]

    min_h = min([box[1][1] - box[0][1] for box in boxes])
    font = fit_fontsize('utils/Arial.ttf', min_h)

    for id, box, text in zip(ids, boxes, texts):

        box += np.array([[3, 3], [3, -3], [-3, -3], [-3, 3]])
        cv2.polylines(img, [box.reshape((-1, 1, 2))], True, color=color, thickness=1)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_draw = ImageDraw.Draw(img)

        id_w, id_h = font.getsize(str(id))
        img_draw.rectangle(((box[0, 0] - id_w, box[0, 1]),
                            (box[0, 0], box[0, 1] + id_h)),
                           fill=color[::-1],
                           outline=color[::-1])
        img_draw.text((box[0, 0] - id_w, box[0, 1]),
                      str(id),
                      fill='white',
                      font=font)

        if draw_text:
            text_w, text_h = font.getsize(text)

            img_draw.rectangle(((box[0][0], box[0][1] - text_h),
                                (box[0][0] + text_w, box[0][1])),
                               fill=color[::-1],
                               outline=color[::-1])

            img_draw.text((box[0][0], box[0][1] - text_h),
                          "{}".format(text),
                          fill='white',
                          font=font)

        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def draw_text(img, json_dict):

    color = RED

    img = Image.new('RGBA', (img.shape[1], img.shape[0]), (255, 255, 255, 92))
    img_draw = ImageDraw.Draw(img)

    for tb in json_dict['textboxes']:
        id = tb['id']
        box = np.array([[pt['x'], pt['y']] for pt in tb['bndbox']], dtype='int32')
        text = tb['text']

        font = fit_fontsize('utils/Arial.ttf', min(box[1][1] - box[0][1], box[3][0] - box[0][0]) * 2.5/4)

        (_, _), (w, h), rot = cv2.minAreaRect(box)
        text_w, text_h = font.getsize(text)

        if rot > 0:
            if w < h:
                rot = 90 - rot
            else:
                rot = 180 - rot
        elif w < h:
            rot = 90 + rot

        text_img = Image.new('RGBA', (text_w, text_h), (255, 255, 255, 92))
        text_img_draw = ImageDraw.Draw(text_img)
        text_img_draw.text((0, 0), text, font=font, fill=(0, 0, 0))
        text_img = text_img.rotate(rot, expand=1)

        img.paste(text_img, cv2.boundingRect(box)[:2], text_img)

        id_w, id_h = font.getsize(str(id) + ' ')
        img_draw.text((box[0, 0] - id_w, box[0, 1]),
                      str(id),
                      fill=color[::-1],
                      font=font)

    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    return img


def get_image_list(path):
    image_list = None

    if os.path.isdir(path):
        image_list, _, _ = get_files(path)
    else:
    #elif os.path.isfile(path) and os.path.splitext(os.path.basename(path))[1] in ['.jpg', '.jpeg', '.png']:
        image_list = [path]

    return image_list
