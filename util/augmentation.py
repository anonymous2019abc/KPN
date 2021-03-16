import numpy as np
import math
import cv2
import copy
import numpy.random as random

from shapely.geometry import Polygon
import time

###<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<###
###<<<<<<<<<  Function  >>>>>>>>>>>>###
###>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>###
def crop_first(image, polygons, scale =10):
    polygons_new = copy.deepcopy(polygons)
    h, w, _ = image.shape
    pad_h = h // scale
    pad_w = w // scale
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

    text_polys = []
    pos_polys = []
    for polygon in polygons_new:
        rect = cv2.minAreaRect(polygon.points.astype(np.int32))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        text_polys.append([box[0], box[1], box[2], box[3]])
        if polygon.label != -1:
            pos_polys.append([box[0], box[1], box[2], box[3]])

    polys = np.array(text_polys, dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)  # 四舍五入
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text 保证截取区域不会横穿文字
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    pp_polys = np.array(pos_polys, dtype=np.int32)

    return h_axis, w_axis, pp_polys


def fill_text(image, polygons=None):
    # KPN 20201104
    image_fill_text = image.copy() + 0
    color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    color = tuple([int(x) for x in color])
    # print(color)
    if polygons is not None:
        for poly_idx, poly in enumerate(polygons):
            poly_in32_ = poly.points.astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(image_fill_text, poly_in32_, tuple(color))
    return image_fill_text

####<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<####
####<<<<<<<<<<<  Class  >>>>>>>>>>>>>####
####>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>####
class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class SquarePadding(object):

    def __call__(self, image, polygons=None):

        H, W, _ = image.shape

        if H == W:
            return image, polygons

        padding_size = max(H, W)
        (h_index, w_index) = (np.random.randint(0, H*7//8),np.random.randint(0, W*7//8))
        # img_cut = image[h_index:(h_index+H//9),w_index:(w_index+W//9)]
        #KPN 20201104
        image_fill_text = fill_text(image, polygons)
        img_cut = image_fill_text[h_index:(h_index+H//9),w_index:(w_index+W//9)]
        expand_image = cv2.resize(img_cut,(padding_size, padding_size))
        #expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)
        #expand_image=img_cut[:,:,:]
        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if polygons is not None:
            for polygon in polygons:
                polygon.points += np.array([x0, y0])
        expand_image[y0:y0+H, x0:x0+W] = image
        image = expand_image

        return image, polygons



class Resize(object):
    def __init__(self, size=(480, 1024)):
        self.size = size
        self.SP = SquarePadding()

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons


class ResizeSquare(object):
    def __init__(self, size=(480, 1280)):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        img_size_min = min(h, w)
        img_size_max = max(h, w)

        if img_size_min < self.size[0]:
            im_scale = float(self.size[0]) / float(img_size_min)  # expand min to size[0]
            if np.round(im_scale * img_size_max) > self.size[1]:  # expand max can't > size[1]
                im_scale = float(self.size[1]) / float(img_size_max)
        elif img_size_max > self.size[1]:
            im_scale = float(self.size[1]) / float(img_size_max)
        else:
            im_scale = 1.0

        new_h = int(int(h * im_scale/32)*32)
        new_w = int(int(w * im_scale/32)*32)
        #new_h = int(h * im_scale)
        #new_w = int(w * im_scale)
        if new_h*new_w >= 1600*1600:
            im_scale = 1600/float(img_size_max)
            new_h = int(int(h * im_scale/32)*32)
            new_w = int(int(w * im_scale/32)*32)
            #new_h = int(h * im_scale)
            #new_w = int(w * im_scale)

        image = cv2.resize(image, (new_w, new_h))
        scales = np.array([new_w / w, new_h / h])
        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales

        return image, polygons




class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Normalize(mean=self.mean, std=self.std),
        ])
    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            # Resize(size),
            ResizeSquare(size=self.size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransformNresize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)
