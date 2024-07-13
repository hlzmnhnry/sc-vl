import numpy as np
import random
import torch
import math
import cv2

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def rotatedRectWithMaxArea(w, h, angle):

    if w <= 0 or h <= 0: return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin, cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))

    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr,hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr

class BaseDataSet(Dataset):

    def __init__(self, root, split, mean, std, base_size=None, augment=True, 
        val=False, crop_size=0, scale=True, flip=True, rotate=False,
        blur=False, brightness=False, return_id=False, ignore_label=255):
        
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size

        if self.augment:

            self.brightness = brightness
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.ignore_label = ignore_label
        self.return_id = return_id

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):

        if self.crop_size:

            h, w = label.shape

            # Scale the smaller side to crop size
            h, w = (self.crop_size, int(self.crop_size * w / h)) if h < w else \
                (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size ) // 2
            start_w = (w - self.crop_size ) // 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        return image, label

    def _augmentation(self, image, label):
        
        # save channel size
        h, w, c = image.shape

        # zoom into random areas
        if self.scale:

            zoom_factor = random.uniform(0.75, 1.)
            zoom_h = round(zoom_factor * h)
            zoom_w = round(zoom_factor * w)

            start_h = random.randint(0, h - zoom_h)
            start_w = random.randint(0, w - zoom_w)

            end_h = start_h + zoom_h
            end_w = start_w + zoom_w

            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        # reshape in order to keep channel size
        image = image.reshape((image.shape[0], image.shape[1], c))
        h, w, c = image.shape

        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            
            angle = random.randint(-15, 15)
            center = (w/2, h/2)

            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)

            max_w, max_h = rotatedRectWithMaxArea(w, h, np.deg2rad(angle))
            y0, y1 = int(center[1] - max_h//2), int(center[1] + max_h//2)
            x0, x1 = int(center[0] - max_w//2), int(center[0] + max_w//2)
            
            image = image[y0:y1, x0:x1]
            label = label[y0:y1, x0:x1]

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

            if random.random() > 0.5:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)

        # Random H/V flip
        if self.flip:
            
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

            if random.random() > 0.5:
                image = np.flipud(image).copy()
                label = np.flipud(label).copy()

        # Random Brightness and Contrast change
        if self.brightness:

            alpha = random.uniform(0.9, 1.1)
            beta = random.randint(-30, 30)
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # Gaussian Blur (sigma between 0 and 1)
        if self.blur:

            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101)
            
        # reshape in order to keep channel size
        image = image.reshape((image.shape[0], image.shape[1], c))
        h, w, c = image.shape

        # Random rectangular crop
        if self.crop_size:

            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)

            pad_kwargs = {
                "top": pad_h // 2,
                "bottom": pad_h // 2 + pad_h % 2,
                "left": pad_w // 2,
                "right": pad_w // 2 + pad_w % 2,
                "borderType": cv2.BORDER_CONSTANT
            }

            if pad_h > 0 or pad_w > 0:

                label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)
                image = cv2.copyMakeBorder(image, value=self.ignore_label, **pad_kwargs)

                image = image.reshape((image.shape[0], image.shape[1], c))
                h, w, c = image.shape

            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)

            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size

            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # reshape in order to keep channel size
        image = image.reshape((image.shape[0], image.shape[1], c))

        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image, label, image_id = self._load_data(index)

        if self.val: image, label = self._val_augmentation(image, label)
        elif self.augment: image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image.squeeze()))

        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id

        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):

        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "\t#data: {}\n".format(self.__len__())
        fmt_str += "\tSplit: {}\n".format(self.split)
        fmt_str += "\tRoot: {}".format(self.root)

        return fmt_str
