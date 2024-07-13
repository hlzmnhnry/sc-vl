from PIL import Image
import numpy as np
import os

from base import BaseDataSet, BaseDataLoader
from utils import palette
from glob import glob

IGNORE_LABEL = 255

ID_TO_TRAINID = {
    0: IGNORE_LABEL,
    1: 0,   # city is 1
    2: 1,   # forest is 2
    3: 2,   # water is 3
    4: 3    # agriculture 4
}

class FourseasonsDataset(BaseDataSet):

    def __init__(self, **kwargs):

        self.num_classes = 4
        self.palette = palette.Fourseasons_palette
        self.id_to_trainId = ID_TO_TRAINID
        
        super(FourseasonsDataset, self).__init__(ignore_label=IGNORE_LABEL, **kwargs)

    def _set_files(self):

        assert self.split in ["train", "val"]

        # fourseasons
        # '--> images
        #       '--> train
        #       '--> val
        # '--> labels
        #       '--> train
        #       '--> val

        label_path = os.path.join(self.root, "labels", self.split)
        image_path = os.path.join(self.root, "images", self.split)

        assert os.listdir(image_path) == os.listdir(label_path)

        image_paths = sorted(glob(os.path.join(image_path, "*.png")))
        label_paths = sorted(glob(os.path.join(label_path, "*.png")))

        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index):
        
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image = image.reshape((image.shape[0], image.shape[1], 1))
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        for k, v in self.id_to_trainId.items():
            label[label == k] = v

        return image, label, image_id

class Fourseasons(BaseDataLoader):

    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True,
        num_workers=1, val=False, shuffle=False, flip=False, rotate=False, blur=False,
        brightness=False, augment=False, val_split= None, return_id=False):

        # version 2.1
        # Mean over TRAIN images: 0.4595016350385154
        # Std over TRAIN images: 0.2174796899718768
        # Mean over ALL images: 0.4651325818301502
        # Std over ALL images: 0.21397184485040677

        # version 3.1
        # Mean over TRAIN images: 0.47512318902238065
        # Std over TRAIN images: 0.20924651854601792
        # Mean over ALL images: 0.4795455861590171
        # Std over ALL images: 0.20505177729551483
        self.MEAN = [0.47512318902238065]
        self.STD = [0.20924651854601792]

        kwargs = {
            "root": data_dir,
            "split": split,
            "mean": self.MEAN,
            "std": self.STD,
            "augment": augment,
            "crop_size": crop_size,
            "base_size": base_size,
            "scale": scale,
            "flip": flip,
            "blur": blur,
            "brightness": brightness,
            "rotate": rotate,
            "return_id": return_id,
            "val": val
        }

        self.dataset = FourseasonsDataset(**kwargs)
        super(Fourseasons, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
