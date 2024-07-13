import argparse
import cv2
import os
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
from tqdm import tqdm
from math import ceil
from glob import glob
from PIL import Image
import dataloaders
import models
from utils.helpers import colorize_mask
from collections import OrderedDict

def pad_image(img, target_size):

    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)

    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)

    return padded_img

def sliding_predict(model, image, num_classes, flip=True):

    image_size = image.shape
    tile_size = (int(image_size[2]//2.5), int(image_size[3]//2.5))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)

    total_predictions = np.zeros((num_classes, image_size[2], image_size[3]))
    count_predictions = np.zeros((image_size[2], image_size[3]))

    tile_counter = 0

    for row in range(num_rows):

        for col in range(num_cols):

            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = image[:, :, y_min:y_max, x_min:x_max]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            padded_prediction = model(padded_img)

            if flip:
                fliped_img = padded_img.flip(-1)
                fliped_predictions = model(fliped_img)
                padded_prediction = 0.5 * (fliped_predictions.flip(-1) + padded_prediction)

            predictions = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.data.cpu().numpy().squeeze(0)

    total_predictions /= count_predictions

    return total_predictions

def multi_scale_predict(model, image, scales, num_classes, device, flip=False):

    input_size = (image.size(2), image.size(3))
    upsample = nn.Upsample(size=input_size, mode="bilinear", align_corners=True)
    total_predictions = np.zeros((num_classes, image.size(2), image.size(3)))

    image = image.data.data.cpu().numpy()

    for scale in scales:

        scaled_img = ndimage.zoom(image, (1.0, 1.0, float(scale), float(scale)),
            order=1, prefilter=False)
        scaled_img = torch.from_numpy(scaled_img).to(device)
        scaled_prediction = upsample(model(scaled_img).cpu())

        if flip:
            fliped_img = scaled_img.flip(-1).to(device)
            fliped_predictions = upsample(model(fliped_img).cpu())
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)

        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)

    return total_predictions

def save_images(image, mask, output_path, image_file, palette):

    image_file = os.path.basename(image_file).split(".")[0]
    colorized_mask = colorize_mask(mask, palette)
    colorized_mask.save(os.path.join(output_path, image_file+".png"))

def save_uncertainty_masks(prediction, output_directory, file_name):

    C, _, _ = prediction.size()

    for i in range(C):
        
        mask = prediction[i].cpu().numpy() * 255
        mask = mask.round().astype(np.uint8)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_COOL)

        name, ext = os.path.splitext(os.path.basename(file_name))
        cv2.imwrite(os.path.join(output_directory, f"{name}-{i}{ext}"), mask)

def main():

    args = parse_arguments()
    config = json.load(open(args.config))

    # Dataset used for training the model
    dataset_type = config["train_loader"]["type"]
    assert dataset_type in ["Fourseasons"]

    scales = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    loader = getattr(dataloaders, config["train_loader"]["type"])(**config["train_loader"]["args"])
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(loader.MEAN, loader.STD)
    num_classes = loader.dataset.num_classes
    palette = loader.dataset.palette

    # Model
    model = getattr(models, config["arch"]["type"])(num_classes, **config["arch"]["args"])
    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:0" if len(availble_gpus) > 0 else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint.keys():
        checkpoint = checkpoint["state_dict"]

    # If during training, we used data parallel
    if "module" in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    output_dir = args.output

    if output_dir is None:

        output_name = "_".join(os.path.split(args.config)[0].split("/")[1:])
        output_dir = os.path.join("inference", output_name)

        if args.subdirectory is not None:
            output_dir = os.path.join(output_dir, f"{output_name}_{args.subdirectory}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.uncertainty and not os.path.exists(f"{output_dir}_uncertainty"):
        os.makedirs(f"{output_dir}_uncertainty")

    image_files = set(glob(os.path.join(args.images, f"*.{args.extension}")))

    if args.whitelist is not None:
        with open(args.whitelist) as fwhitelist:
            # whitelist consists of lines whith basename of images, e.g.
            #   2022-02-23-15-53-20_146229.png
            #   2022-10-19-14-14-59_206632.png
            #   ...
            whitelist = [line.replace("\n", "") for line in fwhitelist.readlines()]
            whitelist_files = {os.path.join(args.images, fname) for fname in whitelist}
            image_files = image_files.intersection(whitelist_files)

    image_files = sorted(list(image_files))

    with torch.no_grad():

        tbar = tqdm(image_files, ncols=100)

        for img_file in tbar:

            image = Image.open(img_file)

            if config["arch"]["args"]["in_channels"] == 3:
                image = image.convert("RGB")

            input = normalize(to_tensor(image)).unsqueeze(0)
            
            if args.mode == "multiscale":
                prediction = multi_scale_predict(model, input, scales, num_classes, device)
            elif args.mode == "sliding":
                prediction = sliding_predict(model, input, num_classes)
            else:
                prediction = model(input.to(device))
                prediction = prediction.squeeze(0).cpu().numpy()

            prediction = F.softmax(torch.from_numpy(prediction), dim=0)

            if args.uncertainty:
                save_uncertainty_masks(prediction, f"{output_dir}_uncertainty", img_file)

            prediction = prediction.argmax(0).cpu().numpy()
            save_images(image, prediction, output_dir, img_file, palette)

def parse_arguments():

    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("-c", "--config", default="VOC", type=str, help="The config used to train the model")
    parser.add_argument("-mo", "--mode", default="multiscale", type=str, help="Mode used for prediction: either [multiscale, sliding]")
    parser.add_argument("-m", "--model", default="model_weights.pth", type=str, help="Path to the .pth model checkpoint to be used in the prediction")
    parser.add_argument("-i", "--images", default=None, type=str, help="Path to the images to be segmented")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output Path")
    parser.add_argument("-w", "--whitelist", default=None, type=str, help="Path to whitelist file")
    parser.add_argument("-e", "--extension", default="png", type=str, help="The extension of the images to be segmented")
    parser.add_argument("-s", "--subdirectory", default=None, type=str, help="Subdirectory in output where images should be stored")
    parser.add_argument("-u", "--uncertainty", action="store_true", default=False, help="Store uncertainty masks")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    main()
