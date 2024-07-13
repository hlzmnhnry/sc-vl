import numpy as np
import torch

class AverageMeter(object):

    def __init__(self):

        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):

        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):

        if not self.initialized: self.initialize(val, weight)
        else: self.add(val, weight)

    def add(self, val, weight):

        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(predict, target, labeled):

    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"

    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):

    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def confusion_matrix(y_true, y_pred, num_classes):

    y_true = y_true.clone().detach().to(dtype=torch.long)
    y_pred = y_pred.clone().detach().to(dtype=torch.long)

    y = num_classes * y_true + y_pred
    y = torch.bincount(y)

    if len(y) < num_classes * num_classes:
        y = torch.cat((y, torch.zeros(num_classes * num_classes - len(y), device=y.get_device(),
            dtype=torch.long)))

    y = y.reshape(num_classes, num_classes)

    return y

def eval_metrics(output, target, num_class):

    _, predict = torch.max(output.data, dim=1)
    predict, target = predict + 1, target + 1

    labeled = (target > 0) * (target <= num_class)
    correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
    inter, union = batch_intersection_union(predict, target, num_class, labeled)

    conf_mat = confusion_matrix((labeled * target).view(-1), (labeled * target).view(-1),
        num_classes=num_class+1)
    conf_mat = conf_mat[1:, 1:]

    return [np.round(correct, 5), np.round(num_labeled, 5), np.round(inter, 5),
        np.round(union, 5), conf_mat]
