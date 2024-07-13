import torch
import torch.nn as nn
import torch.nn.functional as F

def make_one_hot(labels, classes):

    one_hot = torch.FloatTensor(labels.size()[0], classes, 
        labels.size()[2], labels.size()[3]).zero_().to(labels.device)

    return one_hot.scatter_(1, labels.data, 1)

def translate_ignore_index(target, ignore_index):

    if ignore_index not in range(target.min(), target.max()):
            
        if (target == ignore_index).sum() > 0:
            target[target == ignore_index] = target.min()
                
    return target

class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, ignore_index=255, reduction="mean"):

        weight_tensor = torch.tensor(weight, device="cuda:0") if weight is not None else None

        super(CrossEntropyLoss2d, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index,
            reduction=reduction)

    def forward(self, output, target):
        
        loss = self.cross_entropy(output, target)

        return loss

class FocalTverskyLoss(nn.Module):

    def __init__(self, smooth=1., ignore_index=255):

        super(FocalTverskyLoss, self).__init__()

        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target, alpha=0.7, gamma=0.75):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        target = translate_ignore_index(target, self.ignore_index)
        target = make_one_hot(target.unsqueeze(dim=1), classes=C)
        target = target * mask_keep

        output = F.softmax(output, dim=1)
        output = output * mask_keep

        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        TP = (output_flat * target_flat).sum() 
        FP = ((1 - target_flat) * output_flat).sum()
        FN = (target_flat * (1 - output_flat)).sum()
       
        Tversky = (TP + self.smooth) / (TP + alpha * FN + (1 - alpha) * FP + self.smooth)  
        
        return (1 - Tversky)**gamma

class DiceLoss(nn.Module):

    def __init__(self, smooth=1., ignore_index=255):

        super(DiceLoss, self).__init__()

        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        target = translate_ignore_index(target, self.ignore_index)
        target = make_one_hot(target.unsqueeze(dim=1), classes=C)
        target = target * mask_keep

        output = F.softmax(output, dim=1)
        output = output * mask_keep

        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
            (output_flat.sum() + target_flat.sum() + self.smooth))

        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):

        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)

        logpt = self.cross_entropy(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt)**self.gamma) * logpt

        loss = loss[mask_keep]

        if self.size_average:
            return loss.mean()

        return loss.sum()

class CE_DiceLoss(nn.Module):

    def __init__(self, smooth=1, reduction="mean", ignore_index=255, weight=None):

        super(CE_DiceLoss, self).__init__()

        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction,
            ignore_index=ignore_index)

    def forward(self, output, target):

        ce_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)

        return ce_loss + dice_loss

class ComboLoss(nn.Module):

    def __init__(self, smooth=1., ignore_index=255, alpha=0.5, ce_ratio=0.5, eps=1e-9):

        super(ComboLoss, self).__init__()

        self.ignore_index = ignore_index
        self.ce_ratio = ce_ratio
        self.smooth = smooth
        self.alpha = alpha
        self.eps = eps

    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        target = translate_ignore_index(target, self.ignore_index)
        target = make_one_hot(target.unsqueeze(dim=1), classes=C)
        target = target * mask_keep

        output = F.softmax(output, dim=1)
        output = output * mask_keep

        mask_keep_flat = mask_keep.contiguous().view(-1)
        output_flat = output.contiguous().view(-1)[mask_keep_flat]
        target_flat = target.contiguous().view(-1)[mask_keep_flat]

        TP = (output_flat * target_flat).sum()
        dice = (2. * TP + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)

        output_flat = torch.clamp(output_flat, self.eps, 1 - self.eps)

        out = -(self.alpha * ((target_flat * torch.log(output_flat)) + 
            ((1 - self.alpha) * (1.0 - target_flat) * torch.log(1.0 - output_flat))))
        weighted_ce = out.mean(-1)
        combo = (self.ce_ratio * weighted_ce) - ((1 - self.ce_ratio) * dice)
        
        return combo

class JaccardLoss(nn.Module):

    def __init__(self, smooth=1., ignore_index=255):

        super(JaccardLoss, self).__init__()

        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        target = translate_ignore_index(target, self.ignore_index)
        target = make_one_hot(target.unsqueeze(dim=1), classes=C)
        target = target * mask_keep

        output = F.softmax(output, dim=1)
        output = output * mask_keep

        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (output_flat * target_flat).sum() 
        total = (output_flat + target_flat).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smooth) / (union + self.smooth)

        return 1 - IoU

class DistributionLoss(nn.Module):

    def __init__(self, ignore_index=255) -> None:
        
        super(DistributionLoss, self).__init__()

        self.ignore_index = ignore_index

    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)

        # normalize along classes
        output = F.softmax(output, dim=1)
        # keep N x H x W tensor which contains indices for
        # classes with highest probability at each pixel
        output_dist = torch.argmax(output, dim=1, keepdims=False)
        # output_dist now is a N x H x W tensor and each pixel
        # has a value in [0, ...] indicating the plane/class
        # with the highest probability, now shift by 1 in order to
        # for the planes/classes to be in range [1, ...] and multiply
        # with mask_keep s.t. a 0 now indicate 'ignore'
        output_dist = (output_dist + 1) * mask_keep
        # count how often each class occurs as best guess 
        # and keep an additional bin for 'ignore' by C+1
        # and keep ignore_index out by slicing ([:, 1:])
        output_dist = torch.nn.functional.one_hot(output_dist.view(N, -1), C+1)
        output_dist = output_dist.sum(dim=1)[:, 1:].float()
        # normalize each histogram in batch to receive distribution
        output_dist = output_dist / output_dist.sum(dim=1).unsqueeze(dim=1).repeat(1, C)

        # target label values should be contiguous and ideally start at 0
        # if they do, shift by 1 in order to keep '0' for 'ignore'
        target_dist = target + 1 if target.min() == 0 and self.ignore_index != 0 else target
        target_dist = target_dist * mask_keep
        # count how often each class (and ignore_index) occurs in label
        target_dist = torch.nn.functional.one_hot(target_dist.view(N, -1), C+1)
        target_dist = target_dist.sum(dim=1)[:, 1:].float()
        target_dist = target_dist / target_dist.sum(dim=1).unsqueeze(dim=1).repeat(1, C)

        # L1 norm for each distribution and then sum of all
        loss = torch.abs(target_dist - output_dist).sum(dim=1)

        return loss.mean()
    
class SpeckleLoss(nn.Module):

    def __init__(self, ignore_index=255, num_classes=4, kernel_size=5, scale_factor=1e-4,
        threshold=None) -> None:
        
        super(SpeckleLoss, self).__init__()

        assert kernel_size % 2 == 1

        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.threshold = threshold

        self.kernel = torch.ones((num_classes, 1, kernel_size, kernel_size))
        self.kernel = self.kernel * (1 / (kernel_size**2 - 1))
        self.kernel[:, :, kernel_size//2, kernel_size//2] = -1        

        if threshold is None:
            self.threshold = (kernel_size**2 - kernel_size) * (1 / (kernel_size**2 - 1))
    
    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        # normalize along classes
        output = F.softmax(output, dim=1)

        # set to 1 along class where maximum is
        output_one_hot = torch.argmax(output, dim=1, keepdims=True)
        output_one_hot = torch.zeros_like(output).scatter_(1, output_one_hot, 1.)
        # ignore one hot pixels in regions from ignore_index
        # and set them all to 1 in order to not influence other pixels
        output_one_hot = (output_one_hot * mask_keep) + (~mask_keep).float()

        # set to 2 everywhere where tensor is zero 
        # i.e. that tensor now has 1 where maxima are and elsewhere 2
        output_non_zero = torch.where(output_one_hot.double()==0., 2., output_one_hot.double())
        
        padding = (self.kernel_size//2, self.kernel_size//2, self.kernel_size//2, self.kernel_size//2)
        output_non_zero = F.pad(output_non_zero, padding, "constant", 1)

        self.kernel = self.kernel.to(device=output_non_zero.get_device())

        # 0 when all neighbor pixel (in kernel_size x kernel_size neighborhood) have the same class
        # maximally 1 when all neighbor pixels have different label
        differential = F.conv2d(output_non_zero, self.kernel.double(), padding=0, bias=None,
            groups=self.num_classes)
        differential = torch.where(differential < self.threshold, 0., differential)
        differential = (output * differential) * mask_keep

        loss = differential.sum(-1).sum(-1).sum(-1)
        loss = self.scale_factor * loss.mean()

        return loss
    
class CE_DistributionLoss(nn.Module):

    def __init__(self, reduction="mean", ignore_index=255, weight=None, num_classes=4, ce_ratio=0.5):

        super(CE_DistributionLoss, self).__init__()

        self.ce_ratio = ce_ratio
        self.ignore_index = ignore_index

        self.distribution = DistributionLoss(ignore_index=ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction,
            ignore_index=ignore_index)
    
    def forward(self, output, target):

        distribution_loss = self.distribution(output, target)
        ce_loss = self.cross_entropy(output, target)

        print(f"distribution_loss: {distribution_loss}, ce_loss: {ce_loss}, ce_ratio: {self.ce_ratio}")

        return self.ce_ratio * ce_loss + (1 - self.ce_ratio) * distribution_loss

class CE_SpeckleLoss(nn.Module):

    def __init__(self, reduction="mean", ignore_index=255, weight=None, num_classes=4, ce_ratio=0.6,
        kernel_size=5, scale_factor=1e-4, threshold=None):

        super(CE_SpeckleLoss, self).__init__()

        self.ce_ratio = ce_ratio
        self.ignore_index = ignore_index

        self.speckle = SpeckleLoss(ignore_index=ignore_index, num_classes=num_classes,
            kernel_size=kernel_size, scale_factor=scale_factor, threshold=threshold)
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction,
            ignore_index=ignore_index)

    def forward(self, output, target):

        speckle_loss = self.speckle(output, target)
        ce_loss = self.cross_entropy(output, target)

        return self.ce_ratio * ce_loss + (1 - self.ce_ratio) * speckle_loss

class LogCoshDiceLoss(nn.Module):

    def __init__(self, ignore_index=255, epsilon=1e-6):
        super(LogCoshDiceLoss, self).__init__()

        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, output, target):

        N, C, H, W = output.size()

        # mask for pixels to be considered for loss
        mask_keep = (target != self.ignore_index)
        # transform from N x H x W mask to N x C x H x W mask
        # the mask can differ for each sample in the batch
        # but not b/w the classes for one sample
        mask_keep = mask_keep.unsqueeze(dim=1).repeat(1, C, 1, 1)

        target = translate_ignore_index(target, self.ignore_index)
        target = make_one_hot(target.unsqueeze(dim=1), classes=C)
        target = target * mask_keep

        output = F.softmax(output, dim=1)
        output = output * mask_keep

        numerator = 2. * torch.sum(output * target, dim=(-2, -1))
        denominator = torch.sum(output + target, dim=(-2, -1))

        loss = torch.log(torch.cosh(1 - torch.mean((numerator + self.epsilon) /
            (denominator + self.epsilon))))

        return loss
