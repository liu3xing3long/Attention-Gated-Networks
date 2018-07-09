'''
Misc Utility functions
'''

import os
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
# from utils.metrics import segmentation_scores, dice_score_list
from sklearn import metrics
from .layers.loss import *
from skimage.exposure import rescale_intensity


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imgtype='img', datatype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 4:  # image_numpy (C x W x H x S)
        mid_slice = image_numpy.shape[-1] // 2
        image_numpy = image_numpy[:, :, :, mid_slice]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if imgtype == 'img':
        image_numpy = (image_numpy + 8) / 16.0 * 255.0
    if np.unique(image_numpy).size == int(1):
        return image_numpy.astype(datatype)
    return rescale_intensity(image_numpy.astype(datatype))


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def segmentation_scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    eps = 1e-10
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / (hist.sum() + eps)
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + eps)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + eps)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum() + eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return {'overall_acc': acc,
            'mean_acc': acc_cls,
            'freq_w_acc': fwavacc,
            'mean_iou': mean_iu}


def dice_score_list(label_gt, label_pred, n_class):
    """

    :param label_gt: [WxH] (2D images)
    :param label_pred: [WxH] (2D images)
    :param n_class: number of label classes
    :return:
    """
    epsilon = 1.0e-6
    assert len(label_gt) == len(label_pred)
    batchSize = len(label_gt)
    dice_scores = np.zeros((batchSize, n_class), dtype=np.float32)
    for batch_id, (l_gt, l_pred) in enumerate(zip(label_gt, label_pred)):
        for class_id in range(n_class):
            img_A = np.array(l_gt == class_id, dtype=np.float32).flatten()
            img_B = np.array(l_pred == class_id, dtype=np.float32).flatten()
            score = 2.0 * np.sum(img_A * img_B) / (np.sum(img_A) + np.sum(img_B) + epsilon)
            dice_scores[batch_id, class_id] = score

    return np.mean(dice_scores, axis=0)


def get_optimizer(option, params):
    opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=option.l2_reg_weight)

    if opt_alg == 'adam':
        optimizer = optim.Adam(params,
                               lr=option.lr_rate,
                               betas=(0.9, 0.999),
                               weight_decay=option.l2_reg_weight)

    return optimizer


def get_criterion(opts):
    if opts.criterion == 'cross_entropy':
        if opts.type == 'seg':
            criterion = cross_entropy_2D if opts.tensor_dim == '2D' else cross_entropy_3D
        elif 'classifier' in opts.type:
            criterion = CrossEntropyLoss()
    elif opts.criterion == 'dice_loss':
        criterion = SoftDiceLoss(opts.output_nc)
    elif opts.criterion == 'dice_loss_pancreas_only':
        criterion = CustomSoftDiceLoss(opts.output_nc, class_ids=[0, 2])

    return criterion


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9, ):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * (1 - iter / max_iter) ** power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.size(1)
    pred_lbls = pred_seg.data.max(1)[1].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)

    iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice = dice_score_list(gts, preds, n_class=n_classes)

    return iou, dice


def classification_scores(gts, preds, labels):
    accuracy = metrics.accuracy_score(gts, preds)
    class_accuracies = []
    for lab in labels:  # TODO Fix
        class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
    class_accuracies = np.array(class_accuracies)

    f1_micro = metrics.f1_score(gts, preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro = metrics.recall_score(gts, preds, average='micro')
    f1_macro = metrics.f1_score(gts, preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro = metrics.recall_score(gts, preds, average='macro')

    # class wise score
    f1s = metrics.f1_score(gts, preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls = metrics.recall_score(gts, preds, average=None)

    confusion = metrics.confusion_matrix(gts, preds, labels=labels)

    # TODO confusion matrix, recall, precision
    return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


def classification_stats(pred_seg, target, labels):
    return classification_scores(target, pred_seg, labels)
