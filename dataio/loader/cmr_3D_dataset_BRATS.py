import os
import datetime
from collections import OrderedDict
import random
import logging

import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torchsample.transforms as ts
import torch
import torch.utils.data as data

from .utils import load_nifti_img, check_exceptions, is_image_file


class CMR3DDatasetBRATS(data.Dataset):
    def __init__(self, dataset_folder, subset_folder, label_folder, data_subsets, keywords=["P1", "1", "flair"],
                 mode='train', transform=None, aug_opts=None):
        super(CMR3DDatasetBRATS, self).__init__()

        self.rawim = {}
        self.rawmask = []
        self.keywords = keywords
        self.train_mode = mode == 'train'
        self.test_mode = mode == 'test'
        self.val_mode = mode == 'val'
        self.data_path_map = OrderedDict()
        self.data_name_vec = []

        self.affine_shift_val = (0.1, 0.1)
        self.affine_rotate_val = 15.0
        self.affine_scale_val = (0.7, 1.3)
        self.random_flip_prob = 0.5

        self.origin_size = 0
        # self.bbox = np.array([[0, 0, 0], [0, 0, 0]])
        self.margin = 8

        if aug_opts is None:
            self.preload = False
        else:
            self.preload = aug_opts.preload_data
            self.affine_scale_val = aug_opts.scale
            self.affine_rotate_val = aug_opts.rotate
            self.affine_shift_val = aug_opts.shift
            self.random_flip_prob = aug_opts.random_flip_prob

        if self.train_mode:
            if aug_opts is None:
                self.augment_scale = 0
            else:
                self.augment_scale = aug_opts.augment_scale

            if aug_opts is None:
                self.scale_size = [128, 128, 96]
                self.patch_size = [128, 128, 96]
            else:
                self.scale_size = aug_opts.scale_size
                self.patch_size = aug_opts.patch_size
        elif self.val_mode or self.test_mode:
            # no aug_scale for val and test data
            self.augment_scale = 0
            if aug_opts is None:
                self.scale_size = [128, 128, 96]
                self.patch_size = [128, 128, 96]
            else:
                self.scale_size = aug_opts.scale_size
                self.patch_size = aug_opts.patch_size

        # read csv settings
        csv_folder = os.path.join(dataset_folder, label_folder)

        # since we have LGG and HGG two folders...
        # data_name ---> data_folder_path
        for subset in data_subsets:
            df = pd.read_csv(os.path.join(csv_folder, "subset_{}.csv".format(subset)), header=None)
            for df_idx, df_row in df.iterrows():
                data_name = df_row[0]
                for sub_folder in subset_folder:
                    real_data_folder = os.path.join(dataset_folder, sub_folder)
                    data_real_folder = os.path.join(real_data_folder, data_name)
                    if os.path.isdir(data_real_folder):
                        self.data_path_map[data_name] = data_real_folder
                        self.data_name_vec.append(data_name)
                        break
        for keywd in keywords:
            self.rawim[keywd] = []

        if self.preload:
            for data_name, data_real_folder in tqdm(self.data_path_map.items()):
                #######################
                # reading mask data
                tmp_rawmask = []

                mask_name = "{}_{}.nii.gz".format(data_name, "seg")
                abs_image_path = os.path.join(data_real_folder, mask_name)
                # logging.debug("reading mask from {}".format(abs_image_path))

                mask = sitk.ReadImage(abs_image_path)
                mask_arr = sitk.GetArrayFromImage(mask)
                mask_arr_d, mask_arr_h, mask_arr_w = mask_arr.shape

                mask_arr = mask_arr.astype(np.float)
                # z, y, x -> x, y, z
                mask_arr = np.transpose(mask_arr, [2, 1, 0])
                mask_arr[mask_arr > 0] = 1

                # new_mask_arr = np.zeros(mask_arr.shape)
                # new_mask_arr[mask_arr == 1] = 3
                # new_mask_arr[mask_arr == 4] = 2
                # new_mask_arr[mask_arr > 0] = 1
                # mask_arr = new_mask_arr

                # mask_arr = mask_arr[
                #     max((mask_arr.shape[0] - self.patch_size[0]) / 2, 0):
                #     min(mask_arr.shape[0] - (mask_arr.shape[0] - self.patch_size[0]) / 2, mask_arr.shape[0]),
                #     max((mask_arr.shape[1] - self.patch_size[1]) / 2, 0):
                #     min(mask_arr.shape[1] - (mask_arr.shape[1] - self.patch_size[1]) / 2, mask_arr.shape[1]),
                #     max((mask_arr.shape[2] - self.patch_size[2]) / 2, 0):
                #     min(mask_arr.shape[2] - (mask_arr.shape[2] - self.patch_size[2]) / 2, mask_arr.shape[2])]
                # print("mask_arr shape {}".format(mask_arr.shape))
                self.rawmask.append(mask_arr)

                #######################
                # reading real image data
                # bbox = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                for keywd in self.keywords:
                    image_name = "{}_{}.nii.gz".format(data_name, keywd)
                    abs_image_path = os.path.join(data_real_folder, image_name)
                    # logging.debug("reading image from {}".format(abs_image_path))

                    im = sitk.ReadImage(abs_image_path)
                    im_arr = sitk.GetArrayFromImage(im)
                    im_arr = im_arr.astype(np.float)
                    # z, y, x -> x, y, z
                    im_arr = np.transpose(im_arr, [2, 1, 0])

                    # store the whole shape
                    image_shape = im_arr.shape

                    # calculate bbox
                    # im_arr_non_zero = np.where(im_arr > 0)
                    # this_bbox_min = np.array(
                    #         [np.min(im_arr_non_zero[0]), np.min(im_arr_non_zero[1]), np.min(im_arr_non_zero[2])])
                    # this_bbox_max = np.array(
                    #         [np.max(im_arr_non_zero[0]), np.max(im_arr_non_zero[1]), np.max(im_arr_non_zero[2])])
                    # logging.debug("data {}, mod {},  size {} -> {}, delta {}".format(data_name, keywd,
                    #                                                                  this_bbox_min, this_bbox_max,
                    #                                                                  this_bbox_max - this_bbox_min))
                    # bbox[0] += this_bbox_min
                    # bbox[1] += this_bbox_max

                    # im_arr = im_arr[
                    #            max((im_arr.shape[0] - self.patch_size[0]) / 2, 0):
                    #            min(im_arr.shape[0] - (im_arr.shape[0] - self.patch_size[0]) / 2, im_arr.shape[0]),
                    #            max((im_arr.shape[1] - self.patch_size[1]) / 2, 0):
                    #            min(im_arr.shape[1] - (im_arr.shape[1] - self.patch_size[1]) / 2, im_arr.shape[1]),
                    #            max((im_arr.shape[2] - self.patch_size[2]) / 2, 0):
                    #            min(im_arr.shape[2] - (im_arr.shape[2] - self.patch_size[2]) / 2, im_arr.shape[2])]
                    # print("im_arr shape {}".format(im_arr.shape))
                    # append the images
                    self.rawim[keywd].append(im_arr)

                # # calculate avg bbox
                # bbox[0] = np.maximum(bbox[0] - self.margin, np.array([0, 0, 0]))
                # bbox[1] = np.minimum(bbox[1] + self.margin, image_shape)
                # bbox = bbox.astype(np.int)
                # logging.debug("bbox {} -> {}".format(bbox[0], bbox[1]))

            for keywd in self.keywords:
                # transform to np array
                self.rawim[keywd] = np.array(self.rawim[keywd])

            self.rawmask = np.array(self.rawmask)
            # output data shapes
            for keywd in self.keywords:
                logging.debug(
                        "data shape for modality {} in {} mode is {}".format(keywd, mode, self.rawim[keywd].shape))
            logging.debug("mask shape in {} mode is {}".format(mode, self.rawmask.shape))

            # debug infor output
            logging.debug("total voxels {}, valid voxels {}, rate {}".format(self.rawmask.size, np.sum(self.rawmask),
                                                                             np.sum(self.rawmask) / self.rawmask.size))
            self.origin_size = len(self.data_name_vec)

        else:
            self.origin_size = len(self.data_name_vec)

        # data augmentation
        self.transform = transform

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        b_aug = False
        if index >= self.origin_size:
            index = index % self.origin_size
            b_aug = True

        if self.preload:
            # load the nifti images
            img_tr = []
            mask_tr = []
            for keywd in self.keywords:
                img_tr.append(self.rawim[keywd][index])

            mask_tr.append(self.rawmask[index])

            img_tr = np.array(img_tr)
            mask_tr = np.array(mask_tr)

            img_tr, mask_tr = self.augment(img_tr, mask_tr, b_aug=b_aug)

            return img_tr, mask_tr

        else:
            #######################
            # reading mask data
            data_name = self.data_name_vec[index]
            data_real_folder = self.data_path_map[data_name]
            mask_name = "{}_{}.nii.gz".format(data_name, "seg")
            abs_image_path = os.path.join(data_real_folder, mask_name)
            # logging.debug("reading mask from {}".format(abs_image_path))

            img_tr, mask_tr = [], []
            mask = sitk.ReadImage(abs_image_path)
            mask_arr = sitk.GetArrayFromImage(mask)
            mask_arr_d, mask_arr_h, mask_arr_w = mask_arr.shape
            mask_arr = mask_arr.astype(np.float)
            # z, y, x -> x, y, z
            mask_arr = np.transpose(mask_arr, [2, 1, 0])
            mask_arr[mask_arr > 0] = 1

            # update multi
            # new_mask_arr = np.zeros(mask_tr.shape)
            # new_mask_arr[mask_tr > 0] = 1
            # new_mask_arr[mask_tr == 1] = 3
            # new_mask_arr[mask_tr == 4] = 2
            # mask_tr = new_mask_arr
            # mask_arr = mask_arr[
            #            max((mask_arr.shape[0] - self.patch_size[0]) / 2, 0):
            #            min(mask_arr.shape[0] - (mask_arr.shape[0] - self.patch_size[0]) / 2, mask_arr.shape[0]),
            #            max((mask_arr.shape[1] - self.patch_size[1]) / 2, 0):
            #            min(mask_arr.shape[1] - (mask_arr.shape[1] - self.patch_size[1]) / 2, mask_arr.shape[1]),
            #            max((mask_arr.shape[2] - self.patch_size[2]) / 2, 0):
            #            min(mask_arr.shape[2] - (mask_arr.shape[2] - self.patch_size[2]) / 2, mask_arr.shape[2])]
            # print("mask_arr shape {}".format(mask_arr.shape))
            mask_tr.append(mask_arr)
            mask_tr = np.array(mask_tr)

            #######################
            # reading real image data
            for keywd in self.keywords:
                image_name = "{}_{}.nii.gz".format(data_name, keywd)
                abs_image_path = os.path.join(data_real_folder, image_name)
                # logging.debug("reading image from {}".format(abs_image_path))
                im = sitk.ReadImage(abs_image_path)
                im_arr = sitk.GetArrayFromImage(im)
                im_arr = im_arr.astype(np.float)
                # z, y, x -> x, y, z
                im_arr = np.transpose(im_arr, [2, 1, 0])

                # calculate bbox
                # im_arr_non_zero = np.where(im_arr > 0)
                # this_bbox_min = np.array(
                #         [np.min(im_arr_non_zero[0]), np.min(im_arr_non_zero[1]), np.min(im_arr_non_zero[2])])
                # this_bbox_max = np.array(
                #         [np.max(im_arr_non_zero[0]), np.max(im_arr_non_zero[1]), np.max(im_arr_non_zero[2])])
                # self.bbox[0] += this_bbox_min
                # self.bbox[1] += this_bbox_max

                # im_arr = im_arr[
                #          max((im_arr.shape[0] - self.patch_size[0]) / 2, 0):
                #          min(im_arr.shape[0] - (im_arr.shape[0] - self.patch_size[0]) / 2, im_arr.shape[0]),
                #          max((im_arr.shape[1] - self.patch_size[1]) / 2, 0):
                #          min(im_arr.shape[1] - (im_arr.shape[1] - self.patch_size[1]) / 2, im_arr.shape[1]),
                #          max((im_arr.shape[2] - self.patch_size[2]) / 2, 0):
                #          min(im_arr.shape[2] - (im_arr.shape[2] - self.patch_size[2]) / 2, im_arr.shape[2])]
                img_tr.append(im_arr)

            img_tr = np.array(img_tr)

            img_tr, mask_tr = self.augment(img_tr, mask_tr, b_aug=b_aug)

            return img_tr, mask_tr

    def __len__(self):
        return self.origin_size * (self.augment_scale + 1)

    def augment(self, img_tr, mask_tr, b_aug=True):
        # channel x X x Y x Z
        input_dim = len(img_tr.shape)
        # logging.debug("augment input dim {}".format(input_dim))

        tensor_img, tensor_mask = [], []
        #################################
        pre_transform = ts.Compose([
            ts.PadNumpy(size=self.scale_size),
            ts.ToTensor(),
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float'])])

        for kw_idx in xrange(len(self.keywords)):
            if kw_idx == 0:
                tmp_tensor_img, tmp_tensor_mask = pre_transform(img_tr[kw_idx, ...], mask_tr[kw_idx, ...])
                tensor_img.append(tmp_tensor_img)
                tensor_mask.append(tmp_tensor_mask)
            else:
                tmp_tensor_img = pre_transform(img_tr[kw_idx, ...])
                tensor_img.append(tmp_tensor_img)

        # cat
        tensor_img = torch.stack(tensor_img)
        tensor_mask = torch.stack(tensor_mask)
        # logging.debug("tensor image shape {}, tensor mask shape {}".format(tensor_img.shape, tensor_mask.shape))

        if b_aug:
            #################################
            # augment !
            # 4D image NOT supported...
            # split 4D image into 3D...
            # flip
            if random.random() < self.random_flip_prob:
                h_flip_op = ts.RandomFlip(h=True, v=False, p=1.0)
                for kw_idx in xrange(len(self.keywords)):
                    if kw_idx == 0:
                        tensor_img[kw_idx, ...], tensor_mask[kw_idx, ...] = h_flip_op(tensor_img[kw_idx, ...],
                                                                                      tensor_mask[kw_idx, ...])
                    else:
                        tensor_img[kw_idx, ...] = h_flip_op(tensor_img[kw_idx, ...])

            # logging.debug("tensor image shape {}, tensor mask shape {}".format(tensor_img.shape, tensor_mask.shape))

            if random.random() < self.random_flip_prob:
                v_flip_op = ts.RandomFlip(h=False, v=True, p=1.0)
                for kw_idx in xrange(len(self.keywords)):
                    if kw_idx == 0:
                        tensor_img[kw_idx, ...], tensor_mask[kw_idx, ...] = v_flip_op(tensor_img[kw_idx, ...],
                                                                                      tensor_mask[kw_idx, ...])
                    else:
                        tensor_img[kw_idx, ...] = v_flip_op(tensor_img[kw_idx, ...])
            # logging.debug("tensor image shape {}, tensor mask shape {}".format(tensor_img.shape, tensor_mask.shape))

            # affine
            affine_mat_op = ts.RandomAffine(rotation_range=self.affine_rotate_val,
                                            translation_range=self.affine_shift_val,
                                            zoom_range=self.affine_scale_val, lazy=True)
            # cal matrix using #0 element
            affine_mat = affine_mat_op(tensor_img[0, ...], tensor_mask[0, ...])

            for kw_idx in xrange(len(self.keywords)):
                affine_op = ts.Affine(tform_matrix=affine_mat, interp=('bilinear', 'nearest'))
                if kw_idx == 0:
                    tensor_img[kw_idx, ...], tensor_mask[kw_idx, ...] = affine_op(tensor_img[kw_idx, ...],
                                                                                  tensor_mask[kw_idx, ...])
                else:
                    tensor_img[kw_idx, ...] = affine_op(tensor_img[kw_idx, ...])

        #
        # DUE to channel permute operations, the Tensor Object must be
        # cleared since the original size have changed
        #
        tensor_img_out, tensor_mask_out = [], []
        mid_transform = ts.Compose([ts.NormalizeMedic(norm_flag=(True, False)), ts.ChannelsLast()])
        for kw_idx in xrange(len(self.keywords)):
            if kw_idx == 0:
                tmp_tensor_img, tmp_tensor_mask = mid_transform(tensor_img[kw_idx, ...], tensor_mask[kw_idx, ...])
                tensor_img_out.append(tmp_tensor_img)

                # duplicate the mask since the last_transform needs
                # image and mask share same sizes
                for expand_times in xrange(len(self.keywords)):
                    tensor_mask_out.append(tmp_tensor_mask)
            else:
                tmp_tensor_img = mid_transform(tensor_img[kw_idx, ...])
                tensor_img_out.append(tmp_tensor_img)

        # cat
        tensor_img_out = torch.stack(tensor_img_out)
        tensor_mask_out = torch.stack(tensor_mask_out)
        # logging.debug("tensor image shape {}, tensor mask shape {}".format(tensor_img.shape, tensor_mask.shape))

        # 3D image need adding channel
        if input_dim == 3:
            logging.debug("adding channels at 0 axis")
            ch_add_op = ts.AddChannel(axis=0)
            tensor_img_out = ch_add_op(tensor_img_out)
            tensor_mask_out = ch_add_op(tensor_mask_out)

        #################################
        # crop
        # 4D image supported... channel first
        if b_aug:
            last_transform = ts.Compose([
                ts.RandomCrop(size=self.patch_size),
                ts.TypeCast(['float', 'long'])])
        else:
            last_transform = ts.Compose([
                ts.SpecialCrop(size=self.patch_size, crop_type=0),
                ts.TypeCast(['float', 'long'])])

        tensor_img_out, tensor_mask_out = last_transform(tensor_img_out, tensor_mask_out)
        # logging.debug("augment, tensor image shape {}, tensor mask shape {}".format( \
        # tensor_img_out.shape, tensor_mask_out.shape))

        return tensor_img_out, tensor_mask_out
        # assert np.all(tensor_mask_out[0, ...].numpy() == tensor_mask_out[1, ...].numpy()) and \
        #        np.all((tensor_mask_out[2, ...].numpy() == tensor_mask_out[3, ...].numpy()))
        # return torch.unsqueeze(tensor_img_out[0, ...], 0), \
        #        torch.unsqueeze(tensor_mask_out[2, ...], 0)
