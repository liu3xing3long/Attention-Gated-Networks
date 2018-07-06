import torch.utils.data as data
import numpy as np
import datetime

from os import listdir
import os
# from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file
import pandas as pd
import logging
import SimpleITK as sitk
from tqdm import tqdm
import torchsample.transforms as ts
from collections import OrderedDict


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

        self.origin_size = 0

        if aug_opts is None:
            self.preload = False
        else:
            self.preload = aug_opts.preload_data

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

                self.rawmask.append(mask_arr)

                #######################
                # reading real image data
                for keywd in self.keywords:
                    image_name = "{}_{}.nii.gz".format(data_name, keywd)
                    abs_image_path = os.path.join(data_real_folder, image_name)
                    # logging.debug("reading image from {}".format(abs_image_path))

                    im = sitk.ReadImage(os.path.join(data_real_folder, image_name))
                    im_arr = sitk.GetArrayFromImage(im)
                    im_arr = im_arr.astype(np.float)
                    # z, y, x -> x, y, z
                    im_arr = np.transpose(im_arr, [2, 1, 0])

                    # append the images
                    self.rawim[keywd].append(im_arr)

            for keywd in self.keywords:
                # transform to np array
                self.rawim[keywd] = np.array(self.rawim[keywd])

            self.rawmask = np.array(self.rawmask)
            # output data shapes
            for keywd in self.keywords:
                logging.debug("data shape for modality {} in {} mode is {}".format(keywd, mode, self.rawim[keywd].shape))
            logging.debug("mask shape in {} mode is {}".format(mode, self.rawmask.shape))

            # debug infor output
            logging.debug("total voxels {}, valid voxels {}, rate {}".format(self.rawmask.size, np.sum(self.rawmask),
                                                                             np.sum(self.rawmask) / self.rawmask.size))
            self.origin_size = self.rawmask.shape[0]

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

            if len(img_tr.shape) == 4:
                img_tr = img_tr[0, ...]
            if len(mask_tr.shape) == 4:
                mask_tr = mask_tr[0, ...]

            # handle exceptions
            check_exceptions(img_tr, mask_tr)
            if b_aug:
                if self.transform is not None:
                    img_tr, mask_tr = self.transform(img_tr, mask_tr)
            else:
                transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                        ts.ToTensor(),
                                        ts.ChannelsFirst(),
                                        ts.TypeCast(['float', 'float']),
                                        ts.NormalizeMedic(norm_flag=(True, False)),
                                        ts.ChannelsLast(),
                                        ts.AddChannel(axis=0),
                                        ts.RandomCrop(size=self.patch_size),
                                        ts.TypeCast(['float', 'long'])])
                img_tr, mask_tr = transform(img_tr, mask_tr)

            return img_tr, mask_tr

        else:
            #######################
            # reading mask data
            data_name = self.data_name_vec[index]
            data_real_folder = self.data_path_map[data_name]
            mask_name = "{}_{}.nii.gz".format(data_name, "seg")
            abs_image_path = os.path.join(data_real_folder, mask_name)
            # logging.debug("reading mask from {}".format(abs_image_path))

            img_tr, mask_tr = None, None

            mask = sitk.ReadImage(abs_image_path)
            mask_tr = sitk.GetArrayFromImage(mask)
            mask_arr_d, mask_arr_h, mask_arr_w = mask_tr.shape

            mask_tr = mask_tr.astype(np.float)
            # z, y, x -> x, y, z
            mask_tr = np.transpose(mask_tr, [2, 1, 0])

            mask_tr[mask_tr > 0] = 1

            # update multi
            # new_mask_arr = np.zeros(mask_tr.shape)
            # new_mask_arr[mask_tr > 0] = 1
            # new_mask_arr[mask_tr == 1] = 3
            # new_mask_arr[mask_tr == 4] = 2
            # mask_tr = new_mask_arr

            #######################
            # reading real image data
            for keywd in self.keywords:
                image_name = "{}_{}.nii.gz".format(data_name, keywd)
                abs_image_path = os.path.join(data_real_folder, image_name)
                # logging.debug("reading image from {}".format(abs_image_path))
                im = sitk.ReadImage(abs_image_path)
                img_tr = sitk.GetArrayFromImage(im)
                img_tr = img_tr.astype(np.float)
                # z, y, x -> x, y, z
                img_tr = np.transpose(img_tr, [2, 1, 0])

            if len(img_tr.shape) == 4:
                img_tr = img_tr[0, ...]
            if len(mask_tr.shape) == 4:
                mask_tr = mask_tr[0, ...]

            # handle exceptions
            check_exceptions(img_tr, mask_tr)

            # logging.debug("mask max {}, min {}, mean {}".format(np.max(mask_tr),  np.min(mask_tr), np.mean(mask_tr)))
            if self.transform is not None:
                img_tr, mask_tr = self.transform(img_tr, mask_tr)

            else:
                transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                        ts.ToTensor(),
                                        ts.ChannelsFirst(),
                                        ts.TypeCast(['float', 'float']),
                                        ts.NormalizeMedic(norm_flag=(True, False)),
                                        ts.ChannelsLast(),
                                        ts.AddChannel(axis=0),
                                        ts.RandomCrop(size=self.patch_size),
                                        ts.TypeCast(['float', 'long'])])
                img_tr, mask_tr = transform(img_tr, mask_tr)

            return img_tr, mask_tr

    def __len__(self):
        return self.origin_size * (self.augment_scale + 1)
