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


class CMR3DDatasetBRATS(data.Dataset):
    def __init__(self, dataset_folder, subset_folder, label_folder, data_subsets, keywords=["P1", "1", "flair"],
                 mode='train', transform=None, augment_scale=0):
        super(CMR3DDatasetBRATS, self).__init__()

        self.rawim = {}
        self.rawmask = []
        self.keywords = keywords
        self.train_mode = mode == 'train'
        self.test_mode = mode == 'test'
        self.val_mode = mode == 'val'
        self.data_path_map = {}
        self.data_name_vec = []
        self.augment_scale = augment_scale
        self.origin_size = 0

        # image_dir = os.path.join(root_dir, split, 'image')
        # target_dir = os.path.join(root_dir, split, 'label')
        # self.image_filenames = sorted([os.path.join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])
        # self.target_filenames = sorted([os.path.join(target_dir, x) for x in listdir(target_dir) if is_image_file(x)])

        # read csv settings
        csv_folder = os.path.join(dataset_folder, label_folder)

        # since we have LGG and HGG two folders...
        # data_name ---> data_folder_path
        self.data_path_map = {}
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

            # analyze data_path_map
        # for keywd in self.keywords:
        #     self.src_mean[keywd] = 0
        #     self.src_std[keywd] = 0

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

            # # crop the train image
            # if self.crop_train_image:
            #     zz, yy, xx = np.where(mask_arr == 1)
            #     minzz, maxzz = min(zz), max(zz)
            #     minyy, maxyy = min(yy), max(yy)
            #     minxx, maxxx = min(xx), max(xx)
            #
            #     if self.train_mode or self.val_mode:
            #         # for mask, padding is 0, so do not need re-cal the original array
            #         mask_arr = mask_arr[minzz: maxzz + 1, minyy: maxyy + 1, minxx: maxxx + 1]

            # pad 0 to mask
            # mask_arr, nzhw, padding = split_image(mask_arr, self.side_len, pad_val=0)

            # process all masks and nzhws
            # self.nzhw.append(nzhw)
            self.rawmask.append(mask_arr)

            # if self.crop_train_image:
            #     # re-calculate the  xx,yy,zz according to pad
            #     minzz -= padding[0][0]
            #     maxzz += padding[0][1]
            #     minyy -= padding[1][0]
            #     maxyy += padding[1][1]
            #     minxx -= padding[2][0]
            #     maxxx += padding[2][1]
            #
            #     minzz = max(minzz, 0)
            #     minyy = max(minyy, 0)
            #     minxx = max(minxx, 0)
            #
            #     maxzz = min(maxzz, mask_arr_d)
            #     maxyy = min(maxyy, mask_arr_h)
            #     maxxx = min(maxxx, mask_arr_w)
            #
            #     # logging.debug("image {}, valid voxels {}, valid z,y,x ({}, {}, {}) -> ({}, {}, {})".format(
            #     #         data_name, np.sum(mask_arr), minzz, minyy, minxx, maxzz, maxyy, maxxx))
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

                # # cal the std and mean after image split
                # if self.train_mode:
                #     this_mean = np.mean(im_arr)
                #     this_std = np.std(im_arr)

                # logging.debug("image {}, mod {}, mean {}, std {}".format(data_name, keywd, this_mean, this_std))

                # self.src_mean[keywd] += this_mean
                # self.src_std[keywd] += this_std

                # if self.train_mode or self.val_mode:
                # if self.crop_train_image:
                #     # ONLY in train mode should we crop the image
                #     im_arr = im_arr[minzz: maxzz + 1, minyy: maxyy + 1, minxx: maxxx + 1]

                # im_arr, nzhw, padding = split_image(im_arr, self.side_len)

                # append the images
                self.rawim[keywd].append(im_arr)

        for keywd in self.keywords:
            # transform to np array
            self.rawim[keywd] = np.array(self.rawim[keywd])
            # only in train_mode or val_mode, we concate all data
            # if self.train_mode or self.val_mode:
            #     self.rawim[keywd] = np.concatenate(self.rawim[keywd])

            # if self.train_mode:
            #     self.src_mean[keywd] /= len(self.data_path_map)
            #     self.src_std[keywd] /= len(self.data_path_map)
            # else:
            #     self.src_mean[keywd] = src_mean[keywd]
            #     self.src_std[keywd] = src_std[keywd]
            #
            # if normalize:
            #     # normalize image
            #     self.rawim[keywd] -= self.src_mean[keywd]
            #     self.rawim[keywd] /= self.src_std[keywd]
        # logging.info("image mean: {}, image std: {}, image normalized".format(self.src_mean, self.src_std))

        self.rawmask = np.array(self.rawmask)
        # self.nzhw = np.array(self.nzhw)

        # only in train_mode or val_mode, we concate all data
        # if self.train_mode or self.val_mode:
        #     self.rawmask = np.concatenate(self.rawmask)

        # self.tar_mean = np.mean(self.rawmask)
        # self.tar_std = np.std(self.rawmask)

        # re-cal mean
        # self.tar_mean = 0
        # for keywd in self.keywords:
        #     self.tar_mean += np.sum(self.rawmask) / np.sum(self.rawim[keywd] > 0)
        # self.tar_mean /= len(self.keywords)

        # logging.info("mask mean: {}, mask std: {}".format(self.tar_mean, self.tar_std))

        # output data shapes
        for keywd in self.keywords:
            logging.debug("data shape for modality {} in {} mode is {}".format(keywd, mode, self.rawim[keywd].shape))
        logging.debug("mask shape in {} mode is {}".format(mode, self.rawmask.shape))

        # debug infor output
        logging.debug("total voxels {}, valid voxels {}, rate {}".format(self.rawmask.size, np.sum(self.rawmask),
                                                                         np.sum(self.rawmask) / self.rawmask.size))

        # data augmentation
        self.transform = transform
        self.origin_size = self.rawmask.shape[0]

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        b_aug = True
        if index >= self.origin_size:
            index = index % self.origin_size
            b_aug = True

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

        # logging.debug("during get item, image shape {}, mask shape {}".format(img_tr.shape, mask_tr.shape))
        # handle exceptions
        check_exceptions(img_tr, mask_tr)
        if b_aug:
            if self.transform is not None:
                img_tr, mask_tr = self.transform(img_tr, mask_tr)

        return img_tr, mask_tr

    def __len__(self):
        return self.origin_size * (self.augment_scale + 1)
