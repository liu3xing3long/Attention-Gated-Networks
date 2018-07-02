import numpy
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model
import logging
import time
import os
import numpy as np


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}{:02}'.format(
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)


def init_logging(logFilename):
    """
    Init for logging
    """
    logging.basicConfig(
            level=logging.DEBUG,
            format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=logFilename,
            filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(arguments):
    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training
    model_opts = json_opts.model

    # Architecture type
    arch_type = train_opts.arch_type

    MODEL_TYPE = model_opts.model_type
    output_path = "{}-{}-{}".format("./output/BRATS", MODEL_TYPE, datestr())
    # make dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    init_logging(os.path.join(output_path, "main3d.log"))

    # split used modality
    modality = [mod for mod in train_opts.modality.split()]

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # data paths
    DATA_FOLDER = ds_path
    SUBSET_FOLDERS = ["HGG", "LGG"]
    DATA_LABEL_FOLDER = "labels"

    # subset split
    subset_train = train_opts.subset_train
    subset_test = train_opts.subset_test
    subset_val = train_opts.subset_val
    if subset_train is None:
        subset_train = np.arange(15)
        subset_train = [x for x in subset_train if x not in args.subset_test]
        subset_train = [x for x in subset_train if x not in args.subset_val]

    logging.debug('train: {}, val: {}, test: {}'.format(subset_train, subset_val, subset_test))
    logging.debug('using modality {}'.format(modality))

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(DATA_FOLDER, SUBSET_FOLDERS, DATA_LABEL_FOLDER, subset_train,
                             keywords=modality, mode='train', transform=ds_transform['train'])
    valid_dataset = ds_class(DATA_FOLDER, SUBSET_FOLDERS, DATA_LABEL_FOLDER, subset_val,
                             keywords=modality, mode='val', transform=ds_transform['valid'])
    test_dataset = ds_class(DATA_FOLDER, SUBSET_FOLDERS, DATA_LABEL_FOLDER, subset_val,
                            keywords=modality, mode='test',  transform=ds_transform['valid'])

    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))

        # Training Iterations
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            # model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
            for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()

                merged_dict = {}
                merged_dict.update(errors)
                merged_dict.update(stats)
                # error_logger.update({**errors, **stats}, split=split)
                error_logger.update(merged_dict, split=split)

                # Visualise predictions
                visuals = model.get_current_visuals()
                visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        error_logger.reset()

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
            model.save(epoch)

        # Update the model learning rate
        model.update_learning_rate()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config', help='training config file', required=True)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
