import torch
from torch.autograd import Variable
import torch.optim as optim
import logging

from collections import OrderedDict
# import utils.util as util
from .base_model import BaseModel
from .networks import get_network
from .layers.loss import *
from .networks_other import get_scheduler, print_network, benchmark_fp_bp_time
from .utils import segmentation_stats, get_optimizer, get_criterion, tensor2im
from .networks.utils import HookBasedFeatureExtractor
from torch.nn import DataParallel
import torch.nn.functional as F


class FeedForwardSegmentation(BaseModel):

    def name(self):
        return 'FeedForwardSegmentation'

    def initialize(self, opts, **kwargs):
        BaseModel.initialize(self, opts, **kwargs)
        self.isTrain = opts.isTrain

        # define network input and output pars
        self.input = None
        self.target = None
        self.tensor_dim = opts.tensor_dim

        # load/define networks
        self.net = get_network(opts.model_type, n_classes=opts.output_nc,
                               in_channels=opts.input_nc, nonlocal_mode=opts.nonlocal_mode,
                               tensor_dim=opts.tensor_dim, feature_scale=opts.feature_scale,
                               attention_dsample=opts.attention_dsample)
        if self.use_cuda:
            logging.debug("setting up dataparallel on {} gpus".format(self.gpu_ids))
            self.net = self.net.cuda()
            self.net = DataParallel(self.net, device_ids=self.gpu_ids)

            # for this_net in rep_net:
            #     for name, param in this_net.named_parameters():
            #         logging.debug("param {} on GPU {}".format(name, param.get_device()))

            for name, param in self.net.named_parameters():
                if not param.is_cuda:
                    logging.debug("!!!!!! {} not on GPU !!!!!!".format(name))
                # else:
                #     logging.debug("param {} on GPU {}".format(name, param.get_device()))

        # load the model if a path is specified or it is in inference mode
        if not self.isTrain or opts.continue_train:
            self.path_pre_trained_model = opts.path_pre_trained_model
            if self.path_pre_trained_model:
                self.load_network_from_path(self.net, self.path_pre_trained_model, strict=False)
                self.which_epoch = int(0)
            else:
                self.which_epoch = opts.which_epoch
                self.load_network(self.net, 'S', self.which_epoch)

        # training objective
        if self.isTrain:
            self.criterion = get_criterion(opts)
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_S = get_optimizer(opts, self.net.parameters())
            self.optimizers.append(self.optimizer_S)

            if self.use_cuda:
                # self.criterion = DataParallel(self.criterion)
                self.criterion = self.criterion.cuda()

            # logging.debug the network details
            if kwargs.get('verbose', False):
                logging.debug('Network is initialized')
                print_network(self.net)

    def set_scheduler(self, train_opt):
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, train_opt))
            logging.debug('Scheduler is added for optimiser {0}'.format(optimizer))

    def set_input(self, *inputs):
        # self.input.resize_(inputs[0].size()).copy_(inputs[0])
        for idx, _input in enumerate(inputs):
            # If it's a 5D array and 2D model then (B x C x H x W x Z) -> (BZ x C x H x W)
            bs = _input.size()
            if (self.tensor_dim == '2D') and (len(bs) > 4):
                _input = _input.permute(0, 4, 1, 2, 3).contiguous().view(bs[0] * bs[4], bs[1], bs[2], bs[3])

            # Define that it's a cuda array
            if idx == 0:
                if self.use_cuda:
                    self.input = _input.cuda()
                    # logging.debug("moving input {} to cuda".format(self.input.size()))
                else:
                    self.input = _input
                    # logging.debug("moving input {} to cpu".format(self.input.size()))

            elif idx == 1:
                if self.use_cuda:
                    self.target = Variable(_input.cuda())
                    # logging.debug("moving target {} to cuda".format(self.target.size()))
                else:
                    self.target = Variable(_input)
                    # logging.debug("moving target {} to cpu".format(self.target.size()))

                assert self.input.size() == self.target.size()

        # print "input size {} on device {}".format(self.input.size(), self.input.get_device())
        # print "target size {} on device {}".format(self.target.size(), self.target.data.get_device())

    def forward(self, split):
        if split == 'train':
            # logging.debug("forward data size {} on {}".format(self.input.size(), self.input.get_device()))
            self.prediction = self.net(Variable(self.input))
            # self.prediction = self.data_parallel2(module=self.net, inputs=Variable(self.input), device_ids=self.gpu_ids)
            # self.prediction = self.data_parallel(module=self.net, inputs=Variable(self.input), device_ids=self.gpu_ids)

        elif split == 'test':
            self.prediction = self.net(Variable(self.input, volatile=True))
            # Apply a softmax and return a segmentation map
            # self.logits = self.net.apply_argmax_softmax(self.prediction)
            self.logits = F.softmax(self.prediction, dim=1)
            self.pred_seg = self.logits.data.max(1)[1].unsqueeze(1)

    # def scatter_kwargs(self, inputs, kwargs, target_gpus, dim=0):
    #     r"""Scatter with support for kwargs dictionary"""
    #     inputs = nn.parallel.scatter(inputs, target_gpus, dim) if inputs else []
    #     kwargs = nn.parallel.scatter(kwargs, target_gpus, dim) if kwargs else []
    #     if len(inputs) < len(kwargs):
    #         inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    #     elif len(kwargs) < len(inputs):
    #         kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    #     inputs = tuple(inputs)
    #     kwargs = tuple(kwargs)
    #     return inputs, kwargs
    #
    # def data_parallel2(self, module, inputs, device_ids, output_device=None):
    #     if not device_ids:
    #         return module(inputs)
    #
    #     if output_device is None:
    #         output_device = device_ids[0]
    #
    #     if not isinstance(inputs, tuple):
    #         inputs = (inputs,)
    #
    #     inputs = nn.parallel.scatter(inputs, device_ids)
    #     # logging.debug("scatter input length {}".format(len(inputs)))
    #     inputs = tuple(inputs)
    #
    #     replicas = nn.parallel.replicate(module, device_ids)
    #     replicas = replicas[:len(inputs)]
    #     # logging.debug("rep length {}".format(len(replicas)))
    #
    #     outputs = nn.parallel.parallel_apply(replicas, inputs, None, device_ids)
    #     return nn.parallel.gather(outputs, output_device)
    #
    # def data_parallel(self, module, inputs, device_ids, output_device=None):
    #     if not isinstance(inputs, tuple):
    #         inputs = (inputs,)
    #
    #     if device_ids is None:
    #         device_ids = list(range(torch.cuda.device_count()))
    #
    #     if output_device is None:
    #         output_device = device_ids[0]
    #
    #     inputs, module_kwargs = self.scatter_kwargs(inputs, None, device_ids, 0)
    #     if len(device_ids) == 1:
    #         return module(*inputs[0], **module_kwargs[0])
    #     used_device_ids = device_ids[:len(inputs)]
    #     replicas = nn.parallel.replicate(module, used_device_ids)
    #
    #     outputs = nn.parallel.parallel_apply(replicas, inputs, None, used_device_ids)
    #     return nn.parallel.gather(outputs, output_device, 0)

    def backward(self):
        self.loss_S = self.criterion(self.prediction, self.target)
        self.loss_S.backward()

    def optimize_parameters(self):
        self.net.train()
        self.forward(split='train')
        self.optimizer_S.zero_grad()
        self.backward()
        self.optimizer_S.step()

    # This function updates the network parameters every "accumulate_iters"
    def optimize_parameters_accumulate_grd(self, iteration):
        accumulate_iters = int(2)
        if iteration == 0:
            self.optimizer_S.zero_grad()
        self.net.train()
        self.forward(split='train')
        self.backward()

        if iteration % accumulate_iters == 0:
            self.optimizer_S.step()
            self.optimizer_S.zero_grad()

    def test(self):
        self.net.eval()
        self.forward(split='test')

    def validate(self):
        self.net.eval()
        self.forward(split='test')
        self.loss_S = self.criterion(self.prediction, self.target)

    def get_segmentation_stats(self):
        self.seg_scores, self.dice_score = segmentation_stats(self.prediction, self.target)
        seg_stats = [('Overall_Acc', self.seg_scores['overall_acc']), ('Mean_IOU', self.seg_scores['mean_iou'])]
        for class_id in range(self.dice_score.size):
            seg_stats.append(('Class_{}'.format(class_id), self.dice_score[class_id]))
        return OrderedDict(seg_stats)

    def get_current_errors(self):
        return OrderedDict([('Seg_Loss', self.loss_S.data[0])
                            ])

    def get_current_visuals(self):
        inp_img = tensor2im(self.input, 'img')
        seg_img = tensor2im(self.pred_seg, 'lbl')
        return OrderedDict([('out_S', seg_img), ('inp_S', inp_img)])

    def get_feature_maps(self, layer_name, upscale):
        feature_extractor = HookBasedFeatureExtractor(self.net, layer_name, upscale)
        return feature_extractor.forward(Variable(self.input))

    # returns the fp/bp times of the model
    def get_fp_bp_time(self, size=None):
        if size is None:
            size = (1, 1, 160, 160, 96)

        inp_array = Variable(torch.zeros(*size)).cuda()
        out_array = Variable(torch.zeros(*size)).cuda()
        fp, bp = benchmark_fp_bp_time(self.net, inp_array, out_array)

        bsize = size[0]
        return fp / float(bsize), bp / float(bsize)

    def save(self, epoch_label):
        self.save_network(self.net, 'S', epoch_label, self.gpu_ids)
