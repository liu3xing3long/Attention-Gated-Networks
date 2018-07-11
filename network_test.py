import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleGridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(SimpleGridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.upsample_mode = 'trilinear'

        # Output transform
        self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                          padding=0),
                nn.BatchNorm3d(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                               bias=False)
        self.phi = nn.Conv3d(in_channels=self.gating_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                             bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        # if mode == 'concatenation':
        #     self.operation_function = self._concatenation
        # elif mode == 'concatenation_debug':
        #     self.operation_function = self._concatenation_debug
        # elif mode == 'concatenation_residual':
        #     self.operation_function = self._concatenation_residual
        # else:
        #     raise NotImplementedError('Unknown operation function.')

        # self.W = nn.DataParallel(self.W)
        # self.phi = nn.DataParallel(self.phi)
        # self.psi = nn.DataParallel(self.psi)
        # self.theta = nn.DataParallel(self.theta)

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''
        output = None
        if mode == 'concatenation':
            output = self._concatenation(x, g)
        elif mode == 'concatenation_debug':
            output = self._concatenation_debug(x, g)
        elif mode == 'concatenation_residual':
            output = self._concatenation_residual(x, g)
        else:
            raise NotImplementedError('Unknown operation function.')

        return output
        # input_device_id = x.data.get_device()
        # x = x.cuda(0)
        # g = g.cuda(0)
        # for name, param in self.named_parameters():
        #     print "name {}, on device {}".format(name, param.get_device())
        # print "before operation on device {}".format(input_device_id)
        # output = self.operation_function(x, g)
        # output = self._concatenation(x, g)

        # theta_x = self.theta(x)
        # theta_x_size = theta_x.size()
        # phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        # output = (theta_x, phi_g)

        # output = (output[0].cuda(input_device_id), output[1].cuda(input_device_id))
        # print "after operation output0 device {}".format(output[0].data.get_device())
        # print "after operation output1 device {}".format(output[1].data.get_device())

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        input_device_id = x.data.get_device()
        print "x on device {}".format(input_device_id)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        # self.theta = self.theta.cuda(input_device_id)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        print "theta_x on device {}".format(theta_x.data.get_device())

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        print "phi_g on device {}".format(phi_g.data.get_device())

        f = F.relu(theta_x + phi_g, inplace=True)
        print "f on device {}".format(f.data.get_device())

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))
        print "sigm_psi_f on device {}".format(sigm_psi_f.data.get_device())

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        print "sigm_psi_f on device {}, x on device {}".format(sigm_psi_f.data.get_device(), x.data.get_device())

        # OUTPUT operate with ORIGINAL data
        # should pay super attention, since OUTPUT is on output_device
        # while ORIGINAL data is distibuted by DataParallel
        y = sigm_psi_f.expand_as(x) * x
        print "y on device {}".format(y.data.get_device())

        W_y = self.W(y)
        print "W_y on device {}".format(W_y.data.get_device())

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class SimpleGridAttentionBlock3D(SimpleGridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(SimpleGridAttentionBlock3D, self).__init__(in_channels,
                                                         inter_channels=inter_channels,
                                                         gating_channels=gating_channels,
                                                         dimension=3, mode=mode,
                                                         sub_sample_factor=sub_sample_factor)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        in_size = 16
        gate_size = 64
        inter_size = 16
        sample_factor = (2, 2, 2)
        mode = 'concatenation'

        self.gate_block_1 = SimpleGridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                       inter_channels=inter_size, mode=mode,
                                                       sub_sample_factor=sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True))

    def forward(self, img, gat):
        gate_1, attention_1 = self.gate_block_1(img, gat)
        return self.combine_gates(gate_1), attention_1
        return gate_1, attention_1


class SimpleNet2(nn.Module):
    def __init__(self):
        super(SimpleNet2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3)

    def forward(self, img, gat):
        print "before img on device {}".format(img.data.get_device())
        print "before gat on device {}".format(gat.data.get_device())

        output1 = self.conv1(img)
        output2 = self.conv2(gat)
        print "after output1 on device {}".format(output1.data.get_device())
        print "after output2 on device {}".format(output2.data.get_device())

        return output1, output2


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch.nn import DataParallel
    import os

    mode_list = ['concatenation']
    # mode_list = ['concatenation_sigmoid']

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    for mode in mode_list:

        img = Variable(torch.rand(8, 16, 10, 10, 10))
        gat = Variable(torch.rand(8, 64, 4, 4, 4))

        net = SimpleNet()
        # net = SimpleNet2()

        devices = [0, 1, 2, 3]
        net = net.cuda()
        net = DataParallel(net, device_ids=devices)

        img = img.cuda()
        gat = gat.cuda()

        for epoch in xrange(4):
            print "epoch {}".format(epoch)
            out, sigma = net(img, gat)
            print out.size()
