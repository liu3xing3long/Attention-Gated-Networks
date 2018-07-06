import torch
import torch.nn as nn
from models.layers.grid_attention_layer import GridAttentionBlock3D, GridAttentionBlock3D_TORR


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.mode = 'concatenation'
        self.block1 = GridAttentionBlock3D(in_channels=16, inter_channels=16, gating_channels=64, mode=mode,
                                   sub_sample_factor=(2, 2, 2))
        self.block2 = GridAttentionBlock3D(in_channels=16, inter_channels=16, gating_channels=1, mode=mode,
                                           sub_sample_factor=(2, 2, 2))

    def forward(self, img, gat):
        out, sigma = self.block1(img, gat)
        out, sigma = self.block2(out, sigma)

        return out, sigma


if __name__ == '__main__':
    from torch.autograd import Variable
    from torch.nn import DataParallel

    mode_list = ['concatenation']
    # mode_list = ['concatenation_sigmoid']

    for mode in mode_list:

        img = Variable(torch.rand(8, 16, 10, 10, 10))
        gat = Variable(torch.rand(8, 64, 4, 4, 4))

        net = Net()

        net = DataParallel(net, device_ids=[0, 1, 2, 3])
        net = net.cuda()

        img = img.cuda()
        gat = gat.cuda()

        for epoch in xrange(10000):
            print "epoch {}".format(epoch)
            out, sigma = net(img, gat)
            print out.size()
