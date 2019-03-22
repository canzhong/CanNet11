from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ResNextB(nn.Module):

    
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, widen_factor):
        """
            Initial Constructor
            
            Argument Paramters :
                in_channels : input channel dimension
                out_channels : output channel dimension
                stride : convolution stride
                cardinality : num of convolutional groups in a layer
                base_width : base number of channels in each group
                widen_factor : factor to reduce the input dimension before convolution operator
        """
        super(ResNextB, self).__init__()
        #width_ratio will be decreased for computational purposes.
        #TODO : Determine the complexity differenice when using width_ratio = 32 versus 64
        width_ratio = out_channels / (widen_factor * 32.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)
        
        #Identity mapping shortcut for residual. Computationally cheaper with 1x1 convolutions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))
        
    #Forward pass through this section of the network
    def forward(self, x):
        y = self.conv_reduce.forward(x)
        y = F.leaky_relu(self.bn_reduce.forward(y), 0.1, inplace=True)
        y = self.conv_conv.forward(y)
        y = F.leaky_relu(self.bn.forward(y), 0.1, inplace=True)
        y = self.conv_expand.forward(y)
        y = self.bn_expand.forward(y)
        residual = self.shortcut.forward(x)
        #Try initial network with standard relu, second nn will use leaky relu
        return F.leaky_relu(residual + y, 0.1, inplace=True)


class Bottleneck(nn.Module):
    outchannel_ratio = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, (planes*1), kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d((planes*1))
        self.conv3 = nn.Conv2d((planes*1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.leakyrelu(out)
        out = self.conv2(out)
 
        out = self.bn3(out)
        out = self.leakyrelu(out)
        out = self.conv3(out)

        out = self.bn4(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class PyramidNet(nn.Module):
        
    def __init__(self, depth, alpha, num_classes, bottleneck=True):
        super(PyramidNet, self).__init__()   	
        
        
        self.inplanes = 10

        n = int((depth - 2) / 9)
        block = Bottleneck
        

        self.addrate = alpha / (3*n*1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = nn.Conv2d(1, self.input_featuremap_dim, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim 
        self.layer1 = self.pyramidal_make_layer(block, n)
        self.layer2 = self.pyramidal_make_layer(block, n, stride=1)	
        self.layer3 = self.pyramidal_make_layer(block, n, stride=1)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.leakyrelu_final = nn.LeakyReLU(0.2, inplace=True)
        self.avgpool = nn.AvgPool2d(10)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def pyramidal_make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:# or self.inplanes != int(round(self.featuremap_dim)) * block.outchannel_ratio:
            downsample = nn.AvgPool2d((2,2), stride = (2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim  = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

        return nn.Sequential(*layers)

    def forward(self, x):
                
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_final(x)
        x = self.leakyrelu_final(x)
        x = self.avgpool(x)
        x = self.fc(x)
    
        return x


class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        p = self.pose(x)
        a = self.a(x)
        a = self.sigmoid(a) 
        out = torch.cat([p, a], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out


class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=32, C=32, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        self.ln_2pi = torch.cuda.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) \
                    - torch.log(sigma_sq.sqrt()) \
                    - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
        return r

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        r = torch.cuda.FloatTensor(b, B, C).fill_(1./C)
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
                r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

        return mu, a_out

    def add_pathes(self, x, B, K, psize, stride):
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
        """
        b, h, w, c = x.shape
        assert h == w
        assert c == B*(psize+1)
        oh = ow = int((h - K + 1) / stride)
        idxs = [[(h_idx + k_idx) \
                for k_idx in range(0, K)] \
                for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
        """
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = 1. * torch.arange(h) / h
        coor_h = torch.cuda.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
        coor_w = torch.cuda.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x):
        b, h, w, c = x.shape
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            p_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            v = self.transform_view(p_in, self.weights, self.C, self.P)

            # em_routing
            p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            out = torch.cat([p_out, a_out], dim=3)
        else:
            assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            p_in = p_in.view(b, h*w*self.B, self.psize)
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            a_in = a_in.view(b, h*w*self.B, 1)

            # transform view
            v = self.transform_view(p_in, self.weights, self.C, self.P, self.w_shared)

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)

            # em_routing
            _, out = self.caps_em_routing(v, a_in, self.C, self.eps)

        return out


class CanNet(nn.Module):
    """A network with one ReLU convolutional layer followed by
    a primary convolutional capsule layer and two more convolutional capsule layers.

    Suppose image shape is 28x28x1, the feature maps change as follows:
    1. ReLU Conv1
        (_, 1, 28, 28) -> 5x5 filters, 32 out channels, stride 2 with padding
        x -> (_, 32, 14, 14)
    2. PrimaryCaps
        (_, 32, 14, 14) -> 1x1 filter, 32 out capsules, stride 1, no padding
        x -> pose: (_, 14, 14, 32x4x4), activation: (_, 14, 14, 32)
    3. ConvCaps1
        (_, 14, 14, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 2, no padding
        x -> pose: (_, 6, 6, 32x4x4), activation: (_, 6, 6, 32)
    4. ConvCaps2
        (_, 6, 6, 32x(4x4+1)) -> 3x3 filters, 32 out capsules, stride 1, no padding
        x -> pose: (_, 4, 4, 32x4x4), activation: (_, 4, 4, 32)
    5. ClassCaps
        (_, 4, 4, 32x(4x4+1)) -> 1x1 conv, 10 out capsules
        x -> pose: (_, 10x4x4), activation: (_, 10)

        Note that ClassCaps only outputs activation for each class

    Args:
        A: output channels of normal conv
        B: output channels of primary caps
        C: output channels of 1st conv caps
        D: output channels of 2nd conv caps
        E: output channels of class caps (i.e. number of classes)
        K: kernel of conv caps
        P: size of square pose matrix
        iters: number of EM iterations
        ...
    """
    def __init__(self, A=6, B=12, C=6, D=16, E=10, K=3, P=4, iters=3):
        super(CanNet, self).__init__()
	
        self.resnextb = ResNextB(in_channels=1, out_channels=A, stride=2, cardinality=4, base_width=32, widen_factor=2)
        self.primary_caps = PrimaryCaps(A, B, 1, P, stride=1)
        self.conv_caps1 = ConvCaps(B, C, K, P, stride=2, iters=iters)
        self.conv_caps2 = ConvCaps(C, D, K, P, stride=1, iters=iters)
        self.class_caps = ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)
        self.convt = nn.Conv2d(1, 1 , kernel_size=1, stride=1, padding=1, bias=True)
        self.pyramidblock = PyramidNet(depth=11, alpha=48, num_classes=10)


    def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
        s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
        scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
        return scale * vectors
    def forward(self, x):
        x = self.resnextb.forward(x)
        x = self.primary_caps(x)
        x = self.conv_caps1(x)
        x = self.conv_caps2(x)
        x = self.class_caps(x)
        x = squash(x)
	 
      #  x = x.view((x.size(0),-1))
        x = self.convt(x) 
        x = self.pyramidblock(x)

        return x


def capsules(**kwargs):
    """Constructs a CapsNet model.
    """
    model = CanNet(**kwargs)
    return model


'''
TEST
Run this code with:
```
python -m capsules.py
```
'''
if __name__ == '__main__':
    model = capsules(E=10)
    print(model)
