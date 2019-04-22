
# Matrix Capsules with EM Routing using ResBlock downsampling and ODEBlock feature sampling
A Pytorch implementation variation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) using 2 ResBlocks to downsample data followed by 1 ODEBlock for feature map extraction.

Architecture :

nn.Conv2d(3, 256, 3, 1)

ResBlock(256, 256, stride=2, downsample=conv1x1(256, 256, 2))

ResBlock(256, 256, stride=2, downsample=conv1x1(256, 256, 2))

ODEBlock(ODEfunc(256))

PrimaryCaps(A, B, 1, P, stride=1)

ConvCaps(B, C, K, P, stride=2, iters=iters)

ConvCaps(C, D, K, P, stride=1, iters=iters)

ConvCaps(D, E, 1, P, stride=1, iters=iters, coor_add=True, w_shared=True)

A = 256, B = 32, C = 48, D = 64, E = 10

## CIFAR10 experiments

The experiments are conducted on Nvidia Tesla V100.
Specific setting is `lr=0.0075`, `batch_size=12`, `weight_decay=0`, SGD optimizer

Following is the result after 76 epochs training:

| Arch | Iters | Coord Add | Loss | BN | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=256, B=32, C=48, D=64 | 3 | Y | Spread    | Y | 76.12 |

The training time for a 12 batch is around `0.0417s`.

CanNet11 : Paramter size = 4496660 with 76.12 % accuracy 
(Comparable architecture) DCNET : Parameter size = 16000000 with 82 % accuracy

Coding error, accidently forgot to tab the 'save' component within the epochs for loop. Therefore, the model was never going to save until it ran all epochs which I had originally set to 300. (I really was new at this so I did not know what a good hyperparameter was but I knew I could exit early if it was overfitting, not generalizing, decrease accuracy.) I early stopped after recognizing the situation because it was going to be financially expensive to let it run to completion and I wasn't even sure if it was going to be a better model after 300 models. 

## Reference
The research done is solely for non-profit academic purposes. Code from several repos were altered and pieced together to create an architecture compatible with the experiment hypothesis.

Capsule components were inspired from https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch
ResBlock and ODEBlock components were inspired from https://github.com/rtqichen/torchdiffeq/tree/master/examples
