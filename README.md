## Introduction
This repo shows how to use Theano with multiple GPUs, with the new Theano backend and only implements multi-GPU sync gradient updates (grab a big minibatch, chop it into N parts, feeding each part to one of N GPUs on the same machine node, then aggregating gradients across GPUs with averaging). 

The main source of reference is [Platoon](https://github.com/mila-udem/platoon), which offers a more sophiscated toolkit of parallel training across single/multiple nodes, with ASGD, EASGD, Downpour SGD.  

## Dependency
* [Lasagne](https://github.com/Lasagne/Lasagne)
* [Theano with gpuarray backend, with NCCL support](http://deeplearning.net/software/theano/tutorial/using_gpu.html#gpuarray-backend)

## Usage
`python launcher.py N` where `N` is the number of GPUs on the same motherboard. The default model is VGG16, a ConvNet with ~140M parameters.

## Benchmark
| minibath_size | 1 gpu  | 2 gpus | 3 gpus | 4 gpus |
|---------------|--------|--------|--------|--------|
| 16            | 54.51  | 33.83  | 28.2   | 23.29  |
| 32            | 100.86 | 59.7   | 46.48  | 36.83  |
| 64            | 194.53 | 107.56 | 77.45  | 63.35  |

![Benchmark picture](https://github.com/yaoli/theano_multi_gpu/blob/master/benchmark.png)

## Bonus
It comes with RESNET50 and VGG16 compatible with the gpuarray backend. Both are adapted from Lasagne Model Zoo. 
