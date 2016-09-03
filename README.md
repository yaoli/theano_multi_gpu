## Introduction
This repo shows how to use Theano with multiple GPUs, with the new Theano backend and only implements multi-GPU sync gradient updates (grab a big minibatch, chop it into N parts, feeding each part to one of N GPUs on the same machine node, then aggregating gradients across GPUs with averaging). 

The main source of reference is [Platoon](https://github.com/mila-udem/platoon), which offers a more sophiscated toolkit of parallel training across simple/multiple nodes, with ASGD, EASGD, Downpour SGD.  

## Dependency
* [Lasagne](https://github.com/Lasagne/Lasagne)
* [Theano with gpuarray backend, with NCCL support](http://deeplearning.net/software/theano/tutorial/using_gpu.html#gpuarray-backend)

## Usage
`python launcher.py N` where `N` is the number of GPUs on the same motherboard. The default model is VGG16, a ConvNet with ~160M parameters.

## Benchmark
![Benchmark picture](https://github.com/yaoli/theano_multi_gpu/blob/master/benchmark.png)

