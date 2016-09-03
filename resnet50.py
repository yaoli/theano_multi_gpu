# ResNet-50, network from the paper:
# "Deep Residual Learning for Image Recognition"
# http://arxiv.org/pdf/1512.03385v1.pdf
# License: see https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
import numpy as np
import cPickle as pickle
from PIL import Image
import theano
import theano.tensor as T
import glob, os, time, re
from collections import OrderedDict
import utils

model_file = '/dl1/data/projects/models/resnet50.pkl'

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    names : list of string
        Names of the layers in block

    num_filters : int
        Number of filters in convolution layer

    filter_size : int
        Size of filters in convolution layer

    stride : int
        Stride of convolution layer

    pad : int
        Padding of convolution layer

    use_bias : bool
        Whether to use bias in conlovution layer

    nonlin : function
        Nonlinearity type of Nonlinearity layer

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return OrderedDict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block

    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer

    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block

    ratio_size : float
        Scale factor of filter size

    has_left_branch : bool
        if True, then left branch contains simple block

    upscale_factor : float
        Scale factor of filter bank at the output of residual block

    ix : int
        Id of residual block

    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i_%s_conv', 'res%s_branch%i_%s_bn', 'res%s_branch%i_%s_relu']

    net = OrderedDict()

    # right branch
    #print 'right branch'
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    #print lasagne.layers.get_output_shape(incoming_layer)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    #print lasagne.layers.get_output_shape(incoming_layer)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    #print lasagne.layers.get_output_shape(incoming_layer)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        #print 'left branch'
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, 'd'), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]
        #print lasagne.layers.get_output_shape(incoming_layer)
    #print 'sum left and right'    
    net['res%s_sum' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s_sum' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_model(x):
    net = OrderedDict()
    #print 'inputs (3,224,224)'
    net['input'] = InputLayer((None, 3, 224, 224), x)
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['res1_conv1', 'res1_bn', 'res1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    #print lasagne.layers.get_output_shape(net['res1_relu'])
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    #print 'pooled: ', lasagne.layers.get_output_shape(net['pool1'])
    block_size = range(3)
    parent_layer_name = 'pool1'
    
    #print 'start'
    for c in block_size:
        if c == 0:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2')
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2')
        net.update(sub_net)
    #print '=============================='
    block_size = range(4)
    for c in block_size:
        if c == 0:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3')
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3')
        net.update(sub_net)
    #print '=============================='
    block_size = range(6)
    for c in block_size:
        if c == 0:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4')
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4')
        net.update(sub_net)
    #print '=============================='
    block_size = range(3)
    for c in block_size:
        if c == 0:
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5')
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5')
        net.update(sub_net)
    #print '=============================='
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    file = open(model_file, 'r')
    values = pickle.load(
        open(model_file))['values']

    lasagne.layers.set_all_param_values(
        net['prob'], [v.astype(np.float32) for v in values])
    
    return net

def classify():
    IMAGE = '/dl1/data/projects/imagenet/valid/'
    LABELS = '/dl1/data/projects/imagenet/val.txt'
    MEAN = '/dl1/data/projects/imagenet/ilsvrc_2012_mean.npy'
    EXT = 'JPEG'
    preprocessor = utils.VGGImageFuncs()
    '''build theano fn'''
    x = T.ftensor4('images')
    model = build_model(x)
    y = lasagne.layers.get_output(model['prob'], deterministic=True)
    params = lasagne.layers.get_all_params(model['prob'], trainable=True)
    classify_fn = theano.function([x], y)

    '''perform classification'''
    files = glob.glob(IMAGE + '/*.' + EXT)
    files = utils.sort_by_numbers_in_file_name(files)
    labels = utils.load_txt_file(LABELS)
    labels = [int((label.split(' ')[-1]).strip()) for label in labels]
    # go through minibatches
    idx = utils.generate_minibatch_idx(len(files), 64)
    TOP1s = []
    TOP5s = []
    for i, index in enumerate(idx):
        t0 = time.time()
        current = [files[j] for j in index]
        gts = np.asarray([labels[j] for j in index])
        #inputs =[load_image(im_f) for im_f in current]
        inputs = preprocessor.preprocess(current)
        import ipdb; ipdb.set_trace()
        probs = classify_fn(inputs) # (m, 1000, 1, 1)
        probs = np.squeeze(probs)
        predictions = probs.argsort()[:, ::-1][:, :5]
        for pred, gt in zip(predictions, gts):
            TOP1 = pred[0] == gt
            TOP5 = gt in pred
            TOP1s.append(TOP1)
            TOP5s.append(TOP5)
        print '%d / %d minibatches, acu TOP1 %.4f, TOP5 %.4f, used %.2f'%(
            i, len(idx), np.mean(TOP1s) * 100, np.mean(TOP5s) * 100, time.time()-t0)

if __name__ == "__main__":
    classify()
