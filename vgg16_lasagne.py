'''
should get something like:
0 / 782 minibatches, acu TOP1 64.0625, TOP5 84.3750
1 / 782 minibatches, acu TOP1 67.1875, TOP5 88.2812
2 / 782 minibatches, acu TOP1 69.2708, TOP5 88.5417
3 / 782 minibatches, acu TOP1 68.3594, TOP5 87.5000
4 / 782 minibatches, acu TOP1 66.8750, TOP5 87.1875
5 / 782 minibatches, acu TOP1 67.7083, TOP5 88.0208
6 / 782 minibatches, acu TOP1 68.3036, TOP5 88.1696
7 / 782 minibatches, acu TOP1 70.1172, TOP5 88.8672
8 / 782 minibatches, acu TOP1 69.6181, TOP5 89.0625
9 / 782 minibatches, acu TOP1 68.9062, TOP5 89.5312
10 / 782 minibatches, acu TOP1 69.1761, TOP5 89.0625
...
'''
import glob, os, time, re, socket
import numpy
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
#from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
#from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
import lasagne
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from PIL import Image
import utils

hostname = socket.gethostname()
vgg16_file = '/dl1/data/projects/models/vgg16.pkl'
    
def build_model(x):
    print 'build vgg16 model'
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224), x)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    '''
    download from 
    https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl
    '''
    file = open(vgg16_file, 'r')
    vals = pickle.load(file)

    values = pickle.load(
        open(vgg16_file))['param values']
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
    print 'total number of params ', numpy.sum([param.get_value().size for param in params])
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
