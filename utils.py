import cPickle, os, time, sys, itertools, re, json, pdb, csv
import numpy
import skimage
from skimage.transform import resize
import skimage.io
import dicom_utils

class VGGImageFuncs(object):
    def __init__(self, mean=None):
        self.image_dims = [256, 256]
        self.channel_swap = [2, 1, 0]
        self.raw_scale = 255.0
        self.crop_dims = numpy.array([224, 224])
        self.mean = self.set_mean(mean)
        
    def set_mean(self, mean, mode='elementwise'):
        """
        Set the mean to subtract for data centering.
        mean: (3,x,y)
        Take
        mean: mean K x H x W ndarray (input dimensional or broadcastable)
        mode: elementwise = use the whole mean (and check dimensions)
              channel = channel constant (e.g. mean pixel instead of mean image)
        """
        if mean is None:
            MEAN = '/dl1/data/projects/imagenet/ilsvrc_2012_mean.npy'
            mean = numpy.load(MEAN)
        crop_dims = tuple(self.crop_dims.tolist())
        if mode == 'elementwise':
            if mean.shape[1:] != crop_dims:
                # Resize mean (which requires H x W x K input).
                mean = self.resize_image(mean.transpose((1,2,0)),
                                    crop_dims).transpose((2,0,1))
        elif mode == 'channel':
            mean = mean.mean(1).mean(1).reshape((in_shape[1], 1, 1))
        elif mode == 'nothing':
            mean = mean.mean(0)
        else:
            raise Exception('Mode not in {}'.format(['elementwise', 'channel']))
        return mean
        
    def load_image(self, filename, color=True):
        """
        Load an image converting from grayscale or alpha as needed.

        Take
        filename: string
        color: flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).

        Give
        image: an image with type numpy.float32 in range [0, 1]
            of size (H x W x 3) in RGB or
            of size (H x W x 1) in grayscale.
        """
        img = skimage.img_as_float(skimage.io.imread(filename)).astype(numpy.float32)
        if img.ndim == 2:
            img = img[:, :, numpy.newaxis]
            if color:
                img = numpy.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img
        
    def resize_image(self, im, new_dims, interp_order=1):
        """
        Resize an image array with interpolation.

        Take
        im: (H x W x K) ndarray
        new_dims: (height, width) tuple of new dimensions.
        interp_order: interpolation order, default is linear.

        Give
        im: resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
        if im.shape[-1] == 1 or im.shape[-1] == 3:
            # skimage is fast but only understands {1,3} channel images in [0, 1].
            im_min, im_max = im.min(), im.max()
            im_std = (im - im_min) / (im_max - im_min + numpy.float32(1e-10))
            resized_std = resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # ndimage interpolates anything but more slowly.
            # but this handles batch
            scale = tuple(numpy.array(new_dims) / (numpy.array(im.shape[:2])+.0))
            resized_im = zoom(im, scale + (1,), order=interp_order)
        return resized_im.astype(numpy.float32)

    def preprocess(self, image_paths, oversample=False, transform=True):
        """
        inputs: iterable of (H x W x K) input ndarrays
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.
        """
        t0 = time.time()
        inputs = [self.load_image(image_path) for image_path in image_paths]
        # Scale to standardize input dimensions.
        #t0 = time.time()
        input_ = numpy.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=numpy.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = self.resize_image(in_, self.image_dims)
            if numpy.sum(input_[ix]) == 0:
                # there is a case of totaly black image
                input_[ix] = numpy.float32(1e-5)
        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = oversample_image(input_, self.crop_dims)
        else:
            # Take center crop.
            center = numpy.array(self.image_dims) / 2.0
            crop = numpy.tile(center, (1, 2))[0] + numpy.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        
        def transform(x):
            # x (m,x,y,c)
            x_ = x.astype(numpy.float32, copy=False) # (224,224,3)
            x_ = x_[:, :, :, self.channel_swap] 
            x_ = x_.transpose((0, 3, 1, 2)) # (m, c, x, y)
            x_ *= self.raw_scale
            x_ -= self.mean[numpy.newaxis, :, :, :] # mean is between 0 and 255
            return x_
        
        ins = transform(input_)
        return ins
    
    def reverse_transform(self, images):
        m, c, x, y = images.shape
        out = images + self.mean[numpy.newaxis, :, :, :]
        out = (out + 0.) / self.raw_scale
        out = out[:, self.channel_swap, :, :]
        out = out.transpose((0, 2, 3, 1))
        return out
    
def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'Creating directory %s' % directory
        os.makedirs(directory)
    else:
        print "%s already exists. Skipping" % directory

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

def write_txt_file(path, data):
    f = open(path,'w')
    lines = f.write(data.encode('utf-8'))
    f.close()
    
def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    with open(path, 'rb') as f:
        return cPickle.load(f)

def load_csv(path, delimiter):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        lines = [row for row in reader]
    return lines

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
        
def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]
    
def dicom_to_array(dicom_path):
    res = dicom_utils.dicom_read_file(
        filename=dicom_path, pixel_reader='pydicom', pixel_format='pydicom', convert_pixels="default")
    pixel = res['PixelData']
    return pixel

def read_one_ct_scan_from_paths(dicom_paths):
    dicoms = []
    for ii, dicom_path in enumerate(dicom_paths):
        dicoms.append(dicom_utils.dicom_read_file(filename=dicom_path, pixel_reader='gdcm'))
        #print 'reading %d / %d dicom files'%(ii, len(dicom_paths))
    # organize dicoms according to their natural ordering
    zs = [dicom['SliceLocation'] for dicom in dicoms]
    return [dicoms[index] for index in numpy.argsort(zs)]
    
def dicom_visualize(dicom_path, save_path):
    pixel = dicom_to_array(dicom_path)
    # visualization
    plt.imshow(pixel, cmap=matplotlib.cm.bone)
    plt.savefig(save_path)
    plt.close()
    print 'saved dicom on %s'%save_path

def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size >= minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def divide_to_3_folds(size, mode=[.70, .15, .15]):
    """
    this function shuffle the dataset and return indices to 3 folds
    of train, valid, test

    minibatch_size is not None then, we move tails around to accommadate for this.
    mostly for convnet.
    """
    numpy.random.seed(1234)
    indices = range(size)
    numpy.random.shuffle(indices)
    s1 = int(numpy.floor(size * mode[0]))
    s2 = int(numpy.floor(size * (mode[0] + mode[1])))
    s3 = size
    idx_1 = indices[:s1]
    idx_2 = indices[s1:s2]
    idx_3 = indices[s2:]

    return idx_1, idx_2, idx_3

def sort_by_numbers_in_file_name(list_of_file_names):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
        """
        return [ tryint(c) for c in re.split('(\-?[0-9]+)', s) ]

    def sort_nicely(l):
        """ Sort the given list in the way that humans expect.
        """
        l.sort(key=alphanum_key)
        return l

    return sort_nicely(list_of_file_names)

def five_fold_cv(n):
    numpy.random.seed(1234)
    idx = range(n)
    numpy.random.shuffle(idx)
    f1, f2, f3, f4, f5 = numpy.array_split(idx, 5)
    combo = [
        [numpy.concatenate([f1, f2, f3, f4]), f5, f5],
        [numpy.concatenate([f1, f2, f3, f5]), f4, f4],
        [numpy.concatenate([f1, f2, f5, f4]), f3, f3],
        [numpy.concatenate([f1, f5, f3, f4]), f2, f2],
        [numpy.concatenate([f5, f2, f3, f4]), f1, f1],
        ]
    return combo

def five_fold_double_cv(n):
    numpy.random.seed(1234)
    idx = range(n)
    numpy.random.shuffle(idx)
    f1, f2, f3, f4, f5 = numpy.array_split(idx, 5)
    combo = [
        [numpy.concatenate([f3, f4, f5]), f1, f2],
        [numpy.concatenate([f3, f4, f5]), f2, f1],
        [numpy.concatenate([f2, f4, f5]), f1, f3],
        [numpy.concatenate([f2, f4, f5]), f3, f1],
        [numpy.concatenate([f2, f3, f5]), f1, f4],
        [numpy.concatenate([f2, f3, f5]), f4, f1],
        [numpy.concatenate([f2, f3, f4]), f1, f5],
        [numpy.concatenate([f2, f3, f4]), f5, f1],
        [numpy.concatenate([f1, f4, f5]), f2, f3],
        [numpy.concatenate([f1, f4, f5]), f3, f2],
        [numpy.concatenate([f1, f3, f5]), f2, f4],
        [numpy.concatenate([f1, f3, f5]), f4, f2],
        [numpy.concatenate([f1, f3, f4]), f2, f5],
        [numpy.concatenate([f1, f3, f4]), f5, f2],
        [numpy.concatenate([f1, f2, f5]), f3, f4],
        [numpy.concatenate([f1, f2, f5]), f4, f3],
        [numpy.concatenate([f1, f2, f4]), f3, f5],
        [numpy.concatenate([f1, f2, f4]), f5, f3],
        [numpy.concatenate([f1, f2, f3]), f4, f5],
        [numpy.concatenate([f1, f2, f3]), f5, f4],
        ]
    return combo

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

        """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = file('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

