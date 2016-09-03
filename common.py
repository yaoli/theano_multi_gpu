import cPickle, os
import numpy
from collections import OrderedDict
import theano
import theano.tensor as tensor
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.sandbox.cuda.dnn import dnn_conv

# the dir where there should be a subdir named 'youtube2text_iccv15'
RAB_DATASET_BASE_PATH = '/data/lisatmp3/yaoli/datasets/'
# the dir where all the experiment data is dumped.
RAB_EXP_PATH = '/data/lisatmp3/yaoli/exp/'

relu = lambda x: T.maximum(0.0, x)

def init_tparams_fc(nin, nout, prefix, scale=0.01):
    W = theano.shared(norm_weight(nin, nout, scale), name='%s_W'%prefix)
    b = theano.shared(numpy.zeros((nout,), dtype='float32'),
                           name='%s_b'%prefix)
    return W, b

def init_tparams_matrix(nin, nout, prefix, scale=0.01):
    W = theano.shared(norm_weight(nin, nout, scale), name='%s_W'%prefix)
    return W

def init_tparams_lstm(nin, dim, prefix):
    # Stack the weight matricies for faster dot prods
    # (nin, dim*4)
    W = theano.shared(numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1),
                      name='%s_W'%prefix)
    # (nin, dim*4)
    U = theano.shared(numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1),
                      name='%s_U'%prefix)
    b = theano.shared(numpy.zeros((4 * dim,)).astype('float32'),
                           name='%s_b'%prefix)
    return W, U, b

def init_tparams_gru(nin, dim, prefix):
    # Stack the weight matricies for faster dot prods
    # (nin, dim*3)
    W = theano.shared(numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1),
                      name='%s_W'%prefix)
    # (nin, dim*4)
    U = theano.shared(numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1),
                      name='%s_U'%prefix)
    b = theano.shared(numpy.zeros((3 * dim,)).astype('float32'),
                           name='%s_b'%prefix)
    return W, U, b

def fprop_fc(W, b, x, activation='rectifier', use_dropout=False, dropout_flag=None):
    # x (m, f, dim_frame)
    y = T.dot(x, W) + b
    if activation == 'rectifier':
        y = rectifier(y)
    elif activation == 'tanh':
        y = T.tanh(y)
    elif activation == 'linear':
        pass
    else:
        raise NotImplementedError()
    if use_dropout:
        print 'lstm uses dropout'
        assert dropout_flag is not None
        dropout_mask = T.switch(
            dropout_flag,
            rng_theano.binomial(y.shape, p=0.5, n=1, dtype='float32'),
            T.ones_like(y) * 0.5)
        y = dropout_mask * y
    return y

def fprop_lstm(dim, W, U, b, x, mask, use_dropout=False, dropout_flag=None):
    # x (f, m, dim_frame)
    # m (f, m)
    nsteps = x.shape[0]
    nsamples = x.shape[1]
    state_below = T.dot(x, W) + b
    if use_dropout:
        print 'fc uses dropout'
        assert dropout_flag is not None
        dropout_mask = T.switch(
            dropout_flag,
            rng_theano.binomial(state_below.shape, p=0.5, n=1, dtype='float32'),
            T.ones_like(state_below) * 0.5)
        state_below = state_below * dropout_mask

    init_state = T.alloc(0., nsamples, dim)
    init_memory = T.alloc(0., nsamples, dim)
    def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

    def _step(m_, x_, h_, c_, U):
        preact = T.dot(h_, U)
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, dim))
        f = T.nnet.sigmoid(_slice(preact, 1, dim))
        o = T.nnet.sigmoid(_slice(preact, 2, dim))
        g = T.tanh(_slice(preact, 3, dim))
        c = f * c_ + i * g
        h = o * T.tanh(c)

        if m_.ndim == 0:
            # when using this for minibatchsize=1
            h = m_ * h + (1. - m_) * h_
            c = m_ * c + (1. - m_) * c_
        else:
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            c = m_[:,None] * c + (1. - m_)[:,None] * c_
        return h, c, i, f, o, g, preact
    rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            non_sequences=[U],
            outputs_info = [init_state, init_memory, None, None, None, None, None],
            name='lstm',
            n_steps=nsteps,
            strict=True,
            profile=False)
    return rval

def fprop_gru(dim, W, U, b, x, mask, use_dropout=False, dropout_flag=None):
    # x (f, m, dim_frame)
    # m (f, m)
    nsteps = x.shape[0]
    nsamples = x.shape[1]
    state_below = T.dot(x, W) + b
    if use_dropout:
        print 'fc uses dropout'
        assert dropout_flag is not None
        dropout_mask = T.switch(
            dropout_flag,
            rng_theano.binomial(state_below.shape, p=0.5, n=1, dtype='float32'),
            T.ones_like(state_below) * 0.5)
        state_below = state_below * dropout_mask

    init_state = T.alloc(0., nsamples, dim)
    def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            elif _x.ndim == 2:
                return _x[:, n*dim:(n+1)*dim]
            return _x[n*dim:(n+1)*dim]

    def _step(m_, x_, h_, U):
        recurrence = T.dot(h_, U)
        r = T.nnet.sigmoid(_slice(x_, 0, dim) + _slice(recurrence, 0, dim))
        z = T.nnet.sigmoid(_slice(x_, 1, dim) + _slice(recurrence, 1, dim))
        h_tilde = T.tanh(_slice(x_, 2, dim) + r * _slice(recurrence, 2, dim))
        h =  (1. - z)*  h_ + z * h_tilde
        if m_.ndim == 0:
            # when using this for minibatchsize=1
            h = m_ * h + (1. - m_) * h_
        else:
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
        return h, h_tilde, r, z, recurrence
    rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            non_sequences=[U],
            outputs_info = [init_state, None, None, None, None],
            name='gru',
            n_steps=nsteps,
            strict=True,
            profile=False)
    return rval

def init_tparams_conv(num_filters, num_channels, filter_size, prefix):
    filters = theano.shared(
        norm_weight_tensor((num_filters, num_channels)+filter_size),
        name='%s_filters'%prefix)
    bias = theano.shared(numpy.zeros((num_filters,), dtype='float32'), name='%s_bias'%prefix)
    return filters, bias

def fprop_conv(x, W, b, padding, act='relu'):
    if act != 'relu':
        raise NotImplementedError()
    return relu(dnn_conv(x, W, border_mode=padding) + b[None, :, None, None])

def fc_gru_visualize(x, x_mask, h, h_tilde, z, r, recurrence, W, U, b, save_path):
    import matplotlib.pyplot as plt
        
    idx = numpy.argmax(x_mask.sum(1))
    x = x[:, idx]
    h = h[:, idx]
    h_tilde = h_tilde[:, idx]
    z = z[:, idx]
    r = r[:, idx]
    hU = recurrence[:, idx]
    fig = plt.figure(figsize=(40,20))
    layout = (2,4)
    # first row
    ax = plt.subplot2grid(layout, (0,0))
    img = ax.imshow(h, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('h')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,1))
    img = ax.imshow(h_tilde, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('h_tilde')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,2))
    img = ax.imshow(r, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('r')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,3))
    img = ax.imshow(z, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('z')
    plt.colorbar(img)
    # second row
    ax = plt.subplot2grid(layout, (1,0))
    img = ax.imshow(x, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('x')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (1,1), colspan=2)
    std = hU.std(0)
    mean = hU.mean(0)
    ax.errorbar(range(std.shape[0]), mean, std, fmt='o')
    ax.set_title('hU')
    ax = plt.subplot2grid(layout, (1,3))
    first = numpy.sqrt(((h)**2).sum(1))
    second = numpy.sqrt(((h_tilde)**2).sum(1))
    plt.plot(first, label='h')
    plt.plot(second, label='h_tilde')
    plt.legend()
    ax.set_title('l2 norm')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def fc_lstm_visualize(x, x_mask, h, c, g_i, g_f, g_o, g, preact, W, U, b, save_path):
    # x (f,m,d), x_mask (m,f)
    import matplotlib.pyplot as plt
    idx = numpy.argmax(x_mask.sum(1))
    x = x[:, idx]
    h = h[:, idx]
    c = c[:, idx]
    g_i = g_i[:, idx]
    g_f = g_f[:, idx]
    g_o = g_o[:, idx]
    g = g[:,idx]
    
    preact = preact[:, idx]
    fig = plt.figure(figsize=(40,20))
    layout = (5,5)
    # first row
    ax = plt.subplot2grid(layout, (0,0))
    img = ax.imshow(h, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('h')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,1))
    img = ax.imshow(c, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('c')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,2))
    img = ax.imshow(g_i, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('i')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,3))
    img = ax.imshow(g_f, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('f')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (0,4))
    img = ax.imshow(g_o, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('o')
    plt.colorbar(img)
    # second row
    ax = plt.subplot2grid(layout, (1,0))
    plt.plot(numpy.sqrt((h**2).sum(1)), '-*')
    ax.set_title('L2 norm of h')
    ax = plt.subplot2grid(layout, (1,1))
    plt.plot(numpy.sqrt((c**2).sum(1)), '-*')
    ax.set_title('L2 norm of c')
    ax = plt.subplot2grid(layout, (1,2))
    plt.plot(numpy.sqrt((g_i**2).sum(1)), '-*')
    ax.set_title('L2 norm of i')
    ax = plt.subplot2grid(layout, (1,3))
    plt.plot(numpy.sqrt((g_f**2).sum(1)), '-*')
    ax.set_title('L2 norm of f')
    ax = plt.subplot2grid(layout, (1,4))
    plt.plot(numpy.sqrt((g_o**2).sum(1)), '-*')
    ax.set_title('L2 norm of o')
    # third row
    ax = plt.subplot2grid(layout, (2,0), colspan=5)
    std = preact.std(0)
    mean = preact.mean(0)
    ax.errorbar(range(std.shape[0]), mean, std, fmt='o')
    ax.set_title('preact')
    # fourth row
    ax = plt.subplot2grid(layout, (3,0))
    img = ax.imshow(x, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('input sequence')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (3,1), colspan=2)
    img = ax.imshow(preact, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('preact')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (3,3))
    img = ax.imshow(g, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('g')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (3,4))
    first = numpy.sqrt(((g_f*c)**2).sum(1))
    second = numpy.sqrt(((g_i*g)**2).sum(1))
    plt.plot(first, label='g_f * c')
    plt.plot(second, label='g_i * g')
    plt.legend()
    ax.set_title('l2 norm')
    # fifth row
    ax = plt.subplot2grid(layout, (4,0), colspan=2)
    img = ax.imshow(W, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('W')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (4,2), colspan=2)
    img = ax.imshow(U, aspect='auto', cmap='gray', interpolation='none')
    ax.set_title('U')
    plt.colorbar(img)
    ax = plt.subplot2grid(layout, (4,4))
    plt.plot(b.flatten())
    ax.set_title('b')
    plt.colorbar(img)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
# ==============================================================================
def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)
    return rng_numpy, rng_theano

rng_numpy, rng_theano = get_two_rngs()

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
'''
Theano shared variables require GPUs, so to
make this code more portable, these two functions
push and pull variables between a shared
variable dictionary and a regular numpy 
dictionary
'''
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, 
                         state_before * 
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj

def list_to_dict(tparams):
    # tparams is a list of tparam
    t = OrderedDict()
    for param in tparams:
        assert param.name
        t[param.name] = param
    return t

# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# some utilities
def ortho_weight(ndim):
    """
    Random orthogonal weights, we take
    the right matrix in the SVD.

    Remember in SVD, u has the same # rows as W
    and v has the same # of cols as W. So we
    are ensuring that the rows are 
    orthogonal. 
    """
    W = rng_numpy.randn(ndim, ndim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype('float32')

def norm_weight_svd(nin, nout, scale=0.01):
    W = rng_numpy.randn(nin, nout)
    U, S, V = numpy.linalg.svd(W)
    T = numpy.zeros((U.shape[1], V.shape[0]), dtype='float32')
    numpy.fill_diagonal(T, numpy.ones_like(S).astype('float32'))
    W_ = numpy.dot(numpy.dot(U, T), V).astype('float32')
    return W_
        
def norm_weight(nin,nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * rng_numpy.randn(nin, nout)
    return W.astype('float32')

def norm_weight_tensor(shape, scale=0.001):
    return (scale * rng_numpy.randn(*shape)).astype('float32')

def tanh(x):
    return tensor.tanh(x)

def rectifier(x):
    return tensor.maximum(0., x)

def linear(x):
    return x

# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive'%kk)
        params[kk] = pp[kk]
    return params

def grad_nan_report(grads, tparams):
    numpy.set_printoptions(precision=3)
    D = OrderedDict()
    i = 0
    NaN_keys = []
    magnitude = []
    assert len(grads) == len(tparams)
    for k, v in tparams.iteritems():
        grad = grads[i]
        magnitude.append(numpy.abs(grad).mean())
        if numpy.isnan(grad.sum()):
            NaN_keys.append(k)
        #assert v.get_value().shape == grad.shape
        D[k] = grad
        i += 1
    #norm = [numpy.sqrt(numpy.sum(grad**2)) for grad in grads]
    #print '\tgrad mean(abs(x))', numpy.array(magnitude)
    return D, NaN_keys

# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adadelta(lr, tparams, grads, inp, cost, extra):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rup2'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, [cost] + extra, updates=zgup+rg2up,
                                    profile=False, on_unused_input='ignore')
    
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost, extra, on_unused_input='warn'):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%p.name) for p in tparams]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra, updates=gsup, on_unused_input=on_unused_input)

    lr0 = 0.0002
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams, gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore')

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost, extra):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad'%k) for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, [cost]+extra,
                                    updates=zgup+rgup+rg2up, profile=False)

    updir = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_updir'%k) for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up, on_unused_input='ignore', profile=False)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost, extra):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad'%k) for k, p in enumerate(tparams)]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, [cost]+extra, updates=gsup, profile=False)

    pup = [(p, p - lr * g) for p, g in zip(tparams, gshared)]
    f_update = theano.function([lr], [], updates=pup, profile=False)

    return f_grad_shared, f_update

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()


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

def get_rab_dataset_base_path():
    return RAB_DATASET_BASE_PATH

def get_rab_exp_path():
    return RAB_EXP_PATH

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory

def flatten_list_of_list(l):
    # l is a list of list
    return [item for sublist in l for item in sublist]

def load_txt_file(path):
    f = open(path,'r')
    lines = f.readlines()
    f.close()
    return lines

"""
Some helper function to load images.

This function creates a subprocess which takes requests in a queue, and put 
the fetched image data in the response queue. The subprocess further uses 
a multiprocess pool to fetch multiple videos in parallel.
"""
def pad_frames(frames, limit, jpegs=True):
    # pad frames with 0, compatible with both conv and fully connected layers
    last_frame = frames[-1]
    if jpegs:
        frames_padded = frames + [last_frame]*(limit-len(frames))
    else:
        padding = numpy.asarray([last_frame*0]*(limit-len(frames)))
        frames_padded = numpy.concatenate([frames, padding], axis=0)
    return frames_padded

def pad_frames_copy(frames, limit, jpegs=True):
    # pad frames with 0, compatible with both conv and fully connected layers
    last_frame = frames[-1]
    if jpegs:
        frames_padded = frames + [last_frame]*(limit-len(frames))
    else:
        padding = numpy.asarray([last_frame]*(limit-len(frames)))
        frames_padded = numpy.concatenate([frames, padding], axis=0)
    return frames_padded

def extract_frames_equally_spaced(frames, how_many, sample_frames=False):
    # chunk frames into 'how_many' segments and use the first frame
    # from each segment
    n_frames = len(frames)
    splits = numpy.array_split(range(n_frames), how_many)
    if sample_frames:
        idx_taken = []
        for s in splits:
            idx_taken.append(s[rng_numpy.randint(0, len(s))])
        sub_frames = [frames[index] for index in idx_taken]    
    else:
        # avg within chunks
        idx_taken = [s[0] for s in splits]
        sub_frames = [frames[s[0]] for s in splits]
    return sub_frames, idx_taken

def get_sub_frames(frames, n_subframes, jpegs=True, sample_frames=False):
    # from all frames, take K of them, then add end of video frame
    # jpegs: to be compatible with visualizations
    if len(frames) < n_subframes:
        frames_ = pad_frames(frames, n_subframes, jpegs)
    else:
        frames_, idx = extract_frames_equally_spaced(frames, n_subframes, sample_frames)
    if jpegs:
        frames_ = numpy.asarray(frames_)
    return frames_

def get_sub_frames_idx(frames, n_subframes, jpegs=True, sample_frames=False):
    # from all frames, take K of them, then add end of video frame
    # jpegs: to be compatible with visualizations
    # Return the selected index as idx
    if len(frames) < n_subframes:
        idx = range(len(frames))+ [len(frames)-1]*(n_subframes-len(frames))
        frames_ = pad_frames_copy(frames, n_subframes, jpegs)
    else:
        frames_, idx = extract_frames_equally_spaced(frames, n_subframes, sample_frames)
    if jpegs:
        frames_ = numpy.asarray(frames_)

    mask = numpy.zeros((n_subframes,)).astype('float32')
    mask[:min(len(frames), n_subframes)] = 1

    return frames_, idx, mask

def tile(image_paths):
    from capgen_vid.iccv15_challenge.googlenet import load_image, resize_images
    images = numpy.asarray([load_image(img) for img in image_paths])
    W = 320
    H = 240
    images_small = [img for img in resize_images(images, [H,W])][::-1]
    N = int(numpy.ceil(numpy.sqrt(len(images) / 3.)))
    X = N
    Y = 3 * N
    IMG = numpy.zeros((X*H, Y*W, 3))
    for i in range(X):
        done = False
        for j in range(Y):
            if images_small != []:
                img = images_small.pop()
                IMG[i*H:(i+1)*H, j*W:(j+1)*W] = img
            else:
                done = True
                break
        if done:
            break
    return IMG
