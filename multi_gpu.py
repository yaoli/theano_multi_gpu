import time, os, sys
from  multiprocessing import Process
import subprocess
import zmq
import posix_ipc
import mmap
import cffi
import numpy
import theano
import theano.tensor as T
import lasagne
from lasagne.updates import adam
import utils, common
from worker import Worker
from ops import AllReduceSum
import resnet50, vgg16_lasagne


port_pub_sub = 1111
port_push_pull = 2222
minibatch_size = 32
model = 'vgg16' # or 'debug', 'fc_mnist', 'vgg16', resnet50

class ImageNet(object):
    def __init__(self):
        self.x = numpy.random.uniform(0, 1, size=(minibatch_size*10, 3, 224, 224)).astype('float32')
        self.y = numpy.random.randint(0, 1000, size=(len(self.x,))).astype('int32')
        self.train_minibatch_idx = utils.generate_minibatch_idx(len(self.x), minibatch_size)
        
    def fetch_minibatch(self, idx, which):
        use_x = self.x
        use_y = self.y
        return use_x[idx], use_y[idx]
    
class MNIST(object):
    def __init__(self):
        # load dataset
        base_dir = '/dl1/home/lyao/Data/mnist/'
        self.train_x = numpy.load(base_dir + 'train_x.npy')
        self.train_y = numpy.load(base_dir + 'train_y.npy')
        self.valid_x = numpy.load(base_dir + 'valid_x.npy')
        self.valid_y = numpy.load(base_dir + 'valid_y.npy')
        self.test_x = numpy.load(base_dir + 'test_x.npy')
        self.test_y = numpy.load(base_dir + 'test_y.npy')
        self.train_minibatch_idx = utils.generate_minibatch_idx(len(self.train_x), minibatch_size)
        #self.valid_minibatch_idx = utils.generate_minibatch_idx(len(self.valid_x), minibatch_size)
        #self.test_minibatch_idx = utils.generate_minibatch_idx(len(self.test_x), minibatch_size)

    def fetch_minibatch(self, idx, which):
        if which == 'train':
            use_x = self.train_x
            use_y = self.train_y
        elif which == 'valid':
            use_x = self.valid_x
            use_y = self.valid_y
        else:
            use_x = self.test_x
            use_y = self.test_y
        return use_x[idx], use_y[idx]
    
class Model(object):
    def __init__(self, port_pub_sub, port_push_pull, rank, world_size, job_id):
        print 'model with rank %d'%rank
        self.rank = rank
        if model == 'resnet50' or model == 'vgg16':
            self.engine = ImageNet()
        else:
            self.engine = MNIST()
        self.world_size = world_size
        self.worker = Worker(rank, world_size, port_pub_sub, port_push_pull, job_id)
        
    def build_theano_fn(self):
        if model == 'resnet50':
            self.build_theano_fn_resnet()
        elif model == 'vgg16':
            self.build_theano_fn_vgg()
        elif model == 'simple' or model == 'debug':
            self.build_theano_fn_simple()
        else:
            raise NotImplementedError()

    def build_theano_fn_resnet(self):
        t0 = time.time()
        print '%s build theano fn resnet'%self.rank
        x = T.ftensor4('images')
        y = T.ivector('label')
        model = resnet50.build_model(x)
        prob = lasagne.layers.get_output(model['prob'], deterministic=True)
        self.params = lasagne.layers.get_all_params(model['prob'], trainable=True)
        cost = -T.log(prob[T.arange(prob.shape[0]), y] + 1e-6).mean()
        grads = T.grad(cost, self.params)
        grads_all_reduced = self.grad_all_reduce(grads)
        updates = adam(grads_all_reduced, self.params)
        self.train_fn = theano.function([x, y], [cost, y], updates=updates, accept_inplace=True)
        print '%s finished build theano fn, used %.3f'%(self.rank, time.time()-t0)

    def build_theano_fn_vgg(self):
        t0 = time.time()
        print '%s build theano fn vgg16'%self.rank
        x = T.ftensor4('images')
        y = T.ivector('label')
        model = vgg16_lasagne.build_model(x)
        prob = lasagne.layers.get_output(model['prob'], deterministic=True)
        self.params = lasagne.layers.get_all_params(model['prob'], trainable=True)
        cost = -T.log(prob[T.arange(prob.shape[0]), y] + 1e-6).mean()
        grads = T.grad(cost, self.params)
        grads_all_reduced = self.grad_all_reduce(grads)
        updates = adam(grads_all_reduced, self.params)
        #updates = adam(grads, self.params)
        self.train_fn = theano.function([x, y], [cost, y], updates=updates, accept_inplace=True)
        print '%s finished build theano fn, used %.3f'%(self.rank, time.time()-t0)
        
    def build_theano_fn_simple(self):
        print '%s build theano fn simple'%self.rank
        x = T.fmatrix('x')
        y = T.ivector('y')
        W_1, b_1 = common.init_tparams_fc(784, 1000, 'l1')
        out_1 = T.tanh(T.dot(x, W_1) + b_1)
        W_2, b_2 = common.init_tparams_fc(1000, 2000, 'l2')
        out_2 = T.tanh(T.dot(out_1, W_2) + b_2)
        W_3, b_3 = common.init_tparams_fc(2000, 3000, 'l3')
        out_3 = T.tanh(T.dot(out_2, W_3) + b_3)
        W_4, b_4 = common.init_tparams_fc(3000, 10, 'softmax')
        prob = T.nnet.softmax((T.dot(out_3, W_4) + b_4))
        self.params = [W_1, b_1, W_2, b_2, W_3, b_3, W_4, b_4]
        # cost
        cost = -T.log(prob[T.arange(prob.shape[0]), y] + 1e-6).mean()
        pred = T.argmax(prob, 1)
        grads = T.grad(cost, self.params)
        grads_all_reduced = self.grad_all_reduce(grads)
        updates = adam(grads_all_reduced, self.params)
        #updates = adam(grads, self.params)
        self.train_fn = theano.function([x, y], [cost, prob, pred, y], updates=updates, accept_inplace=True)
        
        # the following code is used for debugging only
        if model == 'debug':
            self.debug_var = theano.shared(numpy.float32(1.))
            debug_var_global = AllReduceSum(self.debug_var, inplace=True, worker=self.worker)
            updates = {self.debug_var: debug_var_global}
            self.debug_fn = theano.function([], [], updates=updates, accept_inplace=True)
        
    def grad_all_reduce(self, local_grads):
        global_grads = []
        for grad in local_grads:
            global_grad = AllReduceSum(grad, inplace=True, worker=self.worker) / self.world_size
            global_grads.append(global_grad)
        return global_grads
    
    def train_server(self):
        print 'training %d'%self.rank
        costs = []
        if model == 'debug':
            t0 = time.time()
            debug_var = []
            assert self.world_size == 4
            for epoch in range(3):
                self.worker.socket_pub_sub.send_pyobj('debug')
                self.debug_fn()
                debug_var.append(self.debug_var.get_value().tolist())
            assert debug_var == [4, 16, 64]
        else:
            t0 = time.time()
            for epoch in range(5):
                for ii, idx in enumerate(self.engine.train_minibatch_idx):
                    microbatch_idx = numpy.array_split(idx, self.world_size)
                    mapping = {i: microbatch_idx[i] for i in range(self.world_size)}
                    self.worker.socket_pub_sub.send_pyobj(mapping)
                    x, y = self.engine.fetch_minibatch(mapping[0], 'train')
                    rval = self.train_fn(x, y)
                    cost = rval[0]
                    costs.append(cost)
                print 'epoch %d, avg cost %.3f (ignore this one for benchmarking purpose)'%(epoch, numpy.mean(costs))
            # terminate all workers
        print 'train end %d'%self.rank    
        if self.world_size > 1:
            self.worker.socket_pub_sub.send_pyobj('terminate')
        print 'total time ', time.time() - t0
        print 'post running clean up'
        os.system('rm comm_id.pkl server_socket_init_done.pkl')
        
    def train_worker(self):
        print 'training %d'%self.rank
        while True:
            msg = self.worker.socket_pub_sub.recv_pyobj()
            if msg == 'terminate':
                break
            if model == 'debug':
                self.debug_fn()
            else:    
                idx = msg[self.rank]
                t2 = time.time()
                x, y = self.engine.fetch_minibatch(idx, 'train')
                rval = self.train_fn(x, y)
        print 'train end %d'%self.rank
        
    def train(self):
        '''sync between server and workers'''
        if self.rank == 0:
            print 'worker %d ready'%self.rank
            # sync with worker, making sure they are running
            for i in range(self.world_size-1):
                msg = self.worker.socket_push_pull.recv()
                print 'worker %s ready'%msg
            self.train_server()
        else:
            # sync with server
            self.worker.socket_push_pull.send(str(self.rank))
            self.train_worker()
        
            
def run():
    assert len(sys.argv) == 4
    rank = int(sys.argv[1])
    world_size = int(sys.argv[2])
    job_id = int(sys.argv[3])
    model = Model(port_pub_sub, port_push_pull, rank, world_size, job_id)
    model.build_theano_fn()
    #model.worker.init_global_params(model.params)
    #model.worker.from_global_to_local(model.params)
    model.train()
            
if __name__ == '__main__':
    run()
        
        
