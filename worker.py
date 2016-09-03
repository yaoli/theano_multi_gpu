import time, os, sys
import zmq
import posix_ipc
import mmap
import cffi
import numpy
import pygpu
from pygpu import collectives
from theano import gpuarray
from experimental import utils

class Worker(object):
    def __init__(self, rank, world_size, port_pub_sub, port_push_pull, job_id):
        self.rank = rank
        self.world_size = world_size
        self.port_pub_sub = port_pub_sub
        self.port_push_pull = port_push_pull
        self.job_id = job_id
        self._lock = posix_ipc.Semaphore("{}_lock".format(self.job_id))
        self.gpu_ctx = gpuarray.get_context(None)
        self.local_id = collectives.GpuCommCliqueId(context=self.gpu_ctx)
        self.lock()
        comm_id_file = 'comm_id.pkl'
        if not os.path.isfile(comm_id_file):
            comm_id = self.local_id.comm_id
            utils.dump_pkl(comm_id, comm_id_file)
        else:
            comm_id = utils.load_pkl(comm_id_file)
            self.local_id.comm_id = comm_id
        self.unlock()    
        print 'local_id ', self.local_id.comm_id
        # the following call is blocked till all workers finish calling it
        #print self.local_id.comm_id, self.job_id
        self.local_comm = collectives.GpuComm(self.local_id, self.world_size, self.rank)
        self.init_socket()
        print 'finish init worker with rank %d'%rank
        
    def init_socket(self):
        if self.rank == 0:
            # use this worker as server
            context = zmq.Context()
            self.socket_pub_sub = context.socket(zmq.PUB)
            self.socket_pub_sub.bind("tcp://*:%s" % self.port_pub_sub)
            print '%s established PUB socket'%self.rank
            context_ = zmq.Context()
            self.socket_push_pull = context_.socket(zmq.PULL)
            self.socket_push_pull.bind("tcp://*:%s" % self.port_push_pull)
            print '%s established PULL socket'%self.rank
            utils.dump_pkl('', 'server_socket_init_done.pkl')
        else:
            while True:
                if os.path.isfile('server_socket_init_done.pkl'):
                    break
            context = zmq.Context()
            self.socket_pub_sub = context.socket(zmq.SUB)
            self.socket_pub_sub.connect("tcp://localhost:%s" % self.port_pub_sub)
            self.socket_pub_sub.setsockopt(zmq.SUBSCRIBE, '')
            print '%s established SUB socket'%self.rank
            context_ = zmq.Context()
            self.socket_push_pull = context_.socket(zmq.PUSH)
            self.socket_push_pull.connect("tcp://localhost:%s" % self.port_push_pull)
            print '%s established PUSH socket'%self.rank
        self.context = [context, context_]

    def lock(self, timeout=None):
        self._lock.acquire(timeout)

    def unlock(self):
        self._lock.release()
        
    def all_reduce(self, src, op, dest):
        # this func is called by the theano op internally
        assert self.local_comm is not None
        assert isinstance(src, pygpu.gpuarray.GpuArray)
        assert isinstance(dest, pygpu.gpuarray.GpuArray)
        self.local_comm.all_reduce(src, op, dest)
        dest.sync()

    def init_global_params(self, params):
        print '%s init global params'%self.rank
        def _mmap(length=0, prot=0x3, flags=0x1, fd=0, offset=0):
            _ffi = cffi.FFI()
            _ffi.cdef("void *mmap(void *, size_t, int, int, int, size_t);")
            _lib = _ffi.dlopen(None)
            addr = _ffi.NULL
            m = _lib.mmap(addr, length, prot, flags, fd, offset)
            if m == _ffi.cast('void *', -1):
                raise OSError(_ffi.errno, "for mmap")
            return _ffi.buffer(m, length)

        def _get_descr_size(dtype, shape):
            size = dtype.itemsize
            for s in shape:
                size *= s
            return size
        params_descr = [(numpy.dtype(p.dtype), p.get_value(borrow=True).shape)
                        for p in params]
        params_size = sum(_get_descr_size(*d) for d in params_descr)
        shared_mem_name = str(self.job_id)
        self.lock()
        self._shmref = posix_ipc.SharedMemory(
                shared_mem_name, posix_ipc.O_CREAT, size=params_size)
        self._shm = _mmap(fd=self._shmref.fd, length=params_size)
        self._shmref.close_fd()
        self.global_params = []
        off = 0
        for dtype, shape in params_descr:
            self.global_params.append(numpy.ndarray(shape, dtype=dtype,
                                                    buffer=self._shm,
                                                    offset=off))
            off += _get_descr_size(dtype, shape)
        self.unlock()
        
    def from_global_to_local(self, params):
        #print 'copy from global ', [param.sum() for param in self.global_params]
        for param_local, param_global in zip(params, self.global_params):
            #assert param_local.get_value().shape == param_global.shape
            param_local.set_value(param_global)
            
    def from_local_to_global(self, params):
        #print 'copy from local'
        for param_local, param_global, i in zip(params, self.global_params, range(len(params))):
            #assert param_local.get_value().shape == param_global.shape
            self.global_params[i][:] = param_local.get_value(borrow=True)
