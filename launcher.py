import sys, time, os, subprocess
import posix_ipc

# The framework has no server, so the following files
# are used to sync between workers.
# It's OK to ignore the warning sent wrt this line.
os.system('rm comm_id.pkl server_socket_init_done.pkl')

def launch():    
    assert len(sys.argv) == 2
    world_size = int(sys.argv[1])
    job_id = int(time.time())
    posix_ipc.Semaphore("{}_lock".format(job_id), posix_ipc.O_CREAT, initial_value=1)
    workers = []
    for rank in range(world_size):
        env = dict(os.environ)
        env['THEANO_FLAGS'] = 'mode=FAST_RUN,optimizer=None,device=cuda%d,floatX=float32,compiledir=/home/lyao/.theano/gpu%d'%(
            rank, rank)
        command = ['python', '-u', 'multi_gpu.py', '%d'%rank, '%d'%world_size, '%d'%int(time.time())]
        worker = subprocess.Popen(command, stdout=None, stderr=None, bufsize=0, env=env)
        workers.append(worker)
    # wait till all process finish    
    for worker in workers:
        worker.wait()
        
if __name__ == '__main__':
    launch()
    
