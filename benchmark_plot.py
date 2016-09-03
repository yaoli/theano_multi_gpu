from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m16 = [54.51, 33.83, 28.2, 23.29]
m32 = [100.86, 59.7, 46.68, 36.83]
m64 = [194.53, 107.56, 77.45, 63.35]
plt.plot([1,2,3,4], m16, '--ro', label='minibatch size 16')
plt.plot([1,2,3,4], m32, '--go', label='minibatch size 32')
plt.plot([1,2,3,4], m64, '--bo', label='minibatch size 64')
plt.title('sync SGD training with VGG16 (Convnet with ~140M parameters)')
plt.ylabel('time in seconds')
plt.xlabel('number of GPUs')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.xlim([0, 5])
plt.savefig('benchmark.png')
