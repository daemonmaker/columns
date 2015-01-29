import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
from pylearn2.utils import serial

dataset_path = 'train_%d.pkl'
nclasses = 10
for idx in range(nclasses):
    dataset = serial.load(dataset_path % idx)
    plt.subplot(4, 3, idx)
    plt.hist(np.argmax(dataset.y, axis=1), nclasses)
    plt.title('Column: %d, Total Samples: %d' % (idx, dataset.y.shape[0]))
    #category_names = [
    #    'Airplane','Automobile', 'Bird', 'Cat', 'Deer',
    #    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
    #]
    #plt.legend(category_names)
plt.show()
