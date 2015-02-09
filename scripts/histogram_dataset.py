import numpy as np
import matplotlib.pyplot as plt
import cPickle as pkl
import argparse
import math
from pylearn2.utils import serial


def main():
    parser = argparse.ArgumentParser(
        description='Utility for visualizing the number of samples per class'
        ' in a given dataset.'
    )

    parser.add_argument(
        'total_columns',
        type=int
    )

    parser.add_argument(
        'columns',
        nargs='+',
        type=int
    )

    args = parser.parse_args()

    if args.total_columns <= 0:
        raise ValueError(
            'The total number of columns (%d) must'
            ' be positive.' % args.total_columns
        )

    dataset_path = 'train_%d.pkl'
    num_columns = len(args.columns)

    if num_columns > 3:
        num_rows = math.ceil(num_columns / 3.0)
        num_columns = 3
    else:
        num_rows = 1
        num_columns = min(num_columns, 3)

    for jdx, idx in enumerate(args.columns):
        dataset = serial.load(dataset_path % idx)
        plt.subplot(num_rows, num_columns, jdx+1)
        plt.hist(np.argmax(dataset.y, axis=1), args.total_columns)
        plt.title('Column: %d, Total Samples: %d' % (idx, dataset.y.shape[0]))
        #category_names = [
        #    'Airplane','Automobile', 'Bird', 'Cat', 'Deer',
        #    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
        #]
        #plt.legend(category_names)
    plt.show()


if __name__ == '__main__':
    main()
