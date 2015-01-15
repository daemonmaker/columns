#! /usr/bin/env python

import cPickle as pkl
import argparse
import os.path as op
import numpy as np
import sys

import theano
from pylearn2.utils import serial
from pylearn2.datasets.dense_design_matrix import (
    DenseDesignMatrix,
    DenseDesignMatrixPyTables
)


class Teacher(object):
    """
    This is the base class for labeling samples using a teacher model. It takes
    the location of a teacher model, compiles it's fprop, and has a utility for
    labeling a batch of inputs.
    """
    def __init__(self, teacher_path, temperature=1):
        assert(op.exists(teacher_path))
        self.teacher_path = teacher_path

        assert(temperature >= 1)
        self.temperature = temperature

        self._load_model()
        self._set_temperature()
        self._compile_fprop()

    def _load_model(self):
        """
        Loads the teacher model.
        """
        # Setup teacher to label samples
        self.model = serial.load(self.teacher_path)

    def _set_temperature(self):
        """
        Sets the temperature of the model.
        """
        # Update the weights and biases by dividing them by the temperature.
        # This effectively softens the predictions.
        sparams = self.model.layers[-1].get_param_values()
        sparams_relaxed = [item/float(self.temperature) for item in sparams]
        self.model.layers[-1].set_param_values(sparams_relaxed)

    def _compile_fprop(self):
        """
        Compiles the models fprop.
        """
        # TODO Can this be made more efficient -- specifically can the data
        # be stored in a shared variable so that it can be indexed?
        X = self.model.get_input_space().make_theano_batch()
        Y = self.model.fprop(X)
        self.fprop = theano.function([X], Y)

    def label(self, input):
        """
        Label sample with the teacher and return the result.
        """
        return self.fprop(input)

    def label_dataset(self, base_dataset, label_path, batch_size=32):
        """
        Method for labeling entire dataset at once.
        """
        assert(dataset.X.shape[0] % batch_size == 0)
        base_dataset.X = base_dataset.X.astype(theano.config.floatX)  # IS THIS THE CORRECT WAY TO DO THIS?

        # Make space for the labels
        y_preds = np.zeros(
            (base_dataset.X.shape[0], self.model.get_output_space().dim)
        )

        # Iterate over the data
        iterator = base_dataset.iterator(
            mode='sequential',
            batch_size=batch_size,
            data_specs=self.model.cost_from_X_data_specs()
        )

        idx = 0
        for item in iterator:
            if idx % 1000 == 0:
                print idx, '\r'
                sys.stdout.flush()
            x_arg, y_arg = item
            idx_end = idx + y_arg.shape[0]
            y_preds[idx:idx_end] = self.fprop(x_arg)
            idx = idx_end

        # Save the labels -- this is a small hack but it preventsme from
        # reimplementing the PyTables basic functionality.
        label_file = DenseDesignMatrixPyTables(X=y_preds)
        hdf5fh, gcolumns = label_file.init_hdf5(
            label_path,
            (y_preds.shape, dataset.y.shape)
        )
        label_file.resize(hdf5fh, 0, y_preds.shape[0])
        label_file.fill_hdf5(hdf5fh, y_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tool for labeling a dataset using a teacher model.'
    )
    parser.add_argument(
        'teacher_path',
        help='Path to the pickle containing the teacher model.'
    )
    parser.add_argument(
        'dataset_path',
        help='Path to the pickle containing the dataset.'
    )
    parser.add_argument(
        '--destination',
        '-d',
        default=None,
        help='File into which the results should be stored.'
    )

    args = parser.parse_args()

    print args.dataset_path[-4:]
    assert(args.teacher_path[-4:] == '.pkl')
    assert(args.dataset_path[-4:] == '.pkl')
    assert(op.exists(args.dataset_path))

    # Create new file name
    label_path = args.destination
    if args.destination is None:
        dir_path, file_name = op.split(args.dataset_path)
        label_path = file_name[:-4] + '_predicted_by_%s.pkl'
        label_path %= op.split(args.teacher_path)[1][:-4]

    dataset = pkl.load(open(args.dataset_path, 'rb'))

    teacher = Teacher(args.teacher_path)

    #with open(args.destination, 'rb') as fh:
    #    labels = teacher.label_dataset(dataset, fh)
    teacher.label_dataset(
        dataset,
        label_path,
        batch_size=100
    )
