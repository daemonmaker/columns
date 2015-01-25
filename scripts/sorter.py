import os
import os.path as op
import cPickle as pkl
import argparse
import numpy as np

from pylearn2.utils import (
    serial,
    py_integer_types,
    string_utils
)
from pylearn2.config import yaml_parse
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import VectorSpace

import theano
from theano import config


class Sorter(object):
    """
    Base class for common sorting functionality.
    """
    def __init__(
        self,
        total_columns,
        model_file,
        dataset,
        temperature,
        representation_layer=-1,
        columns=[],
        batch_size=1
    ):
        assert(total_columns > 0)
        self.total_columns = total_columns

        assert(temperature >= 1)
        self.temperature = temperature

        assert dataset.X.shape[0] % batch_size == 0

        assert(batch_size >= 1)
        self.batch_size = batch_size

        assert(dataset is not None)
        self.dataset = dataset

        print columns
        assert(type(columns) == list or type(columns) == tuple)
        # Sort for all columns by default
        if len(columns) == 0:
            columns = range(total_columns)
        self.columns = columns

        # Make a map identifing which column goes into which container
        self.idx_map = {}
        for idx, column in enumerate(self.columns):
            self.idx_map[column] = idx

        if not (model_file and op.exists(model_file)):
            raise ValueError('Model file does not exist: %s' % model_file)
        self.model_file = model_file

        # Load model
        self.model = serial.load(self.model_file)
        self.model.set_batch_size(batch_size)

        # Update the weights and biases by dividing them by the temperature.
        # This effectively softens the predictions.
        sparams = self.model.layers[-1].get_param_values()
        sparams_relaxed = [item/float(self.temperature) for item in sparams]
        self.model.layers[-1].set_param_values(sparams_relaxed)

        # Compile fprop
        X = self.model.get_input_space().make_theano_batch()
        representation = X
        for idx in range(representation_layer):
            representation = self.model.layers[idx].fprop(representation)

        # Grab the output space of the layer we're using for the
        # representation.
        if representation_layer > -1:
            self.representation_space = self.model.layers[
                representation_layer-1
            ].get_output_space()

        # If we're using the input then we need to know what space it is in.
        else:
            self.representation_space = self.model.get_input_space()

        #storage_space = VectorSpace(
        #    representation_space.get_total_dimension()
        #    #representation.flatten().shape[0] / float(self.batch_size)
        #)

        # When we're not using a vector space then we need to make sure the
        # the data has the shape ('b', 0, 1, 'c')
	if (
            not isinstance(self.representation_space, VectorSpace)
            and self.representation_space.axes != ('b', 0, 1, 'c')
        ):
            reshape_param = (
                self.representation_space.axes.index('b'),
                self.representation_space.axes.index(0),
                self.representation_space.axes.index(1),
                self.representation_space.axes.index('c'),
            )
            if reshape_param != (0, 1, 2, 3):
                self.axes = ('b', 0, 1, 'c')
                representation = representation.dimshuffle(reshape_param)

        self.fprop = theano.function(
            [X],
            [
                self.model.fprop(X),
                representation,
                #representation_space.format_as(
                #    representation,
                #    storage_space
                #)
            ]
        )

    def full_sort(self, param):
        """
        Runs each sample through the model to identify where they should be
        routed and then copies the data into separate containers for each
        column.
        """
        self._validate_parameter(param)

        # Probe the model to determine the representation shape
        iterator = self.dataset.iterator(
            mode='sequential',
            batch_size=self.batch_size,
            data_specs=self.model.cost_from_X_data_specs()
        )
        item = iterator.next()
        junk_labels, sample_representations = self.fprop(item[0])

        # Now make space to store the representations
        #representations = np.zeros((
        #    self.dataset.X.shape[0],
        #    sample_representations.shape[1]
        #))
        # if not isinstance(self.representation_space, VectorSpace):
        #     sample_representations_shape = sample_representations.shape
        #     data_shape = []
        #     for idx, dim in enumerate(self.representation_space.axes):
        #         if dim != 'b':
        #             data_shape.append(sample_representations_shape[idx])
        #     data_shape = tuple(data_shape)
        # else:
        #     data_shape = sample_representations.shape[1:]
        data_shape = sample_representations.shape[1:]

        representations = np.zeros((self.dataset.X.shape[0],) + data_shape)

        # Create space for column mapping and labels
        destinations = np.zeros((self.dataset.X.shape[0], len(self.columns)))
        soft_labels = np.zeros((self.dataset.X.shape[0], self.total_columns))

        # Iterate over the data
        iterator = self.dataset.iterator(
            mode='sequential',
            batch_size=self.batch_size,
            data_specs=self.model.cost_from_X_data_specs()
        )

        offset = 0
        for item in iterator:
            x_arg, y_arg = item
            batch_labels, representation = self.fprop(x_arg)
            actual_batch_size = batch_labels.shape[0]
            soft_labels[offset:(offset+actual_batch_size)] = batch_labels
            representations[offset:(offset+actual_batch_size)] = representation
            destinations[
                offset:(offset+actual_batch_size)
            ] = self._sort(x_arg, y_arg, batch_labels)
            offset += actual_batch_size

        return destinations, soft_labels, representations

    def pickle_datasets(
        self,
        results_dir,
        name_template,
        destinations,
        soft_labels,
        representations,
        percentage_valid=0.1
    ):
        """
        Makes a dataset given a copy of the original data, a set of soft
        labels, and a mapping of which samples go to which columns.
        """
        assert(percentage_valid > 0.0 and percentage_valid < 1.0)
        for idx in xrange(len(self.columns)):
            train_destination = op.join(
                results_dir,
                name_template % ('train', idx, 'pkl')
            )
            valid_destination = op.join(
                results_dir,
                name_template % ('valid', idx, 'pkl')
            )

            print 'Creating %s' % train_destination

            # Identify samples needed
            row_idxs = destinations[:, self.idx_map[idx]]
            rows = np.where(row_idxs > 0)[0]
            separator = int((1 - percentage_valid)*len(rows))
            assert(separator > 0 and separator < len(rows))

            # Create targets
            y = np.zeros((self.dataset.y.shape[0], self.total_columns))
            for jdx in xrange(self.dataset.y.shape[0]):
                y[jdx][self.dataset.y[jdx]] = 1

            # Create and save the train dataset
            if isinstance(self.representation_space, VectorSpace):
                train_dataset = DenseDesignMatrix(
                    X=representations[rows[:separator]],
                    #y=self.dataset.y[rows[:separator]],
                    #y=soft_labels[rows[:separator]],
                    y=y[rows[:separator]],
                )
            else:
                train_dataset = DenseDesignMatrix(
                    topo_view=representations[rows[:separator]],
                    #y=self.dataset.y[rows[:separator]],
                    #y=soft_labels[rows[:separator]],
                    y=y[rows[:separator]],
                    axes=self.axes
                )
            train_dataset.soft_labels = soft_labels[rows[:separator]]
            train_dataset.use_design_loc(
                op.join(results_dir, name_template % ('train', idx, 'npy'))
            )
            serial.save(train_destination, train_dataset)
            del train_dataset

            # Create and save the valid dataset
            if isinstance(self.representation_space, VectorSpace):
                valid_dataset = DenseDesignMatrix(
                    X=representations[rows[separator:]],
                    #y=self.dataset.y[rows[separator:]],
                    #y=soft_labels[rows[separator:]],
                    y=y[rows[separator:]],
                )
            else:
                valid_dataset = DenseDesignMatrix(
                    topo_view=representations[rows[separator:]],
                    #y=self.dataset.y[rows[separator:]],
                    #y=soft_labels[rows[separator:]],
                    y=y[rows[separator:]],
                    axes=self.axes
                )
            valid_datset.soft_labels = soft_labels[rows[separator:]]
            valid_dataset.use_design_loc(
                op.join(results_dir, name_template % ('valid', idx, 'npy'))
            )
            serial.save(valid_destination, valid_dataset)
            del valid_dataset

    def _map_idx(self, idx):
        return self.idx_map[idx]

    def _sort(self, inputs, labels, outputs):
        """
        Abstract method that identifies which samples should be copied for
        each column according to the rule defined by the derived class.
        """
        raise NotImplementedError(
            'The _sort method is to be implmented by a class that derives'
            ' the Sorter class.'
        )

    def _validate_parameter(self, param):
        """
        Abstract method that validates the sorting parameter.
        """
        raise NotImplementedError(
            'The _validate_parameter method is to be implmented by a class'
            ' that derives the Sorter class.'
        )


class ThresholdSorter(Sorter):
    """
    Sorts data for training columns. Data is sorted according to the
    specified threshold. If the threshold does not result in at least
    two columns being selected then the top two columns are selected.
    """
    def __init__(
        self,
        total_columns,
        model_file,
        dataset,
        temperature,
        representation_layer=-1,
        columns=[],
        batch_size=1,
        min_columns=2
    ):
        assert(min_columns > 0)
        self.min_columns = min_columns

        super(ThresholdSorter, self).__init__(
            total_columns=total_columns,
            model_file=model_file,
            dataset=dataset,
            representation_layer=representation_layer,
            temperature=temperature,
            columns=columns,
            batch_size=batch_size
        )

    def _validate_parameter(self, threshold):
        if threshold is None:
            threshold = 0.3
            print 'Threshold not set. Defaulting to 30%.'

        if not (threshold > 0.0 and threshold <= 1.0):
            raise ValueError('Threshold should be in (0.0, 1.0].')

        self.threshold = threshold

    def _sort(self, inputs, labels, outputs):
        destinations = np.zeros((outputs.shape[0], len(self.columns)))

        # Sort by activation level
        argsorted_idxs = np.argsort(outputs)

        # Copy each sample to the appropriate columns data
        for row_idx, row in enumerate(outputs):
            idxs = argsorted_idxs[row_idx]

            # Determine whether enough columns are above the desired threshold
            # and lower the threshold if not.
            threshold = self.threshold
            temp_threshold = row[idxs[-self.min_columns]]
            if temp_threshold < self.threshold:
                threshold = temp_threshold

            # Identify columns to which the sample should be copied. Note we
            # only want to copy the sample to the columns for which we are
            # sorting.
            column_idxs = set(
                np.where(row >= threshold)[0]
            ).intersection(self.columns)

            # Mark the sample to be copied to the appropriate outputs.
            if len(column_idxs) > 0:
                destinations[row_idx][map(self._map_idx, column_idxs)] = 1

        return destinations


class ColumnsSorter(Sorter):
    """
    Sorts data for training columns. Data is sorted into specified number
    of columns by selecting the top num_columns.
    """
    def _validate_parameter(self, num_columns):
        if not num_columns:
            num_columns = 2
            print 'Number of columns not specified. Defaulting to 2.'

        # How to check the largest possible number of columns?
        if num_columns < 1:
            raise ValueError('The number of columns must be at least 1.')

        self.num_columns = num_columns

    def _sort(self, inputs, labels, outputs):
        destinations = np.zeros((outputs.shape[0], len(self.columns)))

        # Identify columns to which the sample should be copied. Note we
        # only want to copy the sample to the columns for which we are
        # sorting.
        argsorted_idxs = np.argsort(outputs)[:, -self.num_columns:]

        for row_idx, row in enumerate(outputs):
            column_idxs = set(
                argsorted_idxs[row_idx]
            ).intersection(self.columns)

            # Mark the sample to be copied to the appropriate outputs.
            if len(column_idxs) > 0:
                destinations[row_idx][map(self._map_idx, column_idxs)] = 1

        return destinations


def main():
    sort_types = ('Threshold', 'Columns')

    parser = argparse.ArgumentParser(
        description='Tool for sorting data using a trained gater.'
    )
    parser.add_argument(
        'total_columns',
        type=int,
        help='The total number of available columns.'
    )
    parser.add_argument(
        'gater_model',
        help='Name of the file to hold the gater model.'
    )
    parser.add_argument(
        'dataset',
        help='The dataset that should be sorted by the gater.'
    )
    parser.add_argument(
        '--results_dir',
        '-r',
        default=None,
        help='Directory into which the results should be stored. The default'
        'The name will be the present working directory and the name will be'
        ' <the name of the data_pkl>_<column index>.pkl.'
    )
    parser.add_argument(
        '--temperature',
        '-T',
        type=int,
        default=1,  # Don't soften
        help='The temperature to use to soften the data.'
    )
    parser.add_argument(
        '--type',
        '-t',
        default='Threshold',
        help='Type of data sorting to use.'
        ' One of %s. Defaults to Threshold.' % list(sort_types)
    )
    parser.add_argument(
        '--parameter',
        '-p',
        default=None,
        help='Value in (0, 1] if sorting by threshold and [1, N] where N is'
        ' the number of columns supported by the gater if sorting by columns.'
    )
    parser.add_argument(
        '--columns',
        '-c',
        type=int,
        nargs='+',
        default=[],  # All columns
        help='A list of the columns for which this process should sort the'
        ' data. Multiple processes can be executed in parallel to speed'
        ' sorting. By default sorts the data for all columns.'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        default=100,
        help='The size of the batches to process at once.'
    )
    parser.add_argument(
        '--representation_layer',
        '-l',
        type=int,
        default=-1,
        help='The layer of the gater/teacher from which the representation'
        ' should be extraced. If this is set to -1, as defaulted, then the'
        ' the original input will be stored.'
    )
    parser.add_argument(
        '--no_save',
        '-n',
        default=False,
        action='store_true',
        help='Whether to save the results.'
    )

    args = parser.parse_args()

    if not args.type in sort_types:
        parser.print_help()
        raise ValueError('Sort type must bye one of %s' % list(sort_types))

    # TODO With a little import magic this could import any pylearn2 dataset
    pylearn2_data_path = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
    if args.dataset == 'CIFAR100':
        #from pylearn2.datasets.cifar100 import CIFAR100
        #dataset = CIFAR100('train')
        from pylearn2.datasets.zca_dataset import ZCA_Dataset
        data_dir = op.join(
            pylearn2_data_path,
            'cifar100',
            'pylearn2_gcn_whitened'
        )
        dataset = ZCA_Dataset(
            preprocessed_dataset=serial.load(op.join(data_dir, 'train.pkl')),
            preprocessor=serial.load(op.join(data_dir, 'preprocessor.pkl')),
            start=0,
            stop=40000,
            axes=['c', 0, 1, 'b']
        )
    elif args.dataset == 'CIFAR10':
        from pylearn2.datasets.cifar10 import CIFAR10
        dataset = CIFAR10('train')
        from pylearn2.datasets.zca_dataset import ZCA_Dataset
        data_dir = op.join(
            pylearn2_data_path,
            'cifar10',
            'pylearn2_gcn_whitened'
        )
        dataset = ZCA_Dataset(
            preprocessed_dataset=serial.load(op.join(data_dir, 'train.pkl')),
            preprocessor=serial.load(op.join(data_dir, 'preprocessor.pkl')),
            start=0,
            stop=40000,
            axes=['c', 0, 1, 'b']
        )

    else:
        raise ValueError('Unknown dataset: %s' % args.dataset)

    # Instantiate the sorter
    sorter = eval('%sSorter' % args.type)(
        total_columns=args.total_columns,
        model_file=args.gater_model,
        dataset=dataset,
        temperature=args.temperature,
        representation_layer=args.representation_layer,
        columns=args.columns,
        batch_size=args.batch_size
    )

    # Calculate the sorting for the data, the softened labels, and the
    # representations.
    (
        destinations,
        soft_labels,
        representations
    ) = sorter.full_sort(args.parameter)

    # Save off the data
    if not args.no_save:
        name_template = '%s_%d.%s'
        results_dir = args.results_dir
        if results_dir is None:
            results_dir = os.getcwd()
        sorter.pickle_datasets(
            results_dir,
            name_template,
            destinations,
            soft_labels,
            representations
        )

if __name__ == '__main__':
    main()
