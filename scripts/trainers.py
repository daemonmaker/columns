import os.path as op
import cPickle as pkl

import theano.tensor as TT


class Trainer(object):
    def __init__(self, teacher):
        self.teacher = teacher

    def train(self):
        """
        Trains the model.
        """
        raise NotImplementedError()


class JLGater(Trainer):
    def train(self):
        pass


class JLColumn(Trainer):
    def train(self):
        pass


class FitNetGater(Gater):
    def __init__(self, teacher, student_yaml, regressor_type='conv'):
        super(FitNetGater, self).__init__(teacher)

        if not op.exists(student_yaml):
            raise ValueError('YAML file does not exist: %s' % student_yaml)

        self.student_yaml = student_yaml

        self.regressor_type = regressor_type

    def train(self):
        import FitNets.scripts.fitnets_training as ft
        ft.main([self.yaml, self.regressor_type])


class FitNetColumn(Trainer):
    def __init__(self, teacher, student_yaml):
        super(FitNetColumn, self).__init__(teacher)

        if not op.exists(student_yaml):
            raise ValueError('YAML file does not exist: %s' % student_yaml)

        self.student_yaml = student_yaml

    def train(self):
        # Load column model
        # Load data
        # Train with

        pass
