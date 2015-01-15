#! /usr/bin/env python

"""
Gater training - trains a gater
-Load train object with yaml_parse.load
-Train model with train.main_loop()

Sorting - Create a repository like the original but with only the data for each column
-Load the gater model
-Load the data
-Iterate over data and classify
-Store sample in appropriate repositories

Column training - trains a column
-Load the column yaml template
-Inject the data repository for the column being trained into yaml template
-Load train object with yaml_parse.load
-Train column with train.main_loop()

TODO
-Make JL layer
-Make JL gater YAML
"""


import argparse
import sys
import trainers
import sorters

from pylearn2.config import yaml_parse


def main(argv):
    available_gaters = ('FitNetGater', 'JLGater')

    parser = argparse.ArgumentParser(
        description='Tool for training nets using columns. One of the'
        ' train_gater or train_column options must be selected.'
    )
    parser.add_argument(
        '--train_gater',
        '-tg',
        default=None,
        help='Type of gater to train. One of %s' % list(available_gaters)
    )
    parser.add_argument(
        '--gater_yaml',
        '-gy',
        help='YAML describing the gater model.'
    )
    parser.add_argument(
        '--gater_model',
        '-gm',
        default=None,
        help='Name of the file to hold the gater model.'
    )
    parser.add_argument(
        '--train_columns',
        '-tc',
        nargs='+',
        type=int,
        default=-1,
        help='Which columns to train. None by default.'
    )
    parser.add_argument(
        '--column_yaml',
        '-cy',
        help='YAML describing the model to be used for the column(s).'
    )

    args = parser.parse_args()

    if not (args.train_gater or args.train_column):
        print 'One of the train_gater or train_column options must be selected.'
        exit()

    if args.train_gater:
        #with open(args.gater_yaml, 'r') as gy:
        #    gater_model = yaml_parse.load(gy)
        if not (args.gater_type in available_gaters):
            parser.print_help()
            raise ValueError(
                'Gater type must be one of %s' % (available_gaters)
            )

        # Instantiate trainer and train
        gater = eval(
            'trainers.%s' % args.gater_type
        )(args.gater_yaml, args.gater_model)
        gater.train()

    if type(args.train_columns) == list and len(args.train_columns) > 0:
        pass


if __name__ == "__main__":
    main(sys.argv)
