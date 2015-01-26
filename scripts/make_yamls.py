#! /usr/bin/env python

import argparse
import os
import os.path as op

def main():
    parser = argparse.ArgumentParser(
        description='Tool for generating yaml files for training columns.'
    )
    parser.add_argument(
        'template_yaml',
        help='Location of the template YAML file.'
    )
    parser.add_argument(
        'total_columns',
        type=int,
        help='The total number of columns. One YAML will be made for each.'
    )
    parser.add_argument(
        'dataset_dir',
        default='/data/lisatmp2/webbd/columns/datasets/CIFAR10',
        help='The directory where the training and validation sets are stored.'
    )
    parser.add_argument(
        'teacher_path',
        default='/data/lisatmp3/romerosa/compression/teachers/trained_models/CIFAR10/CIFAR10teacher_maxout_aug.pkl',
        help='The path to the teacher model.'
    )
    parser.add_argument(
        'model_name',
        default='fitnet11_9conv_2fc_',
        help='Name to give the trained models.'
    )

    args = parser.parse_args()

    assert(op.exists(args.template_yaml))
    path, template_name = op.split(args.template_yaml)
    if len(path) == 0:
        path = os.getcwd()
    template_name = template_name[:-5] + '_%d.yaml'
    with open(args.template_yaml, 'r') as template_fh:
        template_data = template_fh.read()
        for idx in xrange(args.total_columns):
            new_yaml_path = op.join(path, template_name % idx)
            print 'Making %s' % new_yaml_path
            fh = open(new_yaml_path, 'w')
            fh.write(template_data % {
                    'dataset_dir': args.dataset_dir,
                    'teacher_path': args.teacher_path,
                    'model_name': args.model_name,
                    'idx': idx
            })
            fh.close()

if __name__ == '__main__':
    main()
