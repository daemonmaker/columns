import os
import os.path as op
import argparse
import tempfile
import FitNets.scripts.fitnets_training as fitnets_training


def main():
    parser = argparse.ArgumentParser(
        description='Script for starting the fitnets_training.py. It creates'
        ' a random directory for the resuls, compiles the parameters for the'
        ' script, and finally starts the training.'
    )

    parser.add_argument(
        'yaml_path',
        help='The path to the yaml used for training.'
    )

    parser.add_argument(
        'column',
        help='The column being trained.'
    )

    parser.add_argument(
        '--scale_learning_rate',
        '-s',
        type=int,
        default=1,
        help='Whether to scale the learning rate of the bottom layers after'
        ' the first phase of training.'
    )

    args = parser.parse_args()

    # Determine the base directory wherein the models for the current column
    # should be stored.
    base_dir = op.join(os.getcwd(), args.column)
    if not op.exists(base_dir):
        os.mkdir(base_dir)

    # Create a temporary directory and set it as the current working directory
    tempfile.tempdir = base_dir
    new_dir = tempfile.mktemp()
    os.mkdir(new_dir)
    os.chdir(new_dir)

    # Start the training
    argv = [args.yaml_path, 'conv', args.scale_learning_rate]
    fitnets_training.main()


if __name__ == '__main__':
    main()
