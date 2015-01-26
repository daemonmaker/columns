import os
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
        '--scale_learning_rate',
        '-s',
        type=int,
        default=1,
        help='Whether to scale the learning rate of the bottom layers after'
        ' the first phase of training.'
    )

    args = parser.parse_args()

    # Create a temporary directory and set it as the current working directory
    tempfile.tempdir = os.getcwd()
    new_dir = tempfile.mktemp()
    os.mkdir(new_dir)
    os.chdir(new_dir)

    # Start the training
    argv = [args.yaml_path, 'conv', args.scale_learning_rate]
    fitnets_training.main(argv)


if __name__ == '__main__':
    main()
