import os
import os.path as op
import argparse
import tempfile
import subprocess
from FitNets.scripts import fitnets_training
from FitNets.scripts import fitnets_stage1


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
        'load_layer',
        type=int,
        default=None,
        help='Integer indicating the hint layer from which to start training.'
    )

    parser.add_argument(
        '--scale_learning_rate',
        '-s',
        type=float,
        default=0.05,
        help='Percentage to scale the learning rate of the bottom layers after'
        ' the first phase of training.'
    )

    parser.add_argument(
        '--two_stage',
        '-t',
        default=False,
        action='store_true',
        help='Whether to use two stage training.'
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

    scale_learning_rate = None
    if args.scale_learning_rate > 0:
        scale_learning_rate = args.scale_learning_rate

    if not args.two_stage:
        fitnets_training.execute(args.yaml_path, 'conv', scale_learning_rate)
    else:
        # Start the training
        fitnets_stage1.execute(args.yaml_path, 'conv')
        
        # If we make it this far then the first stage did not die of an exception.
        # As such we can issue a command to start the second stage.
        subprocess.check_call(
            "jobdispatch --gpu --duree=12:00:00 --mem=6G"
            " --env=THEANO_FLAGS=floatX=float32,device=gpu,force_device=True,base_compiledir='$RAMDISK_USER'"
            " --repeat_jobs=1"
            " python /home/webbd/columns/FitNets/scripts/fitnets_training.py"
            " %(yaml_path)s %(load_layer)d -lrs %(lr_scale)f"
            % {
                'yaml_path': args.yaml_path,
                'load_layer': args.load_layer,
                'lr_scale': scale_learning_rate
                }
            )

if __name__ == '__main__':
    main()
