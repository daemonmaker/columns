#! /usr/bin/env python

import sys
import re
import shutil
import glob
import os.path as op
from itertools import product


def main(dir_path):
    dirs = glob.glob(op.join(dir_path, 'pylearn2_gcn_whitened_*'))
    files = ('test', 'train')

    for params in product(dirs, files):
        full_dir = params[0]
        dir_base, dir_name = op.split(full_dir)
        file_name = params[1]
        file_root = op.join(full_dir, file_name)

        pkl_path = '%s.pkl' % file_root
        data_path = '%s.npy' % file_root

        # Backup the file
        shutil.copy(pkl_path, '%s.bak' % pkl_path)

        with open(pkl_path, 'rb') as fh:
            data = fh.read()

        fixed = re.sub(
            ('/u/ebrahims/tinyImages:/u/ebrahims/tinyImages:/%s/%s.npy' % (dir_name, file_name)),
            '%s/%s.npy' % (full_dir, file_name),
            data
        )

        with open(pkl_path, 'wb') as fh:
            fh.write(fixed)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Path to files required."
        sys.exit(1)

    main(sys.argv[1])
