#!/usr/bin/env python3

import argparse
import os
import sys
from vp_tree import VPTree
from image_hash import compute_dir_phash, get_impath2phash
import json


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Detect objects with DeepDetector')
    parser.add_argument('--images-dir', required=True, type=str)
    parser.add_argument('--radius', required=False, default=0, type=int)
    parser.add_argument('--save-to', required=True, type=str)
    parser.add_argument('--remove-duplicates', required=False, type=bool, default=False)

    args = parser.parse_args()
    return args


def get_duplicates(vp_tree, phash2impath, radius):
    used_hashes = set()
    dups = []

    if radius > 0:
        for i, (phash, imlist) in enumerate(phash2impath.items()):
            if i % 200 == 0:
                print('{}/{}'.format(i + 1, len(phash2impath)))
                sys.stdout.flush()

            if phash in used_hashes:
                continue

            neighbors = vp_tree.get_all_in_range(vp_tree, phash, radius)
            for d, thash in neighbors:
                used_hashes.add(thash)

            tdups = set()
            for dist, tphash in neighbors:
                tdups |= set(phash2impath[tphash])

            if len(tdups) > 1:
                dups.append(list(tdups))

    else:
        dups = [x for x in phash2impath.values() if len(x) > 1]

    return dups


def remove_duplicates(dups):
    for dups_group in dups:
        images_sizes = [os.path.getsize(path) for path in dups_group]
        max_indx = images_sizes.index(max(images_sizes))
        del dups_group[max_indx]

        for image_path in dups_group:
            os.remove(image_path)


if __name__ == '__main__':
    args = parse_args()

    phash2impath = compute_dir_phash(args.images_dir)

    impath2phash = get_impath2phash(phash2impath)
    phash_list = list(impath2phash.values())
    vp_tree = VPTree(phash_list, lambda p1, p2: p1-p2)

    dups = get_duplicates(vp_tree, phash2impath, args.radius)

    num_dups = sum(len(x) for x in dups) - len(dups)
    print('Found %d duplicates' % num_dups)

    if args.remove_duplicates:
        remove_duplicates(dups)

    with open(args.save_to, 'w') as f:
        json.dump(dups, f)