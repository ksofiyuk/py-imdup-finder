from collections import defaultdict
import cv2  
import numpy as np
import scipy, scipy.fftpack
import os
import sys


class ImageHash:
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """
    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return self._binary_array_to_hex(self.hash.flatten())

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError('Other hash must not be None.')
        if self.hash.size != other.hash.size:
            raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
        return (self.hash.flatten() != other.hash.flatten()).sum()

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        # this returns a 16+eps bit integer, intentionally shortening the information
        return sum([2**(i % 16) for i, v in enumerate(self.hash.flatten()) if v])
    
    @staticmethod
    def _binary_array_to_hex(arr):
        """
        internal function to make a hex string out of a binary array
        """
        h = 0
        s = []
        for i, v in enumerate(arr.flatten()):
            if v: 
                h += 2**(i % 8)
            if (i % 8) == 7:
                s.append(hex(h)[2:].rjust(2, '0'))
                h = 0
        return "".join(s)


def compute_phash(image, hash_size=8, highfreq_factor=4):
    """
    Perceptual Hash computation.

    Implementation follows http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

    @image must be a uint8 numpy array with BGR channels.
    """
    img_size = hash_size * highfreq_factor

    image = image[:, :, :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = cv2.resize(image, (img_size, img_size))
    pixels = np.array(image, dtype=np.float32).reshape((img_size, img_size))

    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return ImageHash(diff)


def compute_dir_phash(userpath):
    def is_image(filename):
        f = filename.lower()
        return f.endswith(".png") or f.endswith(".jpg") or \
            f.endswith(".jpeg") or f.endswith(".bmp")

    image_filenames = [os.path.join(userpath, path) 
                       for path in os.listdir(userpath) if is_image(path)]

    images_phash = defaultdict(list)
    for i, img_path in enumerate(sorted(image_filenames)):
        img = cv2.imread(img_path)
        if img is None:
            continue

        phash = compute_phash(img)
        images_phash[phash].append(img_path)

        if i % 500 == 0:
            print("{}/{}".format(i, len(image_filenames)))
            sys.stdout.flush()
            
    return images_phash


def get_impath2phash(phash2impath):
    phashes = {}
    for phash, imlist in phash2impath.items():
        for im_path in imlist:
            phashes[im_path] = phash
    return phashes
