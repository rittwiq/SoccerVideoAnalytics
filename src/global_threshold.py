import cv2
import numpy as np
from sys import argv

"""
Global Thresholding
-----------------------------

Let t be an arbitrary threshold value
*	binary threshold image at t and copy to A
*	Invert A to B
*	find mean of A and B, Ua and Ub
*	t = Ua + Ub
"""

def get_mean(src):
	src_l = src.flatten().tolist()
	src_l = list(filter(lambda x: x>0, src_l))
	return sum(src_l) / len(src_l)

def global_threshold(src):
	t0 = 128
	t = 0
	while t0 != t:
		t0 = t
		ret, mask = cv2.threshold(src, 125, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		A = cv2.bitwise_and(src, src, mask = mask)
		B = cv2.bitwise_and(src, src, mask = mask_inv)
		u1 = get_mean(A)
		u2 = get_mean(B)
		t = (u1 + u2)/2
	return t

def main(argv, argc):
	src = cv2.imread(argv[1], 0)
	val = global_threshold(src)
	ret, dst = cv2.threshold(src, val, 255, cv2.THRESH_TOZERO)
	cv2.imshow("global_thresh", dst)
	cv2.waitKey(0)

if __name__ == "__main__":
    if(len(argv) >= 2):
        main(argv, len(argv))
        # debug(argv)
    else :
        print("Syntax: 1_foo.py /path/to/video/file ")