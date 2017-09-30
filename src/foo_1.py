import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from operator import itemgetter
from functools import reduce

"""
Phase-1

Soccer match based video analytics

*	Find corners using harris
"""

def skeleton_mask(mask):
	A = mask.copy()
	B = np.ones((5,5),np.uint8)
	tmp = np.zeros(A.shape, np.uint8)
	while cv2.countNonZero(A) != 0:
		AoB = cv2.morphologyEx(A, cv2.MORPH_OPEN, B, iterations = 1)
		tmp = tmp + (A - AoB)
		A = cv2.erode(A,B)

	tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, B, iterations = 1)
	cv2.imshow("skeleton", tmp)
	cv2.waitKey(0)
	return tmp

def get_mean(src):
	src_l = src.flatten().tolist()
	src_l = list(filter(lambda x: x>0, src_l))
	return sum(src_l) / len(src_l)

def global_threshold(src):
	t0 = 128
	t = 0
	while t0 != t:
		t0 = t
		ret, mask = cv2.threshold(src, 200, 255, cv2.THRESH_BINARY)
		mask_inv = cv2.bitwise_not(mask)
		A = cv2.bitwise_and(src, src, mask = mask)
		B = cv2.bitwise_and(src, src, mask = mask_inv)
		u1 = get_mean(A)
		u2 = get_mean(B)
		t = (u1 + u2)/2
	return t


def get_mask(src):
	# create a mask of all white. Required because the green component is also high in white
	
	lb = np.array([0,0,0])
	ub = np.array([150,200,150])
	mask = cv2.inRange(src,lb,ub)
	
	# Get the green component from the image
	
	[blue,green,red]= cv2.split(src)

	# Create a binary frame which consists of all highly green parts of the image 
	# (includes white)
	
	ret,green = cv2.threshold(green,120,255,cv2.THRESH_BINARY)
	
	# Remove the white from the image as only green is needed(field is green)
	
	img = cv2.bitwise_and(mask,green)
	kernel = np.ones((5,5),np.uint8)

	cv2.imshow('1_foo.py', img)
	cv2.waitKey(0)

	r_tmp = img.copy()
	cv2.floodFill(r_tmp, None, (0, 0), 255)
	tmp = cv2.bitwise_not(r_tmp)
	img = cv2.bitwise_or(tmp, img)

	cv2.imshow('1_foo.py', img)
	cv2.waitKey(0)

	mask = cv2.erode(img, kernel, iterations = 2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 10)
	cv2.imshow('1_foo.py', mask)
	cv2.waitKey(0)

	#Return the plane which now consists of only the field
	return mask

def main(argv, argc):
	cap = cv2.VideoCapture(argv[1])
	ret, frame = cap.read()
	while ret:
		get_feature_vector(frame)
		ret, frame = cap.read()

def debug(argv):
	src = cv2.imread(argv[1])
	height, width, nchannels = src.shape
	mask = get_mask(src)
	
	gry = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	gry = cv2.bitwise_and(gry, gry, mask = mask)
	
	val = global_threshold(gry)
	ret, gry = cv2.threshold(gry, 150, 255, cv2.THRESH_BINARY)

	gry = skeleton_mask(gry)

	tmp = np.zeros((height, width, nchannels))
	lines = cv2.HoughLines(gry,1,np.pi/180,200)

	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho

		x1 = 0
		y1 = int(- (a/b)*x1 + (rho/b))
		
		x2 = width
		y2 = int(- (a/b)*x2 + (rho/b))

		cv2.line(gry,(x1,y1),(x2,y2),255,1)
		cv2.line(src,(x1,y1),(x2,y2),255,1)

	# kernel = np.ones((5,5),np.uint8)
	# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	cv2.imshow('Final Mask', gry)
	cv2.waitKey(0)


	# kernel = np.zeros((3,3),np.uint8)
	# kernel[0][0] = 1
	# kernel[1][1] = 1
	# kernel[2][2] = 1
	# kernel[2][0] = 1
	# kernel[0][2] = 1

	# print(kernel)
	# erosion = cv2.erode(gry,kernel)

	# kernel = np.ones((5,5),np.uint8)
	# erosion = cv2.dilate(erosion, kernel)

	# cv2.imshow('foo_1.py', erosion)
	# cv2.waitKey(0)

	corners = cv2.goodFeaturesToTrack(gry,25,0.01,10)
	corners = np.int0(corners)

	for i in corners:
		x,y = i.ravel()
		cv2.circle(src,(x,y),5,[255, 0, 0],1)

	dst = cv2.cornerHarris(gry, 2, 3, 0.04)
	dst = cv2.dilate(dst, None)

	src[dst>0.01*dst.max()] = [0, 0, 255]
	cv2.imshow('foo_1.py', src)
	cv2.waitKey(0)

if __name__ == "__main__":
	if(len(argv) >= 2):
		# main(argv, len(argv))
		debug(argv)
		cv2.destroyAllWindows()
	else :
		print("Syntax: 1_foo.py /path/to/video/file ")