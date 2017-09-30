import cv2
import numpy as np
from sys import argv
from math import degrees, pi
from statistics import median

def skeleton_morph(mask):
	"""
	skeleton_morph: image skeletonized image
	mask: binary image

	Skeletonization involving forming the basic thin structure of the image
	"""
	
	A = mask.copy()
	B = np.ones((5,5),np.uint8)
	C = np.zeros(A.shape, np.uint8)
	while cv2.countNonZero(A) != 0:
		AoB = cv2.morphologyEx(A, cv2.MORPH_OPEN, B, iterations = 1)
		C = C + (A - AoB)
		A = cv2.erode(A,B)
	C = cv2.morphologyEx(C, cv2.MORPH_CLOSE, B, iterations = 1)
	return C

def foreground_mask(src):
	"""
	foreground_mask: binary image with only foreground
	src: source int8_1c image ( grayscale image)
	"""
	
	lb = np.array([0,0,0])
	ub = np.array([150,200,150])
	mask = cv2.inRange(src,lb,ub)
	
	# Get the green component from the image
	[blue ,green ,red]= cv2.split(src)

	# Create a binary frame which consists of all highly green parts of the image 
	# (includes white)
	ret,green = cv2.threshold(green,120,255,cv2.THRESH_BINARY)
	
	# Remove the white from the image as only green is needed(field is green)
	img = cv2.bitwise_and(mask,green)
	kernel = np.ones((5,5),np.uint8)
	r_tmp = img.copy()
	cv2.floodFill(r_tmp, None, (0, 0), 255)
	tmp = cv2.bitwise_not(r_tmp)
	img = cv2.bitwise_or(tmp, img)
	mask = cv2.erode(img, kernel, iterations = 2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 10)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 10)
	return mask

def get_means(dist):
	t0 = 0
	t = pi / 2
	A = []
	B = []
	while t0 != t:
		t0 = t
		A = list(filter(lambda x: x>t, dist))
		B = list(filter(lambda x: x<=t, dist))
		u1 = sum(A)/(len(A)+1)
		u2 = sum(B)/(len(B)+1)
		t = (u1 + u2)/2
	return median(A), median(B)

def min_mask(src):
	"""
	min_mask: returns a binary image with minimum of all channels
	src: RGB image
	"""

	height, width, nchannels = src.shape
	mask = np.array(
		[[ min(src.item(i, j, 0), src.item(i, j, 1), src.item(i, j, 2)) for j in range(width)]
		for i in range(height)], dtype = "uint8")
	return mask

def lines_mask(src):
	"""
	lines_mask: returns binary image containing only lines
	src: binary image with only edges
	"""

	kernel = np.ones((5,5),np.uint8)
	height, width = src.shape
	gry = src.copy()
	tmp = np.zeros((height, width),np.uint8)
	lines = cv2.HoughLines(gry,1,np.pi/180,200, 0, 0)
	for line in lines:
		for rho, theta in line:
			a = np.cos(theta)
			b = np.sin(theta)
			if b == 0:
				b = 0.000001
			x0 = a*rho
			y0 = b*rho
			x1 = 0
			y1 = int(- (a/b)*x1 + (rho/b))
			x2 = width
			y2 = int(- (a/b)*x2 + (rho/b))
			cv2.line(tmp,(x1,y1),(x2,y2),128,1)
	return tmp

def find_corners(src):
	"""
	find_corners: returns corner points
	src: binary image with only edges
	mask: binary image with the foreground
	"""

	gry = src.copy()
	corners = cv2.goodFeaturesToTrack(gry,25,0.01,100)
	corners = np.int0(corners)
	return corners

def extract_cam_pos(src):
	[blue ,green ,red]= cv2.split(src)
	gry = red.copy()
	ret, gry = cv2.threshold(gry, 220, 255, cv2.THRESH_BINARY)
	cv2.imshow("foo", gry)
	cv2.waitKey(0)
	corners = find_corners(gry)
	print("{}".format(len(corners)))
	for i in corners:
		x,y = i.ravel()
		p00 = gry.item(y-1, x-1)
		p01 = gry.item(y  , x-1)
		p02 = gry.item(y+1, x-1)
		p10 = gry.item(y-1, x  )
		p11 = gry.item(y  , x  )
		p12 = gry.item(y+1, x  )
		p20 = gry.item(y-1, x+1)
		p21 = gry.item(y  , x+1)
		p22 = gry.item(y+1, x+1)
		print("{} {}".format(x, y))
		print("{}".format(abs(p12 - p11)))
		print("{}".format(abs(p02 - p11)))
		print("{}".format(abs(p01 - p11)))
		print("{}".format(abs(p00 - p11)))
		print("{}".format(abs(p10 - p11)))
		print("{}".format(abs(p20 - p11)))
		print("{}".format(abs(p21 - p11)))
		print("{}".format(abs(p22 - p11)))
		cv2.circle(src, (x, y), 5, 128, -1)
	cv2.imshow("foo", src)
	cv2.waitKey(0)

def main(argv):
	for i in range(1, len(argv)):
		src = cv2.imread(argv[i])
		extract_cam_pos(src)

if __name__ == "__main__":
	if(len(argv) >= 2):
		main(argv)
		cv2.destroyAllWindows()
	else :
		print("Syntax: 1_foo.py /path/to/image/file ")