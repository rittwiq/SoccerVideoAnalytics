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
	"""
	get_means: get the medians (name is misleading will be changed in next revision) of two peaks
	dist: distribution of angles
	"""

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
	tmp = cv2.bitwise_and(tmp, gry)
	tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations = 1)
	return tmp

def find_corners(src, mask):
	"""
	find_corners: returns corner points
	src: binary image with only edges
	mask: binary image with the foreground
	"""

	gry = src.copy()
	gry = skeleton_morph(gry)
	tmp = lines_mask(gry)
	corners = cv2.goodFeaturesToTrack(tmp,25,0.01,100)
	corners = np.int0(corners)
	corners = np.array(list(filter(lambda pt: mask.item(pt.ravel()[1], pt.ravel()[0]), corners)))
	return corners

def dist(a, b):
	"""
	dist: euclidean distance for nn
	a, b: vectors
	
	Next revision may involve using other distance measures
	"""

	d = 0
	for i in range(len(a)):
		d+= (a[i] - b[i])*(a[i] - b[i])
	return d**0.5

def nn(avail, val):
	"""
	nn: returns the set of points which closely resembles the graph
	avail: available set from file
	val: testing set of values
	
	Subsequent revision may involve checking distance for noisey data (is noise return -1, -1)
	"""

	m   = 100000000
	m_x = 100000000
	m_y = 100000000
	for i in range(len(avail)):
		v = dist(avail[i]['degree'], val)
		if m > v :
			m   = v
			m_x = avail[i]['x']
			m_y = avail[i]['y']
	print("{} {}".format(m_x, m_y))
	return m_x, m_y


def extract_cam_pos(src, standard_corner_grad):
	"""
	extract_cam_pos: find where exactly the camera is
	src: original frame
	standard_corner_grad: gradient of corners of a premade soccer field
	"""
	
	[blue ,green ,red]= cv2.split(src)
	field = cv2.imread("Soccer_field.jpg")
	gry = blue.copy()
	mask = foreground_mask(src)
	gry = cv2.equalizeHist(gry)
	gry = cv2.bitwise_and(gry, gry, mask = mask)
	ret, gry = cv2.threshold(gry, 220, 255, cv2.THRESH_BINARY)
	corners = find_corners(gry, mask)
	src = cv2.bitwise_and(src, src, mask = mask)
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
		l = [abs(p12 - p11), abs(p02 - p11), abs(p01 - p11), abs(p00 - p11), abs(p10 - p11), 
		abs(p20 - p11), abs(p21 - p11), abs(p22 - p11)]
		cv2.circle(src, (x, y), 5, 128, -1)
		x, y = nn(standard_corner_grad, l)
		if x == y == -1:
			continue
		else :
			cv2.circle(field, (x, y), 5, (0, 0, 255), -1)
	cv2.imshow("foo", src)
	cv2.waitKey(0)
	cv2.imshow("foo", gry)
	cv2.waitKey(0)
	cv2.imshow("foo", field)
	cv2.waitKey(0)

def get_grad_data(filename = "field_grad_1.dat"):
	"""
	get_grad_data: get the gradient of corners of a premade soccer field
	file_name: persistent storage for gradent data
	
	FORMAT:
	N : Number of corners
	for each N:
		x, y: real world corner points
		degree: gradient along 0, 45, 90, 135, 180, 225, 270, 315

	Next revision may include assessing neighbouring pixels of corners for higher accuracy
	"""

	l = []
	f = open(filename, "r")
	N = int(f.readline())
	for i in range(N):
		x, y = f.readline().split(' ')
		obj = {'x': int(x), 'y': int(y)}
		deg = []
		for j in range(8):
			deg.append(int(f.readline()))
		obj['degree'] = deg
		l.append(obj)
	return l

def main(argv):
	standard_corner_grad = get_grad_data()
	for i in range(1, len(argv)):
		print("-----------------------------------------------------------------------------")
		print("\t\t\t{}\t\t\t".format(argv[i]))
		print("-----------------------------------------------------------------------------")
		src = cv2.imread(argv[i])
		extract_cam_pos(src, standard_corner_grad)

if __name__ == "__main__":
	if(len(argv) >= 2):
		main(argv)
		cv2.destroyAllWindows()
	else :
		print("Syntax: 1_foo.py /path/to/image/file ")