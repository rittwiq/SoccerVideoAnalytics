import cv2
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from operator import itemgetter
from functools import reduce

"""
Phase-1

Soccer match based video analytics

Features to extract:
*	Field Lines - 
	1.	Center line
	2.  Goal line
	3. 	Throw in line
	4.	Penalty line
* Goal post (If possible)
"""

def triangle_thresholding(src):
	"""
	%     Triangle algorithm
%     This technique is due to Zack (Zack GW, Rogers WE, Latt SA (1977), 
%     "Automatic measurement of sister chromatid exchange frequency", 
%     J. Histochem. Cytochem. 25 (7): 74153, )
%     A line is constructed between the maximum of the histogram at 
%     (b) and the lowest (or highest depending on context) value (a) in the 
%     histogram. The distance L normal to the line and between the line and 
%     the histogram h[b] is computed for all values from a to b. The level
%     where the distance between the histogram and the line is maximal is the 
%     threshold value (level). This technique is particularly effective 
%     when the object pixels produce a weak peak in the histogram.

%     Use Triangle approach to compute threshold (level) based on a
%     1D histogram (lehisto). num_bins levels gray image. 

%     INPUTS
%         lehisto :   histogram of the gray level image
%         num_bins:   number of bins (e.g. gray levels)
%     OUTPUT
%         level   :   threshold value in the range [0 1];

	"""
	hist = cv2.calcHist([src], [0], None, [256], [0, 256]).tolist()
	h = max(hist)
	max_list = [i for i, j in enumerate(hist) if j==h]
	xmax = round(reduce(lambda x, y: x+y, max_list) / len(max_list))
	h = hist[xmax][0]
	indi = list(filter(lambda x: x[1][0]>0.0, enumerate(hist)))
	fnz = indi[0][0]
	lnz = indi[len(indi) - 1][0]
	lspan = xmax - fnz
	rspan = lnz - xmax
	isflip = rspan>lspan
	if isflip:
		hist.reverse()
		a = 256 - lnz + 1
		b = 256 - xmax + 1
	else:
		a = fnz
		b = xmax
	m = h / (b-a)
	x1 = range(0, b-a)
	y1 = [hist[i+a][0] for i in x1]
	beta = [ (x1[i] + y1[i])/m for i in range(len(x1))]
	x2 = [i/(m+1/m) for i in beta]
	y2 = [m*i for i in x2]
	L = [((y2[i]-y1[i])**2 + (x2[i]-x1[i])**2)**0.5 for i in range(len(x1))]
	max_L = max(L)
	level_list = [i for i, j in enumerate(L) if j==max_L]
	level = a + round(reduce(lambda x, y: x+y, level_list) / len(level_list))
	return level

def get_edge_map(src):
	val = triangle_thresholding(src)
	edges = cv2.Canny(src, val, 255)
	ret, img = cv2.threshold(src, val, 255, cv2.THRESH_BINARY)
	plt.subplot(131), plt.imshow(src, "gray"), plt.title('Original image')
	plt.subplot(132), plt.hist(src.ravel(), 256), plt.title('Histogram')
	plt.subplot(133), plt.imshow(img, "gray"), plt.title('Thresholded image')
	plt.show()
	return edges

def get_line_features(edges):
	lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
	return lines

def get_circle_features(edges):
	img = edges.copy()
	height, width = edges.shape[:2]
	templ = np.zeros([int(height/4), int(width/4), 1], np.uint8)
	cv2.circle(templ, (int(height/8), int(width/8)), int(width/8), (255, 255, 255), 2)
	res = cv2.matchTemplate(edges, templ, cv2.TM_CCOEFF)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + int(width/4), top_left[1] + int(height/4))
	cv2.rectangle(img ,top_left, bottom_right, 128, 2)

	plt.subplot(121), plt.imshow(edges, "gray"), plt.title("Original Image")
	plt.subplot(122), plt.imshow(res, "gray"), plt.title("Probablity Image")
	plt.show()
	plt.subplot(121), plt.imshow(edges, "gray"), plt.title("Original Image")
	plt.subplot(122), plt.imshow(img, "gray"), plt.title("Detected Object")
	plt.show()
	# plt.subplot(131), plt.imshow(edges), plt.title("Original Image")
	# circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50,
		# param1=50,param2=30,minRadius=0,maxRadius=0)
	# return circles

def get_feature_vector(src):
	"""
	get_feature_vector: list of contours
	src: image source
	"""
	b, g, r = cv2.split(src)
	dst = cv2.addWeighted(b, 0.3333, g, 0.3333, 0)
	dst = cv2.addWeighted(dst, 1, r, 0.3333, 0)
	edges = get_edge_map(dst)
	kernel = np.ones((5,5),np.uint8)
	# erosion = cv2.erode(edges ,kernel,iterations = 1)
	edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(30,30)))
	plt.subplot(121), plt.imshow(dst, "gray"), plt.title('Original image')
	plt.subplot(122), plt.imshow(edges, "gray"), plt.title('Edge map')
	plt.show()

	lines = get_line_features(edges)
	# circles = get_circle_features(edges)
	for rho, theta in lines[0]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(src, (x1,y1),(x2,y2),(0,0,255),2)
	# circles = np.uint16(np.around(circles))
	# for i in circles[0,:]:
	# 	cv2.circle(src,(i[0],i[1]),i[2],(0,255,0),2)
	# 	cv2.circle(src,(i[0],i[1]),2,(0,0,255),3)
	plt.subplot(121), plt.imshow(dst, "gray"), plt.title('Edge image')
	plt.subplot(122), plt.imshow(src), plt.title('Lines image')
	plt.show()

def main(argv, argc):
	cap = cv2.VideoCapture(argv[1])
	ret, frame = cap.read()
	while ret:
		get_feature_vector(frame)
		ret, frame = cap.read()

def debug(argv, argc):
	src = cv2.imread(argv[1])
	get_feature_vector(src)

if __name__ == "__main__":
    if(len(argv) >= 2):
        # main(argv, len(argv))
        debug(argv, len(argv))
    else :
        print("Syntax: 1_foo.py /path/to/video/file ")