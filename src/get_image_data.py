import cv2
import numpy as np
from sys import argv



def main(argv, argc):
	img_list = []
	cap = cv2.VideoCapture(argv[1])
	cap.set(2,0.28)
	ret, frame = cap.read()
	print("press q to quit")
	print("press ' ' to capture frame")
	while ret:
		ret, frame = cap.read()
		cv2.imshow("Video", frame)
		ch = cv2.waitKey(1)
		if ch == ord(' '):
			img_list.append(frame)
		if ch == ord('q'):
			break
	for i in range(len(img_list)):
		cv2.imwrite("match-{}-.jpg".format(i), img_list[i])

if __name__ == "__main__":
    if(len(argv) >= 2):
        main(argv, len(argv))
    else :
        print("Syntax: get_image_data.py /path/to/video/file ")