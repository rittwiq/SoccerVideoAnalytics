# SoccerVideoAnalytics
OpenCV based project on tracking player locations

* Hey guys this is the git we'll be using for the project

* Any changes done lets record it here.

* We'll make a proper README once this is completed

* If we need to employ a SDLC model lets do incremental (thats right we're going full blown Agile!!)

* We'll be using C++ once we've formulated the project approach

* Make changes on separate files we will import your module and run in a main file

RULES
==========
# Please indent using tabs not spaces. (Commits will be reverted without tabs)

# Fork to your own repo

# Commit and Push to your own repo 

# Once done commit to this repo

# If possible document every method used


Logs
-----------------------------------------------

ROLES:

$ This is incremental so I am dividing the project into features pick up and work:
	
	* Check if frame is useless:
		Find if the frame is showing spectators or useless data.
		Assignee: (RITTWIK)

	* Camera Position:
		Find which part of the field the camera is pointing at.
		Assignee: (RITTWIK)

	* Player Detection:
		Segment player from field.
		Get good points to track
		Assignee: ()

	* Player Tracking:
		Track player movements in the field (Optic flow(Lucas Kanade)/Mean Shift/CAM shift/(SSR tracking)).
		Assignee: ()

	* Ball Tracking:
		Track ball location
		Assignee: ()

	* Player Identification:
		Identify player and team (Employ better approach than former)
		Assignee: ()

	* IF TIME: SKELETONIZE PLAYER:


get_image_data.py: Just captures frames for testing

30 Sept, 2017
===========================================

I have done alot of experimenting with finding camera coordinates

My approach is finding certain points aka (corner points, Penalty corners, D corners, Center line corners)

Center line corners are characterised by '+' gradient
D corners are characterised by '|-' gradient
Penalty corners andcorner points are similar to D but the green patch varies

To get these points on an ideal image involves a simple openCV API good features to track using Chi and
Tomasi's approach. 

For real images it is different as there are players and spectators

To remove spectators
From the former DIP project, I have used the field segmentation technique to remove spectators. I added an
extra flooding technique (https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/)

To remove the players
I have thresholded and then skeletonized the edge map. 
The edge map is then hough transformed and AND-ed with the edge map to get only the required line from the 
map and nothing else.

We then apply a Shi Tomasi Corner detection on this image

foo_1.py: Discovering how to skeletonize
global_threshold.py: Global thresholding approach to get a good value
template_corners.py: get corners in an ideal image
main.py: A failed waterfall approach to the project

hough_tranform.py: The current working model



+++++ Current Status +++++
*	This project has been completed and is available in bitbucket on request
