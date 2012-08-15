********************************************************************************************************************************
********************************************************************************************************************************
******************************************* MULTIPLE FACE TRACKER **************************************************************
********************************************************************************************************************************
********************************************************************************************************************************

Author	:	Ritesh Ranjan
Website	:	www.bytesandlogics.wordpress.com
Date	:	26 July 2012

####Instruction####

OpenCV 2.* needs to be installed on the system.
Distribuions Supported: Linux

Note:	Please Change the path for the cascade on line 51 to where you have installed it.
	Mine was a default installation.This path should work on most systems.
	Code edited and the xml file is provided with the package.This will not be a problem anymore I 	guess.

#### Compile ####

g++ `pkg-config --cflags opencv` multi_face_tracking.cpp -o multi_face `pkg-config --libs opencv`


#### Run ####

./multi_face

The initial Red box is the output of Haar Cascade.
Press Spacebar to Run the Median Flow Tracker part of the code.
The passed points are shown in White.
The center of the Box is the output of the tracker.
To run Haar again 	: PRESS SPACEBAR
to Exit at any point	: PRESS ESC
