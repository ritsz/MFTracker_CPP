/*
 *
 *
 * multi_face_tracking.cpp
 * 
 * Copyright 2012 RITESH <ritesh@ritsz>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 * 
 * Program	:	multi_face_tracking.cpp
 * Author	:	Ritesh Ranjan
 * Website	:	www.bytesandlogics.wordpress.com
 * Abstract	:	Code is written using the OpenCV libraries.Use Haar to
 * 			detect a bounding box around the face.Then use the median
 * 			flow tracker to track that box. Can be used for maximun 
 * 			two users at a time. 
 */


// HEADER//
#include <cv.h>
#include <highgui.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <time.h>

//Namespace

using namespace cv;
using namespace std;

//Definitions

static CvMemStorage* storage = 0;		// Dynamically growing memory for calculations

static CvHaarClassifierCascade* cascade = 0;	// Load a trained cascade

const char* cascade_name = "./haarcascade_frontalface_alt.xml";

CvSeq* faces;
IplImage* frame = 0;
IplImage *frame1_1C = NULL,*frame2_1C = NULL,*pyramid1 = NULL, *pyramid2 = NULL;
IplImage* displayed_image = 0;

double bbox1_x,bbox1_y,bbox2_y,bbox2_x,bbox_x,bbox_y;
double bbox1_width,bbox1_height,bbox2_width,bbox2_height,bbox_width,bbox_height;
double xmean = 0;
double ymean = 0;
float median;
bool fast = false;

/* -------------------------CODE-----------------------------*/

double euclid_dist(const CvPoint2D32f* point1, const CvPoint2D32f* point2) 
{
	
	/*
	 * This function calculates the euclidean distance between 2 points
	*/
	
	double distance, xvec, yvec;
	xvec = point2->x - point1->x;
	yvec = point2->y - point1->y;
	distance = sqrt((xvec * xvec) + (yvec * yvec));
	return distance;
}



void pairwise_dist(const CvPoint2D32f* features, double *edist, int npoin) 
{
	/*
	 * calculate m x n euclidean pairwise distance matrix.
	*/
	for (int i = 0; i < npoin; i++) {
		for (int j = 0; j < npoin; j++) {
			int ind = npoin*i + j;
			edist[ind] = euclid_dist(&features[i],&features[j]);
		}
	}
}


void ncc_filter(IplImage *frame1, IplImage *frame2, CvPoint2D32f *prev_feat, CvPoint2D32f *curr_feat, 
				int npoin, int method, IplImage *rec0, IplImage *rec1, IplImage *res, int *ncc_pass) 
{
	/*
	 * Normalized Cross Correlation Filter 	 
	*/
	
	int filt = npoin/2;
	vector<float> ncc_err (npoin,0.0);

	for (int i = 0; i < npoin; i++) {
		cvGetRectSubPix( frame1, rec0, prev_feat[i] );
		cvGetRectSubPix( frame2, rec1, curr_feat[i] );
		cvMatchTemplate( rec0,rec1, res, method );
		ncc_err[i] = ((float *)(res->imageData))[0]; 
	}
	vector<float> err_copy (ncc_err);
	sort(ncc_err.begin(), ncc_err.end());
	median = (ncc_err[filt]+ncc_err[filt-1])/2.;
	for(int i = 0; i < npoin; i++) {
		if (err_copy[i] > median) {
			ncc_pass[i] = 1;		
		}
		else {
			ncc_pass[i] = 0;
		}
	}	
}


void fb_filter(const CvPoint2D32f* prev_features, const CvPoint2D32f* backward_features, 
				const CvPoint2D32f* curr_feat, int *fb_pass, const int npoin) {
	/*
	 * This function implements forward-backward error filtering
	*/
	vector<double> euclidean_dist (npoin,0.0);
	
	int filt = npoin/2;
	for(int i = 0; i < npoin; i++) {
		euclidean_dist[i] = euclid_dist(&prev_features[i], &backward_features[i]);
	}
	
	vector<double> err_copy (euclidean_dist);
	
	//use the STL sort algorithm to filter results
	
	sort(euclidean_dist.begin(), euclidean_dist.end());
	double median = (euclidean_dist[filt]+euclidean_dist[filt-1])/2.;
	for(int i = 0; i < npoin; i++) {
		if (err_copy[i] < median) {
			fb_pass[i] = 1;		
		}
		else {
			fb_pass[i] = 0;
		}
	}
}


void bbox_move(const CvPoint2D32f* prev_feat, const CvPoint2D32f* curr_feat, const int npoin,
				double &xmean, double &ymean)
 {
	/*
	 * Calculate bounding box motion. 
	 */
	vector<double> xvec (npoin,0.0);
	vector<double> yvec (npoin,0.0);
	for (int i = 0; i < npoin; i++) {
		xvec[i] = curr_feat[i].x - prev_feat[i].x;
		yvec[i] = curr_feat[i].y - prev_feat[i].y;
	}	
	
	sort(xvec.begin(), xvec.end());
	sort(yvec.begin(), yvec.end());
	
	xmean = xvec[npoin/2];
	ymean = yvec[npoin/2];		//The final mostion is that of the mean of all the points. 
}


/*----------------------- MAIN FUNCTION ---------------------------*/
int main( int argc,char** argv)
{
	CvCapture* capture;
	
	if( argc != 2)
	{
		capture = cvCreateCameraCapture(0); 		//Grab frames from camera
	}
	
	else
	{
		capture = cvCaptureFromFile( argv[1] );		//Grab frames from video
	}
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name,0,0,0);
	
	
	detect:if(!cascade){ fprintf(stderr,"ERROR: Cascade not loaded!! \nCheck the path given for it on line 24 of code \n"); return -1; }  // Check whether cascades have been loaded
	
	if(!capture){ fprintf(stderr,"ERROR: Camera not loaded \n"); exit;}	// Check whether camera has been loaded
	
	storage = cvCreateMemStorage(0);
	
	
	CvPoint pta,ptb;
	pta.x = 0;
	pta.y = 0;
	int i;
	cvClearMemStorage(storage);	
	cvNamedWindow("Results",CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Results",700,0);
	
	// Loop to get the Bounding Box from haar cascades
	while(1)
	{
		frame = cvQueryFrame(capture);
		
		if(!frame) { fprintf(stderr,"Camera doesn't return frames \n"); return -1;}
		
	    displayed_image = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,3); // The Image that is displayed finally
	    cvCopy(frame,displayed_image,NULL);
					
		faces = cvHaarDetectObjects(frame,cascade,storage,1.1,2,CV_HAAR_DO_CANNY_PRUNING,cvSize(40,40));
		for(i = 0;i<(faces? faces->total:0); i++)
		{
			CvRect* r = (CvRect*)cvGetSeqElem(faces,i);
			pta.x = r->x;
			ptb.x = r->x + r->width;
			pta.y = r->y;
			ptb.y = r->y + r->height;
			
			cvRectangle(displayed_image,pta,ptb,CV_RGB(255,0,0),3,8,0);
			 
		}
		
		if(pta.x!=0 && pta.y!=0) break;
		if(fast) break;
		
		cvShowImage("Results",displayed_image);
				
		char esc = cvWaitKey(33);
		if(esc == 27) return -1;
		if(esc == 32) break;
		
	}
	
	/*------------- If one face is found ----------------------*/
	
	if(faces->total == 1)  
	{	
		CvRect* r = (CvRect*)cvGetSeqElem(faces,0);
		bbox_x = r->x;
		bbox_y = r->y;
		bbox_width = r->width;
		bbox_height = r->height;
		
		IplImage *frame1_1C = NULL, *frame2_1C = NULL, *pyramid1 = NULL, *pyramid2 = NULL;
	
	frame1_1C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
	frame2_1C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1); // Grayscale images
	
	int npoints = 400;				// Number of feature points
	int pcount;						// Number of passed points that finally track
	int winsize = 30;				
	IplImage *rec0 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *rec1 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *res  = cvCreateImage( cvSize( 1, 1 ), IPL_DEPTH_32F, 1 );	//  for the NCC function
	
	/* This array will contain the locations of the points from frame 1 in frame 2. */
	vector<CvPoint2D32f> frame1_features(npoints);
	vector<CvPoint2D32f> frame2_features(npoints);
	vector<CvPoint2D32f> FB_features(npoints);
	
	/* The i-th element of this array will be non-zero if and only if the i-th feature of
	 * frame 1 was found in frame 2.
	 */
	 
	char optical_flow_found_feature[npoints];    		//features in first frame
	char optical_flow_found_feature2[npoints];			//features in second frame
	float optical_flow_feature_error[npoints];			//error in Optical Flow 
	vector<int> fb_pass(npoints);						//Points that passed fb
	vector<int> ncc_pass(npoints);						//Points that passed ncc
		
	printf("\n\tRectangle made at (%g,%g) and size (%g,%g)\n\tWe have got a Bounding Box\n\n",bbox_x,bbox_y,bbox_width,bbox_height);
	
	//Main Loop for tracking
	while(true)
	{	
		cvConvertImage(frame,frame1_1C,0);
		
		//Making new points for PyrLK to track
		for(i = 0;i<20;i++)
		{
			for(int j = 0;j<20;j++)
			{
				int l = i*20 + j;
				
				frame1_features[l].x = bbox_x + (bbox_width/20)*j + (bbox_width/40);
				frame1_features[l].y = bbox_y + (bbox_height/20)*i + (bbox_height/40);
			}
		}
		//New feature points made
		
		//Second Frame Capture, converted to grey scale and displayed 
		frame = cvQueryFrame(capture);
		cvConvertImage(frame,frame2_1C,0);
		cvCopy(frame,displayed_image,NULL);
		
		// Pyr Lucas kanade Optical Flow
		
		CvTermCriteria term = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.3);
		
		pyramid1 = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
		pyramid2 = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
		
		cvCalcOpticalFlowPyrLK(frame1_1C,frame2_1C,pyramid1,pyramid2,&frame1_features[0],&frame2_features[0],npoints,cvSize(3,3),5,optical_flow_found_feature,optical_flow_feature_error,term,0);
		cvCalcOpticalFlowPyrLK(frame2_1C,frame1_1C,pyramid1,pyramid2,&frame2_features[0],&FB_features[0]    ,npoints,cvSize(3,3),5,optical_flow_found_feature2,optical_flow_feature_error,term,0);
		
		double xmean = 0;
		double ymean = 0;
		
		//filter the cascade
		fb_filter(&frame1_features[0], &FB_features[0], &frame2_features[0], &fb_pass[0], npoints);
		ncc_filter(frame1_1C,frame2_1C,&frame1_features[0],&frame2_features[0],
				   npoints,CV_TM_CCOEFF_NORMED, rec0, rec1, res, &ncc_pass[0]);
		
		int pcount_prev = 1;
		pcount = 0;
		
		for(int i = 0; i<npoints;i++)
		{
			if(fb_pass[i] && ncc_pass[i] && (frame2_features[i].x>bbox_x) && (frame2_features[i].y>bbox_y) && (frame2_features[i].x < bbox_x + bbox_width ) && (frame2_features[i].y < bbox_y +bbox_height) )
			{
				pcount++;
			}
		}
		
		fprintf(stderr,"Passed Points %d ### %f\t",pcount , median);
		if((median)<0.37 ) {goto detect; }		
		if(pcount == 0) {  fprintf(stderr," No point tracked currently: DETECTING AGAIN"); pta.x = 0; pta.y = 0;goto detect;  } // If no points detected, run haar again
		
		vector<CvPoint2D32f> curr_features2(pcount),prev_features2(pcount);
		int j = 0;
		
		for( int i = 0; i< npoints; i++)
		{
			if(fb_pass[i] && ncc_pass[i] && (frame2_features[i].x>bbox_x) && (frame2_features[i].y>bbox_y) && (frame2_features[i].x < bbox_x + bbox_width ) && (frame2_features[i].y < bbox_y +bbox_height) )
			{
				curr_features2[j] = frame2_features[i];
				prev_features2[j] = frame1_features[i];
				j++;
			}
		}
		
		int n2 = pcount*pcount;
		
		vector<double> pdist_prev(n2),pdist_curr(n2),pdiv(n2);
		
		pairwise_dist(&prev_features2[0],&pdist_prev[0],pcount); // Find distance btw all points
		pairwise_dist(&curr_features2[0],&pdist_curr[0],pcount);
		
		//Divide corresponding distances to find the amount of scaling 
		
		for (int i = 0; i < n2; i++) {
			if (pdist_prev[i] > 0.0) {
				pdiv[i] = pdist_curr[i]/pdist_prev[i];
			}
		}
		sort(pdiv.begin(),pdiv.end());
			
		double box_scale;
		box_scale = pdiv[n2/2]; // Scaling set to the median of all values
		
		/*
		 * Bounding Box is moved using the points that were able to pass FB and NCC 
		 */		  
		bbox_move(&prev_features2[0],&curr_features2[0],pcount,xmean,ymean);
		bbox_x = bbox_x + (xmean) - bbox_width*(box_scale - 1.)/2.;
		bbox_y = bbox_y + (ymean) - bbox_height*(box_scale - 1.)/2.;
		bbox_width = bbox_width * (box_scale);
		bbox_height = bbox_height * (box_scale);
		double track_x = bbox_x + bbox_width/2;
		double track_y = bbox_y + bbox_height/2;
		
		if( (track_x >600)||(track_y > 450)|| (track_x <20) || (track_y<20) ) { pta.x = 0; pta.y = 0; goto detect;}
		
		cvRectangle(displayed_image, cvPoint(bbox_x, bbox_y), 
					cvPoint(bbox_x+bbox_width,bbox_y+bbox_height),
					cvScalar(0xff,0x00,0x00) );
					
		for (int i = 0; i < pcount; i++) {
			cvCircle(displayed_image, cvPoint(curr_features2[i].x, curr_features2[i].y), 1, cvScalar(255,255,255));
		}
		
		cvCircle(displayed_image,cvPoint(track_x,track_y) , 3,cvScalar(0,0,255),4,8);
		
		printf(" TRACKING POINT ( %d , %d )\n",int(track_x),int(track_y));
		
		char online_dist[] = "MFTRACKING";
		CvFont bfont;
		double hscale = 0.5;
		double vscale = 0.5;
		cvInitFont(&bfont, CV_FONT_HERSHEY_SIMPLEX, hscale, vscale,0,1);
		cvPutText(displayed_image, online_dist, cvPoint(bbox_x+bbox_width + 20,bbox_y), &bfont, cvScalar(0,0,255));
		
		cvShowImage("Results",displayed_image);
		char esc = cvWaitKey(33);
		if(esc == 27) break;					// Press Esc to end the program
		if(esc == 32) {pta.x = 0; pta.y = 0;goto detect;}				// Press spacebar for running Haar again

	} 			// End of Main While Loop

/*
 * Cleaning up the mess we created
 */ 	
cvReleaseImage(&frame);
cvDestroyWindow("Results");	
cvReleaseImage(&frame1_1C);
cvReleaseImage(&frame2_1C);
cvReleaseImage(&pyramid1);
cvReleaseImage(&pyramid2);
cvReleaseCapture(&capture);
		
	}
	
/*---------------------- If two faces are found ------------------*/	
	
	if( (faces->total) >= 2)
	{
		CvRect* r = (CvRect*)cvGetSeqElem(faces,0);
		bbox1_x = r->x;
		bbox1_y = r->y;
		bbox1_width = r->width;
		bbox1_height = r->height;
		r = (CvRect*)cvGetSeqElem(faces,1);
		bbox2_x = r->x;
		bbox2_y = r->y;
		bbox2_width = r->width;
		bbox2_height = r->height;
		
		printf("\n\tRectangle made at (%g,%g) and size (%g,%g)\n\tWe have got a Bounding Box\n\n",bbox1_x,bbox1_y,bbox1_width,bbox1_height);
		printf("\n\tRectangle made at (%g,%g) and size (%g,%g)\n\tWe have got a Bounding Box\n\n",bbox2_x,bbox2_y,bbox2_width,bbox2_height);
	
	static IplImage *frame1_1C = NULL, *frame2_1C = NULL, *pyramid1 = NULL, *pyramid2 = NULL,*frame1_2C = NULL , *frame2_2C = NULL;
	
	frame1_1C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
	frame2_1C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1); // Grayscale images
	
	int npoints = 100;				// Number of feature points
	int pcount;						// Number of passed points that finally track
	int winsize = 30;
	
	vector<CvPoint2D32f> frame1_features1(npoints);
	vector<CvPoint2D32f> frame2_features1(npoints);
	vector<CvPoint2D32f> FB_features1(npoints);
	
	vector<CvPoint2D32f> frame1_features2(npoints);
	vector<CvPoint2D32f> frame2_features2(npoints);
	vector<CvPoint2D32f> FB_features2(npoints);
	
	char optical_flow1_found_feature[npoints];    		//features in first frame
	char optical_flow1_found_feature2[npoints];			//features in second frame
	float optical_flow1_feature_error[npoints];			//error in Optical Flow 
	vector<int> fb_pass1(npoints);
	
	char optical_flow2_found_feature[npoints];    		//features in first frame
	char optical_flow2_found_feature2[npoints];			//features in second frame
	float optical_flow2_feature_error[npoints];			//error in Optical Flow 
	vector<int> fb_pass2(npoints);
	
	
	IplImage *rec0 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *rec1 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *res1  = cvCreateImage( cvSize( 1, 1 ), IPL_DEPTH_32F, 1 );	//  for the NCC function
	
	IplImage *rec2 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *rec3 = cvCreateImage( cvSize(winsize, winsize), 8, 1 );
	IplImage *res2  = cvCreateImage( cvSize( 1, 1 ), IPL_DEPTH_32F, 1 );	//  for the NCC function
	
	vector<int> ncc_pass1(npoints);	
	vector<int> ncc_pass2(npoints);	
	
	//Main While loop
	while(1)
	{
		cvConvertImage(frame,frame1_1C,0);
		frame1_2C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
		frame2_2C = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1); // Grayscale images
		cvConvertImage(frame,frame1_2C,0);
		
		//Making new points for PyrLK to track
		for(i = 0;i<10;i++)
		{
			for(int j = 0;j<10;j++)
			{
				int l = i*10 + j;
				
				frame1_features1[l].x = bbox1_x + (bbox1_width/10)*j + (bbox1_width/20);
				frame1_features1[l].y = bbox1_y + (bbox1_height/10)*i + (bbox1_height/20);
			}
		}
		//New feature points made
		
		frame = cvQueryFrame(capture);
		cvConvertImage(frame,frame2_1C,0);
		cvConvertImage(frame,frame2_2C,0);
		cvCopy(frame,displayed_image,NULL);
	
		// Pyr Lucas kanade Optical Flow
		
		CvTermCriteria term = cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.3);
		
		pyramid1 = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
		pyramid2 = cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
		
		cvCalcOpticalFlowPyrLK(frame1_1C,frame2_1C,pyramid1,pyramid2,&frame1_features1[0],&frame2_features1[0],npoints,cvSize(3,3),5,optical_flow1_found_feature,optical_flow1_feature_error,term,0);
		cvCalcOpticalFlowPyrLK(frame2_1C,frame1_1C,pyramid1,pyramid2,&frame2_features1[0],&FB_features1[0] ,npoints,cvSize(3,3),5,optical_flow1_found_feature2,optical_flow1_feature_error,term,0);
		
		double xmean = 0;
		double ymean = 0;
		
		//filter the cascade
		fb_filter(&frame1_features1[0], &FB_features1[0], &frame2_features1[0], &fb_pass1[0], npoints);
		ncc_filter(frame1_1C,frame2_1C,&frame1_features1[0],&frame2_features1[0],
				   npoints,CV_TM_CCOEFF_NORMED, rec0, rec1, res1, &ncc_pass1[0]);
		
		pcount = 0;
		
		for(int i = 0; i<npoints;i++)
		{
			if(fb_pass1[i] && ncc_pass1[i]){
				pcount++;
			}
		}
		
		fprintf(stderr,"Passed Points %d\t",pcount);
		if(median < 0.37 ) {fast = true; goto detect; }		

		if(pcount == 0) {  fprintf(stderr," No point tracked currently: DETECTING AGAIN");pta.x = 0; pta.y = 0; goto detect; } // If no points detected, run haar again
		
		vector<CvPoint2D32f> curr_features1(pcount),prev_features1(pcount);
		int j = 0;
		
		for( int i = 0; i< npoints; i++)
		{
			if(fb_pass1[i] && ncc_pass1[i] )
			{
				curr_features1[j] = frame2_features1[i];
				prev_features1[j] = frame1_features1[i];
				j++;
			}
		}
		
		int n2 = pcount*pcount;
		vector<double> pdist_prev1(n2),pdist_curr1(n2),pdiv1(n2);
		
		pairwise_dist(&prev_features1[0],&pdist_prev1[0],pcount); // Find distance btw all points
		pairwise_dist(&curr_features1[0],&pdist_curr1[0],pcount);
		
		//Divide corresponding distances to find the amount of scaling 
		
		for (int i = 0; i < n2; i++) {
			if (pdist_prev1[i] > 0.0) {
				pdiv1[i] = pdist_curr1[i]/pdist_prev1[i];
			}
		}
		sort(pdiv1.begin(),pdiv1.end());
			
		double box_scale;
		box_scale = pdiv1[n2/2]; // Scaling set to the median of all values
		
		/*
		 * Bounding Box is moved using the points that were able to pass FB and NCC 
		 */		  
		bbox_move(&prev_features1[0],&curr_features1[0],pcount,xmean,ymean);
		bbox1_x = bbox1_x + (xmean) - bbox1_width*(box_scale - 1.)/2.;
		bbox1_y = bbox1_y + (ymean) - bbox1_height*(box_scale - 1.)/2.;
		bbox1_width = bbox1_width * (box_scale);
		bbox1_height = bbox1_height * (box_scale);
		double track1_x = bbox1_x + bbox1_width/2;
		double track1_y = bbox1_y + bbox1_height/2;
		
		cvRectangle(displayed_image, cvPoint(bbox1_x, bbox1_y), 
					cvPoint(bbox1_x+bbox1_width,bbox1_y+bbox1_height),
					cvScalar(0xff,0x00,0x00) );
					
		for (int i = 0; i < pcount; i++) {
			cvCircle(displayed_image, cvPoint(curr_features1[i].x, curr_features1[i].y), 1, cvScalar(255,255,255));
		}
		
		cvCircle(displayed_image,cvPoint(track1_x,track1_y) , 3,cvScalar(255,0,0),4,8);
		
		printf(" TRACKING ( %d , %d )\t",int(track1_x),int(track1_y));
		
		char online_dist[] = "MFTRACKING";
		CvFont bfont;
		double hscale = 0.5;
		double vscale = 0.5;
		cvInitFont(&bfont, CV_FONT_HERSHEY_SIMPLEX, hscale, vscale,0,1);
		cvPutText(displayed_image, online_dist, cvPoint(bbox1_x+bbox1_width + 20,bbox1_y), &bfont, cvScalar(0,0,255));
		
		
		
		// SECOND DETECT TRACK
		
		for(int k = 0;k<10;k++)
		{
			for(int j = 0;j<10;j++)
			{
				int l = k*10 + j;
				
				frame1_features2[l].x = bbox2_x + (bbox2_width/10)*j + (bbox2_width/20);
				frame1_features2[l].y = bbox2_y + (bbox2_height/10)*k + (bbox2_height/20);
			}
		}
		
		cvCalcOpticalFlowPyrLK(frame1_2C,frame2_2C,pyramid1,pyramid2,&frame1_features2[0],&frame2_features2[0],npoints,cvSize(3,3),5,optical_flow2_found_feature,optical_flow2_feature_error,term,0);
		cvCalcOpticalFlowPyrLK(frame2_2C,frame1_2C,pyramid1,pyramid2,&frame2_features2[0],&FB_features2[0]    ,npoints,cvSize(3,3),5,optical_flow2_found_feature2,optical_flow2_feature_error,term,0);
	
		xmean = 0;
		ymean = 0;
		
		fb_filter(&frame1_features2[0], &FB_features2[0], &frame2_features2[0], &fb_pass2[0], npoints);
		ncc_filter(frame1_2C,frame2_2C,&frame1_features2[0],&frame2_features2[0],
				   npoints,CV_TM_CCOEFF_NORMED, rec2, rec3, res2, &ncc_pass2[0]);
		
		pcount = 0;
		
		for(int i = 0; i<npoints;i++)
		{
			if(fb_pass2[i] && ncc_pass2[i])
			{
				pcount++;
			}
		}
		
		if(pcount == 0) {  fprintf(stderr," No point tracked currently: DETECTING AGAIN");pta.x = 0; pta.y = 0; goto detect; } // If no points detected, run haar again

		if(median < 0.37 ) {fast = true; goto detect; }
		
		vector<CvPoint2D32f> curr_features2(pcount),prev_features2(pcount);
		int r = 0;
		
		for( int k = 0; k< npoints; k++)
		{
			if(fb_pass2[k] && ncc_pass2[k])
			{
				curr_features2[r] = frame2_features2[k];
				prev_features2[r] = frame1_features2[k];
				r++;
			}
		}
		
		int n1 = pcount*pcount;
		
		vector<double> pdist_prev2(n1),pdist_curr2(n1),pdiv2(n1);
		
		pairwise_dist(&prev_features2[0],&pdist_prev2[0],pcount); // Find distance btw all points
		pairwise_dist(&curr_features2[0],&pdist_curr2[0],pcount);
		
		//Divide corresponding distances to find the amount of scaling 
		
		for (int k = 0; k < n1; k++) {
			if (pdist_prev2[k] > 0.0) {
				pdiv2[k] = pdist_curr2[k]/pdist_prev2[k];
			}
		}
		sort(pdiv2.begin(),pdiv2.end());
			
		box_scale = pdiv2[n1/2]; // Scaling set to the median of all values
		
		bbox_move(&prev_features2[0],&curr_features2[0],pcount,xmean,ymean);
		bbox2_x = bbox2_x + (xmean) - bbox2_width*(box_scale - 1.)/2.;
		bbox2_y = bbox2_y + (ymean) - bbox2_height*(box_scale - 1.)/2.;
		bbox2_width = bbox2_width * (box_scale);
		bbox2_height = bbox2_height * (box_scale);
		double track2_x = bbox2_x + bbox2_width/2;
		double track2_y = bbox2_y + bbox2_height/2;
		
		cvRectangle(displayed_image, cvPoint(bbox2_x, bbox2_y), 
					cvPoint(bbox2_x+bbox2_width,bbox2_y+bbox2_height),
					cvScalar(0x00,0x00,0xff) );
					
		for (int k = 0; k < pcount; k++) {
			cvCircle(displayed_image, cvPoint(curr_features2[k].x, curr_features2[k].y), 1, cvScalar(255,255,255));
			}
			
		cvCircle(displayed_image,cvPoint(track2_x,track2_y) , 3,cvScalar(0,0,255),4,8);
		
		printf(" AND ( %d , %d )\n",int(track2_x),int(track2_y));
		
		char online_dist2[] = "MULTI_MFTRACKING";
		CvFont bfont2;
		double hscale2 = 0.5;
		double vscale2 = 0.5;
		cvInitFont(&bfont2, CV_FONT_HERSHEY_SIMPLEX, hscale2, vscale2,0,1);
		cvPutText(displayed_image, online_dist2, cvPoint(bbox2_x+bbox2_width + 20,bbox2_y), &bfont2, cvScalar(0,0,255));

		cvShowImage("Results",displayed_image);
		char esc = cvWaitKey(33);
		if(esc == 27) break;					// Press Esc to end the program
		if(esc == 32) {pta.x = 0; pta.y = 0;goto detect;}				// Press spacebar for running Haar again
		
		
	}//End of main While Loop
	
	cvReleaseImage(&frame);
cvDestroyWindow("Results");	
cvReleaseImage(&frame1_1C);
cvReleaseImage(&frame2_1C);
cvReleaseImage(&frame1_2C);
cvReleaseImage(&frame2_2C);
cvReleaseImage(&pyramid1);
cvReleaseImage(&pyramid2);
cvReleaseCapture(&capture);
		
	}
	
	
}
