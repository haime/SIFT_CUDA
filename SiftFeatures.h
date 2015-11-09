#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <algorithm>  

#include <stdio.h>

using namespace cv;
using namespace std;


#define INTVLS 3
#define BW 32




struct ArrayImage
{
	float* image;
};

struct MinMax
{
	float * minMax;
};


struct keyPoint
{
	float orientacion;
	int x,y,octv; 

};



int PyramidKDoG(vector<Mat> &PyKDoG, int octvs, int intvls);
int SiftFeatures(Mat Image, vector<Mat> PyDoG,Mat I);
void MaskGenerator(double sigma, int size,Mat mask);
int ResizeImage(Mat image,vector<Mat>& images, int octvs);
void VectorToPointer(vector<Mat> img, ArrayImage * pImg);
int foundIndexesMaxMin(vector<Mat> minMax,vector<int*> & idxMinMax);
