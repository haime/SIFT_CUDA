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
#define BH 32


struct ArrayImage
{
	int cols;
	int rows;
	float* image;
};

int PyramidKDoG(vector<Mat> &PyKDoG, int octvs, int intvls);
int PyramidDoG(Mat Image, vector<Mat> PyDoG);
void MaskGenerator(double sigma, int size,Mat mask);
int ResizeImage(Mat image,vector<Mat>& images, int octvs);
void VectorToPointer(vector<Mat> img, ArrayImage * pImg);
