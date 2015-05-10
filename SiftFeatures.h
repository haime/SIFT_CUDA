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
	int cols;
	int rows;
	float* image;
};

struct MinMax
{
	int col;
	int row;
	int levelPy;
	int idxArray;
};




int PyramidKDoG(vector<Mat> &PyKDoG, int octvs, int intvls);
int SiftFeatures(Mat Image, vector<Mat> PyDoG);
void MaskGenerator(double sigma, int size,Mat mask);
int ResizeImage(Mat image,vector<Mat>& images, int octvs);
void VectorToPointer(vector<Mat> img, ArrayImage * pImg);
