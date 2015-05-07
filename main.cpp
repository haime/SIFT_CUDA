#include <iostream>
#include <opencv2/opencv.hpp>
#include <SiftFeatures.h>
#include <vector>

using namespace cv;
using namespace std;

string TEST= "0.jpg";


int main()
{
	Mat image= imread(TEST,CV_LOAD_IMAGE_GRAYSCALE);
	vector<Mat> PyDoG;

	SiftFeatures(image, PyDoG);
	imshow("test",image);
	waitKey(0);
	destroyAllWindows();

}