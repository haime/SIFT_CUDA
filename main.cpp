#include <iostream>
#include <opencv2/opencv.hpp>
#include <SiftFeatures.h>
#include <vector>

using namespace cv;
using namespace std;



int main(int argc, char *argv[])
{
	if (argc==1)
	{
       cout<<"Debes ingresar parametros ./SIFT_Test [nombre imagen].[extención]"<<endl;
       return 1;
	}
	string TEST= argv[1];
	Mat image= imread(TEST,CV_LOAD_IMAGE_GRAYSCALE);
	if (image.empty())
	{
      cout<<"Debes ingresar nombre de una imagen existente ./SIFT_Test [nombre imagen].[extención]"<<endl;
       return 1;
	}

	double C8TO32=0.003921568627;
	image.convertTo(image,CV_32F,C8TO32);
	vector<Mat> PyDoG;
	SiftFeatures(image, PyDoG);
	imshow("test",image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}