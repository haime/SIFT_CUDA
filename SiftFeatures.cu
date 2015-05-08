#include <SiftFeatures.h>




__global__ void Convolution(uchar* image,float* mask, ArrayImage* PyDoG, int maskR,int maskC, int imgR,int imgC, uchar* imgOut, int idxPyDoG){
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	
		
	int iImg=0;
	int aux=0;
	int pxlThrd = ceil((double)(imgC*imgR)/(gDim*bDim)); ////////numero de veces que caben
														 ////////los hilos en la imagen.
	for(int i = 0; i <pxlThrd; ++i)///////////////////////////// Strike 
	
	{
		//////////////////////////////////////
		//////////////////////////////////////Calculo de indices
		iImg=(tid+(bDim*bid)) + (i*gDim*bDim); //// pixel en el que trabajara el hilo
		//////////////////////////////////////
		//////////////////////////////////////

		if(iImg < imgC*imgR){
			int condition=maskC/2+imgC*(floor((double)maskC/2));
			if (iImg-condition < 0  ||										///condicion arriba
				iImg+condition > imgC*imgR ||								///condicion abajo
				iImg%imgC < maskC/2 ||										///condicion izquierda
				iImg%imgC > (imgC-1)-(maskC/2) )							///condicion derecha
			{
				aux=0;
			}else{		
				int itMask = 0;
				int itImg=iImg-condition;
				for (int j = 0; j < maskR; ++j)
				{		
					for (int h = 0; h < maskC; ++h)
					{
						aux+=image[itImg]*mask[itMask];
						++itMask;
						++itImg;
					}
					itImg+=imgC-maskC;
				}
			}
			aux=(aux<0)?0:aux;
			imgOut[iImg]=(aux>255)?255:aux;
			PyDoG[idxPyDoG].image=imgOut;
			PyDoG[idxPyDoG].cols=imgC;
			PyDoG[idxPyDoG].rows=imgR;
			aux=0;
		}
	}
}


void MaskGenerator(double sigma, int size,Mat mask){//Generate Gaussian Kernel
	double mean = size/2;
	for (int x = 0; x < size; ++x) 
		for (int y = 0; y < size; ++y){
				mask.at<float>(x,y)=exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
									   / (2 * M_PI * sigma * sigma);
		}
}

int ResizeImage(Mat image,vector<Mat>& images, int octvs){
	images.push_back(image);
	for(int i=0; i<octvs-1; ++i)
	{
		Mat aux = images[i];
		resize(aux,aux,Size(images[i].cols/2,images[i].rows/2));
		images.push_back(aux);
	}
	return 0;
}

int PyramidKDoG(vector<Mat> & PyKDoG, int octvs, int intvls){
	vector<double> sig;
	double sigma =1;
	double s= sqrt(2)/3;
	
	vector<Mat> PyGauss;
	int size = 7;//size of gaussian mask
	Mat mask=Mat::ones(size,size,CV_32F);
	MaskGenerator(s,size,mask);

	/*
	
	//////////////////////////////////////////////////////////////////////Calculo de Sigmas
	double k = pow( 2.0, 1.0 / 6); 
	sig.push_back(sigma);
	sigma=sigma * sqrt( k*k -1);
	sig.push_back(sigma);
	for (int i = 2; i < intvls + 3; i++){
		sigma=sigma*k;
		sig.push_back(sigma);
	}
	////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////Piramide de Gausianas
	for(int i=0; i<intvls+3; ++i){	
		Mat aux=Mat::ones(size,size,CV_32F);
		if(i==0){
			PyGauss.push_back(mask);
			Mat aux1=Mat::zeros(896,896,CV_32F);
		   	resize(mask,aux1,aux1.size());
		   	imshow("e",aux1);
    		waitKey(0);
    		destroyAllWindows();
		}
		else{
		   	//GaussianBlur(PyGauss[i-1],aux,Size(0,0),sig[i]);
		   	cout<<sig[i]<<endl;
		   	MaskGenerator(s-sig[i],size,aux);

			PyGauss.push_back(aux);

			Mat aux1=Mat::zeros(896,896,CV_32F);
		   	resize(aux,aux1,aux1.size());
		   	imshow("e",aux1);
    		waitKey(0);
    		destroyAllWindows();
		}
	}

	*/

	for(int i=0; i<intvls+3; ++i){	
		Mat aux=Mat::ones(size,size,CV_32F);
		/*if(i==0){
			PyGauss.push_back(mask);
			Mat aux1=Mat::zeros(896,896,CV_32F);
		   	resize(mask,aux1,aux1.size(),0,0,INTER_CUBIC);
		   	imshow("e",aux1);
    		waitKey(0);
    		destroyAllWindows();
		}
		else{*/
		   	//GaussianBlur(PyGauss[i-1],aux,Size(0,0),sig[i]);
		   	//cout<<sig[i]<<endl;
		   	MaskGenerator(s*0.11,size,aux);
		   	cout<<s*0.11<<endl;
		   	s*=0.11;
			PyGauss.push_back(aux);

			Mat aux1=Mat::zeros(896,896,CV_32F);
		   	resize(aux,aux1,aux1.size(),0,0,INTER_CUBIC);
		   	imshow("e",aux1);
    		waitKey(0);
    		destroyAllWindows();
		//}
	}



	///////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Resta de Gausianas
	for(int i=0; i<intvls+2; ++i){
	//	for(int i=0; i<intvls+3; ++i){
		Mat aux=Mat::ones(size,size,CV_32F);
		Mat aux1=Mat::zeros(size,size,CV_32F);
		subtract(PyGauss[i+1],PyGauss[i],aux);
		//subtract(PyGauss[i],aux1,aux);
		

		PyKDoG.push_back(aux);
	}
	///////////////////////////////////////////////////////////////////////////////////////
	return 0;
}

/*void VectorToPointer(vector<Mat> img, ArrayImage * pImg){
	for(int i=0; i<img.size(); ++i){
		pImg[i].cols=img[i].cols;
		pImg[i].rows=img[i].rows;
		pImg[i].image=img[i].ptr<uchar>();
	}
}*/

int SiftFeatures(Mat Image, vector<Mat> PyDoG){
	const int intvls = 3;
	int octvs;
	//cudaError_t e;
	octvs = log( min( Image.rows, Image.cols ) ) / log(2) - 2;
	vector<Mat> PyKDoG;
	vector<Mat> images;
	PyramidKDoG( PyKDoG,octvs,intvls);
	ResizeImage(Image,images,octvs);
	int idxPyDoG=0;
	
	
	ArrayImage * pyDoG_Out;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
	cudaMalloc(&pyDoG_Out,sizeof(ArrayImage)*images.size()*sizeof(ArrayImage) *PyKDoG.size());
	//cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;

	for (int i = 0; i < images.size() ; ++i)
	{
		
		uchar * img_D;
		int sizeImage = images[i].rows*images[i].cols;
		
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
		
		cudaMalloc(&img_D,sizeof(uchar)*sizeImage);///imagenes
		//cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;
	
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Copio Memoria GPU

		cudaMemcpy(img_D,images[i].ptr<uchar>(),sizeof(uchar)*sizeImage,cudaMemcpyHostToDevice);
		//cout<<cudaGetErrorString(e)<<" cudaMemCopyHD"<<endl;

		int imgBlocks= ceil((double) images[i].cols/BW);
		for (int m = 0; m < PyKDoG.size(); ++m)
		{
			float * pkDoG_D;
			uchar * out_D;
			uchar * out= new uchar[sizeImage];
			int sizeMask=PyKDoG[m].rows*PyKDoG[m].cols;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
			cudaMalloc(&pkDoG_D,sizeof(float)*sizeMask);//mascaras
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask "<<endl;
			cudaMalloc(&out_D,sizeof(uchar)*sizeImage);
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask"<<endl;
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Copio Memoria GPU

			cudaMemcpy(pkDoG_D,PyKDoG[m].ptr<float>(),sizeof(float)*sizeMask,cudaMemcpyHostToDevice);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyHD________Mask"<<endl;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Lanzo Kernel
			
			Convolution<<<imgBlocks,1024>>>(img_D,pkDoG_D,pyDoG_Out,PyKDoG[m].rows,PyKDoG[m].cols,images[i].rows,images[i].cols,out_D,idxPyDoG);
			cudaDeviceSynchronize();
			++idxPyDoG;
			cudaFree(pkDoG_D); 
			

			cudaMemcpy(out,out_D,sizeof(uchar)*sizeImage,cudaMemcpyDeviceToHost);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			Mat image_out(images[i].rows,images[i].cols,CV_8U,out);
			
			imshow("e",image_out*100);
    		waitKey(0);
    		destroyAllWindows();

			delete(out);
			cudaFree(out_D);
		}
		cudaFree(img_D);
	}
	cudaFree(pyDoG_Out);


	return 0;
}
