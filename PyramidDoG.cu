#include <PyramidDoG.h>




__global__ void Blur(uchar* image,float* mask, ArrayImage* PyDoG, int maskR,int maskC, int imgR,int imgC, uchar* imgOut, int idxPyDoG){
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	
		
	int iImg=0;
	uchar aux=0;
	int pxlThrd = ceil((double)(imgC*imgR)/(gDim*bDim)); ////////numero de veces que caben
														 ////////los hilos en la imagen.

	

	//for(int i = 0; i <pxlThrd; ++i)///////////////////////////// Strike 
	for(int i = 0; i <675; ++i)
	{
		//////////////////////////////////////
		//////////////////////////////////////Calculo de indices
		iImg=(tid+(bDim*bid)) + (i*gDim*bDim); //// pixel en el que trabajara el hilo
		printf("%i \n",iImg%imgC);
		//////////////////////////////////////
		//////////////////////////////////////
		if(iImg < imgC*imgR){
			int condition=maskC/2+imgC*(floor((double)maskC/2));
			if (iImg-condition < 0  ||												///condicion arriba
				iImg+condition > imgC*imgR /*||	///condicion abajo
				iImg%imgC < maskC/2 ||				///condicion izquierda
				//iImg%imgC >=maskC/2 */)					///condicion derecha
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
						//printf("%i --- %i %i %i %i %i\n",iImg,i,tid,bDim,bid,gDim);		
						++itMask;
						++itImg;

					}
					itImg+=imgC-maskC;
				}
			}

			


			imgOut[iImg]=aux;

			//PyDoG[idxPyDoG].image=imgOut;
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
	double sigma =1.6;
	vector<Mat> PyGauss;
	int size = 7;//size of gaussian mask
	Mat mask=Mat::ones(size,size,CV_32F);
	MaskGenerator(1.519868415,size,mask);
	//////////////////////////////////////////////////////////////////////Calculo de Sigmas
	double k = pow( 2.0, 1.0 / intvls ); 
	sig.push_back(sigma);
	sigma=sigma * sqrt( k*k- 1 );
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
		}
		else{
		   	GaussianBlur(PyGauss[i-1],aux,Size(0,0),sig[i]);
			PyGauss.push_back(aux);
		}
	}
	///////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Resta de Gausianas
	for(int i=0; i<intvls+2; ++i){
		Mat aux=Mat::ones(size,size,CV_32F);
		subtract(PyGauss[i+1],PyGauss[i],aux);
		PyKDoG.push_back(aux);
	}
	///////////////////////////////////////////////////////////////////////////////////////
	return 0;
}

void VectorToPointer(vector<Mat> img, ArrayImage * pImg){
	for(int i=0; i<img.size(); ++i){
		pImg[i].cols=img[i].cols;
		pImg[i].rows=img[i].rows;
		pImg[i].image=img[i].ptr<float>();
	}
}

int PyramidDoG(Mat Image, vector<Mat> PyDoG){
	const int intvls = 3;
	int octvs;
	cudaError_t e;
	octvs = log( min( Image.rows, Image.cols ) ) / log(2) - 2;
	vector<Mat> PyKDoG;
	vector<Mat> images;
	PyramidKDoG( PyKDoG,octvs,intvls);
	ResizeImage(Image,images,octvs);
	int idxPyDoG=0;
	
		
	ArrayImage * pyDoG_Out;
	
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
	e=cudaMalloc(&pyDoG_Out,sizeof(ArrayImage)*images.size()*sizeof(ArrayImage) *PyKDoG.size());
	cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;


	//for (int i = 0; i < images.size() ; ++i)
	for (int i = 0; i < 1 ; ++i)
	{
		
		uchar * img_D;
		int sizeImage = images[i].rows*images[i].cols;
		
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
		
		e=cudaMalloc(&img_D,sizeof(uchar)*sizeImage);///imagenes
		cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;
	
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Copio Memoria GPU

		e=cudaMemcpy(img_D,images[i].ptr<uchar>(),sizeof(uchar)*sizeImage,cudaMemcpyHostToDevice);
		cout<<cudaGetErrorString(e)<<" cudaMemCopyHD"<<endl;

		
		//for (int m = 0; m < PyKDoG.size(); ++i)
		for (int m = 0; m < 1; ++m)
		{
			float * pkDoG_D;
			uchar * out_D;
			uchar * out= new uchar[sizeImage];
			int sizeMask=PyKDoG[m].rows*PyKDoG[m].cols;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
			e=cudaMalloc(&pkDoG_D,sizeof(float)*sizeMask);//mascaras
			cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask "<<endl;
			e=cudaMalloc(&out_D,sizeof(uchar)*sizeImage);
			cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask"<<endl;
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Copio Memoria GPU

			e=cudaMemcpy(pkDoG_D,PyKDoG[m].ptr<float>(),sizeof(float)*sizeMask,cudaMemcpyHostToDevice);
			cout<<cudaGetErrorString(e)<<" cudaMemCopyHD________Mask"<<endl;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Lanzo Kernel
			
			Blur<<<8,32>>>(img_D,pkDoG_D,pyDoG_Out,PyKDoG[m].rows,PyKDoG[m].cols,images[i].rows,images[i].cols,out_D,idxPyDoG);
			cudaDeviceSynchronize();
			++idxPyDoG;
			cudaFree(pkDoG_D); 
			

			e=cudaMemcpy(out,out_D,sizeof(uchar)*sizeImage,cudaMemcpyDeviceToHost);
			cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			Mat image_out(images[i].rows,images[i].cols,CV_8U,out);
			//cout<<image_out<<endl;
			imshow("e",image_out);
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
