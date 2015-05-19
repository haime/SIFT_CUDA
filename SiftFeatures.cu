#include <SiftFeatures.h>




__global__ void Convolution(float* image,float* mask, ArrayImage* PyDoG, int maskR,int maskC, int imgR,int imgC, float* imgOut, int idxPyDoG){
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	
		
	int iImg=0;
	float aux=0;
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
				aux=0;;
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
			//aux=(aux<0)?0:aux;
			imgOut[iImg]=aux;//(aux>255)?255:aux;
			aux=0;
		}
	}
	PyDoG[idxPyDoG].image=imgOut;
}

////////////////////arreglar   
__global__ void LocateMaxMin(ArrayImage* PyDoG, int idxPyDoG , float * imgOut ,int maskC, int imgR,int imgC)
{
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	
		
	int iImg=0;
	int pxlThrd = ceil((double)(imgC*imgR)/(gDim*bDim)); ////////numero de veces que caben
														 ////////los hilos en la imagen.

	for(int i = 0; i <pxlThrd; ++i)///////////////////////////// Strike 
	
	{
		int min=0;
		int max=0;
		float value=0.0;
		float compare =0.0;
		//////////////////////////////////////
		//////////////////////////////////////Calculo de indices
		iImg=(tid+(bDim*bid)) + (i*gDim*bDim); //// pixel en el que trabajara el hilo
		//////////////////////////////////////
		//////////////////////////////////////
		
		if(iImg < imgC*imgR){
			
			
			int condition=(maskC/2)+imgC*(maskC/2);
			if (iImg-condition < 0  ||										///condicion arriba
				iImg+condition > imgC*imgR ||								///condicion abajo
				iImg%imgC < maskC/2 ||										///condicion izquierda
				iImg%imgC > (imgC-1)-(maskC/2) )							///condicion derecha
			{
				imgOut[iImg]=0.5;				
			}
			else{
				imgOut[iImg]=1.0;
				value=PyDoG[idxPyDoG].image[iImg];
				
				for (int m = -1; m < 2; ++m)
				{
					int itImg=iImg-(1+imgC);
					
					for (int j = 0; j < 3; ++j)
					{		
						for (int h = 0; h < 3; ++h)
						{
							compare =PyDoG[idxPyDoG+m].image[itImg];
							//if(iImg==10000)printf("%i iImg %i itImg  %i m  %i j %i h\n",iImg,itImg, m,j,h );
							if(value<compare && max==0)
							{
								++min;
							}
							else if(value>compare && min==0)
							{
								++max;
							}
							++itImg;
						}
						itImg+=imgC-3;
					}
				}
  
				if(min==26){
					/////Es Punto extremo;
					 imgOut[iImg]=0.0;
				}else if(max==26){
					/////Es Punto extremo;
					 imgOut[iImg]=0.0;
				}else{
					imgOut[iImg]=0.5;
				}
			
            }
		}
	}
}








void MaskGenerator(double sigma, int size,Mat mask){//Generate Gaussian Kernel
	Mat aux = getGaussianKernel(size,sigma,CV_32F);
	Mat aux_t;
	transpose(aux,aux_t);
	mask=aux*aux_t;
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
	double k = 1.5;
	double sigma= sqrt(2)/6;
	vector<Mat> PyGauss;
	int size = 9;//size of gaussian mask
	Mat mask=Mat::ones(size,size,CV_32F);
	MaskGenerator(sigma,size,mask);
	PyGauss.push_back(mask);

	for(int i=1; i<intvls+3; ++i){	
		Mat aux=Mat::ones(size,size,CV_32F);
		sigma*=k;
		MaskGenerator(sigma,size,aux);
		PyGauss.push_back(aux);
		
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

int SiftFeatures(Mat Image, vector<Mat> PyDoG){
	const int intvls = 2;
	int octvs;
	//cudaError_t e;
	octvs = log( min( Image.rows, Image.cols ) ) / log(2) - 2;
	vector<Mat> PyKDoG;
	vector<Mat> images;
	PyramidKDoG( PyKDoG,octvs,intvls);
	ResizeImage(Image,images,octvs);
	int idxPyDoG=0;
	
	
	ArrayImage * pyDoG;
	//MinMax * minMax;
	int mMidx=1;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
	cudaMalloc(&pyDoG,sizeof(ArrayImage)*images.size()*sizeof(ArrayImage) *PyKDoG.size());
	//cudaMalloc(&minMax,sizeof(MinMax)/*No se tamaño del arreglo*/);
	//cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;

	for (int i = 0; i < images.size() ; ++i)
	//for (int i = 0; i < 1 ; ++i)
	{
		
		float * img_D;
		int sizeImage = images[i].rows*images[i].cols;
		
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
		
		cudaMalloc(&img_D,sizeof(float)*sizeImage);///imagenes
		//cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;
	
		////////////////////////////////////////////////////////////////////////////////////////
		/////////////////////////////////////////////////////////////////////Copio Memoria GPU

		cudaMemcpy(img_D,images[i].ptr<float>(),sizeof(float)*sizeImage,cudaMemcpyHostToDevice);
		//cout<<cudaGetErrorString(e)<<" cudaMemCopyHD"<<endl;

		int imgBlocks= ceil((double) images[i].cols/BW);
		
		////////////////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////Convolucion de mascara con imagen
		/////////////////////////////////////////////////////////////////////Una Octava or ciclo
		for (int m = 0; m < PyKDoG.size(); ++m){
			float * pkDoG_D;
			float * out_D;
			float * out= new float[sizeImage];
			int sizeMask=PyKDoG[m].rows*PyKDoG[m].cols;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
			cudaMalloc(&pkDoG_D,sizeof(float)*sizeMask);//mascaras
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask "<<endl;
			cudaMalloc(&out_D,sizeof(float)*sizeImage);
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask"<<endl;
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Copio Memoria GPU

			cudaMemcpy(pkDoG_D,PyKDoG[m].ptr<float>(),sizeof(float)*sizeMask,cudaMemcpyHostToDevice);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyHD________Mask"<<endl;

			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Lanzo Kernel
			
			Convolution<<<imgBlocks,1024>>>(img_D,pkDoG_D,pyDoG,PyKDoG[m].rows,PyKDoG[m].cols,images[i].rows,images[i].cols,out_D,idxPyDoG);
			cudaDeviceSynchronize();
			++idxPyDoG;
			cudaFree(pkDoG_D); 
			

			cudaMemcpy(out,out_D,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			imshow("tesuto",image_out*5);
    		waitKey(0);
    		destroyAllWindows();

			delete(out);
			//cudaFree(out_D);
		}
		cudaFree(img_D);
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////Busqueda de MinMax
		/////////////////////////////////////////////////////////////////////Una Octava or ciclo
		for(int m = mMidx; m < idxPyDoG-1; ++m){
			
			float * out_D;
			float * out= new float[sizeImage];
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
			cudaMalloc(&out_D,sizeof(float)*sizeImage);
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask"<<endl;
						
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Lanzo Kernel
			cout<<m<<endl;
			LocateMaxMin<<<imgBlocks,1024>>>(pyDoG,m,out_D,PyKDoG[m].cols,images[i].rows,images[i].cols);
			//LocateMaxMin<<<1,1>>>(pyDoG,mMidx,out_D,PyKDoG[m].cols,images[i].rows,images[i].cols);
			cudaDeviceSynchronize();
						

			cudaMemcpy(out,out_D,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			imshow("tesuto",image_out);
    		waitKey(0);
    		destroyAllWindows();

			delete(out);
			//cudaFree(out_D);
		}
		mMidx=idxPyDoG+1;
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////

	}
	cudaFree(pyDoG);



	return 0;
}
