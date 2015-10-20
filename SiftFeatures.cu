#include <SiftFeatures.h>




__global__ void Convolution(float* image,float* mask, ArrayImage* PyDoG, int maskR,int maskC, int imgR,int imgC, float* imgOut, int idxPyDoG)
{
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
			
			
			imgOut[iImg]=aux;
			aux=0;
		}
	}
	PyDoG[idxPyDoG].image=imgOut;
}

__global__ void LocateMaxMin(ArrayImage* PyDoG, int idxPyDoG , float * imgOut ,MinMax * mM, int maskC, int imgR,int imgC, int idxmM)
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
			
			
			int condition=maskC/2+imgC*(floor((double)maskC/2));
			if (iImg-condition < 0  ||										///condicion arriba
				iImg+condition > imgC*imgR ||								///condicion abajo
				iImg%imgC < maskC/2 ||										///condicion izquierda
				iImg%imgC > (imgC-1)-(maskC/2) )							///condicion derecha
			{                  
				imgOut[iImg]=0;				
			}
			else{
				
				value=PyDoG[idxPyDoG].image[iImg];
				
				for (int m = -1; m < 2; ++m)
				{
					int itImg=iImg-(1+imgC);
					
					for (int j = 0; j < 3; ++j)
					{		
						for (int h = 0; h < 3; ++h)
						{
							compare =PyDoG[idxPyDoG+m].image[itImg];
							if(value<=compare && max==0)
							{
								++min;
							}
							else if(value>=compare && min==0)
							{
								++max;
							}
							++itImg;
						}
						itImg+=imgC-3;
					}
				}
  				//printf("%i %i\n",min,max );
				if( (min==26 || max==26)) {
					imgOut[iImg]=1;
				}else{
					imgOut[iImg]=0;
				}
			
            }
		}
	}
	mM[idxmM].minMax=imgOut;
}


__global__ void RemoveOutlier(ArrayImage* PyDoG, MinMax * mM, int idxmM, int idxPyDoG, int imgR,int imgC ,float* auxOut)
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
		//////////////////////////////////////
		//////////////////////////////////////Calculo de indices
		iImg=(tid+(bDim*bid)) + (i*gDim*bDim); //// pixel en el que trabajara el hilo
		//////////////////////////////////////
		//////////////////////////////////////
		
		if(iImg < imgC*imgR){
			
			

			if(mM[idxmM].minMax[iImg]>0 && PyDoG[idxPyDoG].image[iImg]>0.02)
			{
				
				float d, dxx, dyy, dxy, tr, det;
				d = PyDoG[idxPyDoG].image[iImg];
				dxx = PyDoG[idxPyDoG].image[iImg+1]+ PyDoG[idxPyDoG].image[iImg-1] - (2*d);
				dyy = PyDoG[idxPyDoG].image[iImg+imgC]+ PyDoG[idxPyDoG].image[iImg-imgC] - (2*d);
				dxy = (PyDoG[idxPyDoG].image[iImg+1+imgC]- PyDoG[idxPyDoG].image[iImg-1+imgC] - PyDoG[idxPyDoG].image[iImg+1-imgC] + PyDoG[idxPyDoG].image[iImg-1-imgC])/4.0;
				tr = dxx + dyy;
				det = dxx*dyy - dxy*dxy;
				
				if(det<=0 && !(tr*tr/det < 12.1))
					mM[idxmM].minMax[iImg]=0;

			}else
			{
				mM[idxmM].minMax[iImg]=0;
			}

			auxOut[iImg]=mM[idxmM].minMax[iImg];
			

		}
	}
	
}



__global__ void OriMag(ArrayImage* PyDoG, int idxPyDoG, int imgR,int imgC , ArrayImage* Mag, ArrayImage* Ori, int idxMagOri, float* MagAux, float* OriAux) 
{
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	float dx,dy;
			
	int iImg=0;
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
			int condition=1/2+imgC*(floor((double)1/2));
			if (iImg-condition < 0  ||										///condicion arriba
				iImg+condition > imgC*imgR ||								///condicion abajo
				iImg%imgC < 1/2 ||										///condicion izquierda
				iImg%imgC > (imgC-1)-(1/2) )							///condicion derecha
			{                  
				OriAux[iImg]=0;
				MagAux[iImg]=0;

			}
			else{
				dx=PyDoG[idxPyDoG].image[iImg+1]-PyDoG[idxPyDoG].image[iImg-1];
				dy=PyDoG[idxPyDoG].image[iImg+imgC]-PyDoG[idxPyDoG].image[iImg-imgC];
				MagAux[iImg]=sqrt(dx*dx + dy*dy);
				OriAux[iImg]=atan2(dy,dx);
            }
		}
	}
	
	Mag[idxMagOri].image= MagAux;
	Ori[idxMagOri].image= OriAux;

	
}



__global__ void KeyPoints(ArrayImage * Mag, ArrayImage * Ori, MinMax * mM , int idxMOmM, keyPoint * KP, float sigma, int imgR,int imgC, int octava )
{
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	float o = 0;
	int x=0, y=0, octv=0;

	


	int iImg=0;
	int pxlThrd = ceil((double)(imgC*imgR)/(gDim*bDim)); ////////numero de veces que caben
														 ////////los hilos en la imagen.
	for(int i = 0; i <pxlThrd; ++i)///////////////////////////// Strike 
	{
		//////////////////////////////////////
		//////////////////////////////////////Calculo de indices
		iImg=(tid+(bDim*bid)) + (i*gDim*bDim); //// pixel en el que trabajara el hilo
		//////////////////////////////////////
		//////////////////////////////////////
		
		if(iImg < imgC*imgR ){

			if(mM[idxMOmM].minMax[iImg]>0){
				

				int condition=9/2+imgC*(floor((double)9/2));
				if (iImg-condition < 0  ||										///condicion arriba
					iImg+condition > imgC*imgR ||								///condicion abajo
					iImg%imgC < 9/2 ||										///condicion izquierda
					iImg%imgC > (imgC-1)-(9/2) )							///condicion derecha
				{                  
					
					o=-1.0;
					x=-1;
					y=-1;
					octv=-1;
				}
				else{
					
					float histo[36]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
					octv=octava;
					x=iImg%imgC;
					y=iImg/imgC;
					
					int idxMO= (iImg-4)-(4*imgC);
					float exp_denom = 2.0 * sigma * sigma;
					float w;
					int bin;

					for (int i = -4; i < 5; ++i)
					{
						for (int j = -4; j < 5; ++j)
						{
							w = exp( -( i*i + j*j ) / exp_denom );
	  						bin = round((double) ( (36*Ori[idxMOmM].image[idxMO])/6.28318530718));
	  						bin = ( bin < 36 )? bin : 0;
	  						histo[bin]= w*Mag[idxMOmM].image[idxMO];
	  						++idxMO;
						}
						idxMO=idxMO+imgC-9;
					}

					int idxH=0;
					float valMaxH = histo[0];
					for (int i = 1; i < 36; ++i)
					{
						if(histo[i]>valMaxH){
							idxH = i;
						}
					}
					int l = (idxH == 0)? 35:idxH-1;
					int r = (idxH+1)%36;

					float bin_= bin + ((0.5*(histo[l]-histo[r]))/(histo[l]-(2*histo[idxH])+histo[r]));
					bin_= ( bin_ < 0 )? 36 + bin_ : ( bin_ >= 36 )? bin_ - 36 : bin_;
					o=((6.28318530718*bin_)/36)-3.141592654;
		        }


		        KP[iImg].orientacion=o;
		        KP[iImg].x=x;
		        KP[iImg].y=y;
		        KP[iImg].octv=octv;


        	}
		}
	}
	
	



}



__global__ void CountKeyPoint(MinMax * mM, int idxmM, int imgR, int imgC, int * numKeyP)
{
	int tid= threadIdx.x;
	int bDim=blockDim.x;
	
	
	__shared__ int num;
	int iImg=0;
	int pxlThrd = ceil((double)(imgC*imgR)/bDim); ////////numero de veces que caben
	if(tid==0) num=0;
	__syncthreads();
															
	for(int i = 0; i < pxlThrd; ++i)///////////////////////////// Strike
	{
		iImg= tid+(i*bDim);
		if(iImg < imgC*imgR && mM[idxmM].minMax[iImg]>0){
			atomicAdd(&num,1);
		}

	}

	numKeyP[0]=num;
	

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

int foundIndexesMaxMin(float* minMax,vector<int*> & idxMinMax, int count )
{
	vector<int> idxmM;
	for (int c = 0; c <  count; ++c)
	{
		
		if (minMax[c]==0.0)
		{
			idxmM.push_back(c);
			//cout<<c<<endl;
		}
	}
	idxMinMax.push_back(idxmM.data());
	

	return 0;
}

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
	
	
	ArrayImage * pyDoG;
	MinMax * minMax;
	int mMidx=1;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Reservo Memoria GPU



	cudaMalloc(&pyDoG,sizeof(ArrayImage)*images.size()*PyKDoG.size());
	cudaMalloc(&minMax,sizeof(MinMax)*intvls*images.size());
	//cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;
	
	for (int i = 0; i < images.size() ; ++i)
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
			//float * out= new float[sizeImage];
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
			//cudaDeviceSynchronize();
			++idxPyDoG;
			cudaFree(pkDoG_D); 
			
			
			//cudaMemcpy(out,out_D,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			//Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			//imshow("tesuto",image_out);
    		//waitKey(0);
    		//destroyAllWindows();
			
			//delete(out);
			//cudaFree(out_D);
		}
		cudaFree(img_D);
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
	}

	int maskC =PyKDoG[0].cols;
	int idxmM=0;
	for (int i = 0; i <images.size() ; ++i)
	{
		int sizeImage = images[i].rows*images[i].cols;
		int imgBlocks= ceil((double) images[i].cols/BW);
		////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////Busqueda de MinMax
		/////////////////////////////////////////////////////////////////////Una Octava or ciclo
		
		int m=0;
		for(m = mMidx; m < mMidx+intvls; ++m){
			
			float * out_D;
			//float * out = new float[sizeImage];
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
			cudaMalloc(&out_D,sizeof(float)*sizeImage);
			//cout<<cudaGetErrorString(e)<<" cudaMalloc________Mask"<<endl;
						
			////////////////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////Lanzo Kernel
			///////entrega ya los puntos descartanbdo los de bajo contraste
			LocateMaxMin<<<imgBlocks,1024>>>(pyDoG,m,out_D,minMax,maskC,images[i].rows,images[i].cols,idxmM);
			++idxmM;
			//cudaDeviceSynchronize();

			//cudaMemcpy(out,out_D,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			//cout<<cudaGetErrorString(e)<<" cudaMemCopyDH________Mask"<<endl;

			//Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			//imshow("tesuto",image_out);
    		//waitKey(0);
    		//destroyAllWindows();
			
			//delete(out);
			//cudaFree(out_D);
		}
		mMidx=m+2;
		
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
		////////////////////////////////////////////////////////////////////////////////////////
	}



	///////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Remover outliers



	idxPyDoG=1, idxmM=0;
	for(int i = 0; i< images.size(); ++i )
	{	float* out_D;
		int sizeImage = images[i].rows*images[i].cols;
		int imgBlocks= ceil((double) images[i].cols/BW);
		cudaMalloc(&out_D,sizeof(float)*sizeImage);
		float * out = new float[sizeImage];

		for (int j = 0; j < intvls; ++j)
		{
			RemoveOutlier<<<imgBlocks,1024>>>(pyDoG,minMax,idxmM,idxPyDoG, images[i].rows,images[i].cols,out_D);
			cudaMemcpy(out,out_D,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			

			Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			imshow("tesuto",image_out);
    		waitKey(0);
    		destroyAllWindows();
    		
    		
			++idxmM;
			++idxPyDoG;
		}
		idxPyDoG+=2;

		delete(out);
		cudaFree(out_D);
	}




	///////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Calculo de Orientaciones y magnitud en DoG
	

	ArrayImage * Mag;
	ArrayImage * Ori;

	cudaMalloc(&Mag,sizeof(ArrayImage)*intvls*images.size());
	cudaMalloc(&Ori,sizeof(ArrayImage)*intvls*images.size());

	idxPyDoG=1;
	int idxMagOri=0;
	for(int i = 0; i< images.size(); ++i )
	{	
		float * MagAux;
		float * OriAux;
		int sizeImage = images[i].rows*images[i].cols;
		int imgBlocks= ceil((double) images[i].cols/BW);
		cudaMalloc(&MagAux,sizeof(float)*sizeImage);
		cudaMalloc(&OriAux,sizeof(float)*sizeImage);
		float * out = new float[sizeImage];

		for (int j = 0; j < intvls; ++j)
		{
			OriMag<<<imgBlocks,1024>>>(pyDoG,idxPyDoG, images[i].rows,images[i].cols,Mag,Ori,idxMagOri,MagAux,OriAux);
			cudaMemcpy(out,MagAux,sizeof(float)*sizeImage,cudaMemcpyDeviceToHost);
			

			Mat image_out(images[i].rows,images[i].cols,CV_32F,out);
			
			imshow("tesuto",image_out);
    		waitKey(0);
    		destroyAllWindows();

			++idxMagOri;
			++idxPyDoG;
		}
		idxPyDoG+=2;



		delete(out);
		cudaFree(MagAux);
		cudaFree(OriAux);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Obtener orientacion de keypoints


	
	





















	/*



	idxmM=0;
	int * numKeyP_D;
	cudaMalloc(&numKeyP_D,sizeof(int));
	for(int i =0 ; i<images.size(); ++i)
	//for(int i =0 ; i<1 ; ++i)
	{
		int numKeyP;
		for(int j =0; j<intvls ; ++j)
		{
			CountKeyPoint<<<1,1024>>>(minMax,idxmM,images[i].rows,images[i].cols,numKeyP_D);
			
			cudaMemcpy(&numKeyP,numKeyP_D,sizeof(int),cudaMemcpyDeviceToHost);
			cout<<numKeyP<<endl;
			++idxmM;
		}
	}
	cudaFree(numKeyP_D);


	*/
	cudaFree(Ori);
	cudaFree(Mag);
	cudaFree(pyDoG);
	cudaFree(minMax);

	return 0;
}




