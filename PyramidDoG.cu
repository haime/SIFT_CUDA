#include <PyramidDoG.h>




__global__ void ConvolutionDoG(ArrayImage* images,ArrayImage* mask, ArrayImage* PyDoG, int idxImages, int maskSize){
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	int idxPyDoG = idxImages*maskSize;
	
	
	for (int idxMask = 0; idxMask < maskSize; ++idxMask)
	{
		int iImg=0;
		float aux=0;
		int pxlThrd = ceil((double)(images[idxImages].cols*images[idxImages].rows)/(gDim*bDim)); ////////numero de veces que caben
																								 ////////los hilos en la imagen.
		
	
		for (int i = 0; i <pxlThrd; ++i)///////////////////////////// Strike 
		{
			//////////////////////////////////////
			//////////////////////////////////////Calculo de indices
			iImg=(tid+(bDim*bid)) + (i*(gDim*bDim)); //// pixel en el que trabajara el hilo
			//////////////////////////////////////
			//////////////////////////////////////
			
			
			if(iImg < images[idxImages].cols*images[idxImages].rows){
				
				int condition=mask[idxMask].cols/2+images[idxImages].cols;
				 
				if (iImg-condition < 0  ||												///condicion arriba
					iImg+condition > images[idxImages].cols*images[idxImages].rows ||	///condicion abajo
					iImg%images[idxImages].cols < mask[idxMask].cols/2 ||				///condicion izquierda
					iImg%images[idxImages].cols >=mask[idxMask].cols/2 )					///condicion derecha
				{
					aux=0;
					
				}else{		
					
					int itMask = 0;
					int itImg=iImg-condition;
					for (int j = 0; j < mask[idxMask].rows; ++j)
					{		
						for (int h = 0; h < mask[idxMask].cols; ++h)
						{
							aux+=images[idxImages].image[itImg]*mask[idxMask].image[itMask];
							++itMask;
							++itImg;
						}
						itImg+=images[idxImages].cols-mask[idxMask].cols;
					}
				
				}
				//if(tid==0)printf("%i pxlThrd \n", pxlThrd);
				if(tid==0)printf(" %i, ", idxPyDoG);
				PyDoG[idxPyDoG].image[iImg]=aux;//////////////////////////////
				
				aux=0;
			}
		}
		if(tid==0)printf(" %i, ", idxPyDoG);
		++idxPyDoG;
		__syncthreads();
		

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
	ArrayImage * imgs= new ArrayImage[images.size()];
	ArrayImage * pkDoG= new ArrayImage[PyKDoG.size()];
	VectorToPointer(images,imgs);
	VectorToPointer(PyKDoG,pkDoG);
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Reservo Memoria GPU
	ArrayImage * imgs_D;
	ArrayImage * pkDoG_D;
	ArrayImage * pyDoG_Out;
	e=cudaMalloc(&imgs_D,sizeof(ArrayImage)*images.size());///imagenes
	cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;
	
	e=cudaMalloc(&pkDoG_D,sizeof(ArrayImage)*PyKDoG.size());//mascaras
	cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;

	e=cudaMalloc(&pyDoG_Out,sizeof(ArrayImage)*images.size()*sizeof(ArrayImage)*PyKDoG.size());
	cout<<cudaGetErrorString(e)<<" cudaMalloc"<<endl;

	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Copio Memoria GPU
	e=cudaMemcpy(imgs_D,imgs,sizeof(ArrayImage)*images.size(),cudaMemcpyHostToDevice);
	cout<<cudaGetErrorString(e)<<" cudaMemCopyHD"<<endl;
	e=cudaMemcpy(pkDoG_D,pkDoG,sizeof(ArrayImage)*PyKDoG.size(),cudaMemcpyHostToDevice);
	cout<<cudaGetErrorString(e)<<" cudaMemCopyHD"<<endl;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////Lanzo Kernel
	for (int i = 0; i < images.size(); ++i)
	{
		ConvolutionDoG<<<16,32>>>(imgs_D,pkDoG_D,pyDoG_Out,i,PyKDoG.size());/////i es el indice de la imagen que se va a emborrona
		cudaDeviceSynchronize();
		cout<<""<<endl;
		cout<<i<<endl;
		cout<<""<<endl;
	}
	//cout<< PyKDoG.size() <<endl;
	
	ArrayImage * img_out_test = new ArrayImage[sizeof(ArrayImage)*images.size()*sizeof(ArrayImage)*PyKDoG.size()];

	e=cudaMemcpy(img_out_test,pyDoG_Out,sizeof(ArrayImage)*images.size()*sizeof(ArrayImage)*PyKDoG.size(),cudaMemcpyDeviceToHost);
	cout<<cudaGetErrorString(e)<<" cudaMemCopyDH"<<endl;




	cudaFree(imgs_D);
	cudaFree(pkDoG_D);
	cudaFree(pyDoG_Out);
	free(img_out_test);

	return 0;
}
