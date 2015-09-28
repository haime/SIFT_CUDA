
struct Features
{
	int* ftr;
};

struct Match
{
	int idxF;
	int idxO;
	float  Dist;
};



__global__ void Match(Features * ftrsImg, Features * ftrsObj, int ftrsImg_size, int ftrsObj_size, Match * match, float around )
{
	int tid= threadIdx.x;
	int bid= blockIdx.x;
	int bDim=blockDim.x;
	int gDim=gridDim.x;
	int tidx=0;
	
	int pxlThrd = ceil((double)(ftrsImg_size)/(gDim*bDim)); 

	for(int s = 0; s <pxlThrd; ++s)///////////////////////////// Strike 
	{

		tidx=(tid+(bDim*bid)) + (s*gDim*bDim);


		for(int i=0; i<ftrsObj_size; ++i)
		{
			int dist=0;
			for (int j = 0; j < 128; ++j)
			{
				dist += (ftrsImg[tidx].ftr[j]-ftrsObj[i].ftr[j])*(ftrsImg[tidx].ftr[j]-ftrsObj[i].ftr[j])
			}
			if( dist < around && dist< match[tidx].Dist )
			{
				match[tidx].idxF= tidx;
				match[tidx].idxO= i;
				match[tidx].Dist= dist;
			}
		}
	}
}
