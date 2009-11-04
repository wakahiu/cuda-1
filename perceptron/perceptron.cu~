#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Park-Miller quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////

static __global__ void parkmillerKernel(float *d_Output, unsigned int seed, int cycles,unsigned int N) 
{
    unsigned int      tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
    unsigned int  threadN = MUL(blockDim.x, gridDim.x);

    float const a		= 16807;      //ie 7**5
    float const m		= 2147483647; //ie 2**31-1
    float const reciprocal_m	= 1.0/m;

    for(unsigned int pos = tid; pos < N; pos += threadN)
    {
        unsigned int result = 0;
        unsigned int data   = seed + pos;

	for (int i=1; i <= cycles; i++) 
	{
		float temp = data * a;
		result = (int) (temp - m * floor ( temp * reciprocal_m ));
		data = result;
	}

        d_Output[pos] = result / m;
    }
}

int main(int argc, char *argv[])
{
	float *a_h, *a_d;	// host data
	int nBytes, i;

	int seed = 1;

	unsigned int N=10000; //QRNG_DIMENSIONS;
	int cycles = 1000;

	nBytes = N*sizeof(float);
	a_h = (float *)malloc(nBytes);

	cudaMalloc((void **)&a_d, nBytes);

	// for(i=0; i < N; i++) {
	// 	a_h[i] = i;
	// }

	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);

	int blocksize = 128*3;

	dim3 dimBlock(blocksize);
	dim3 dimGrid( 128 );

	parkmillerKernel<<<dimGrid, dimBlock>>>(a_d, seed, cycles, N);

	cudaMemcpy(a_h, a_d, nBytes, cudaMemcpyDeviceToHost);

	for(i=0; i < 100; i++) {
		printf("%f \n", a_h[i]);
	}
	
	free(a_h);
	cudaFree(a_d);

	return 0;
}

