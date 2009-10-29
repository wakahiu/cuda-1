#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

static __global__ void inc_gpu(float *a_d, long time,  int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < N) {
		++a_d[idx];
	}
}

int main(int argc, char *argv[])
{
	float *a_h, *a_d; 	// host data
	int N = 1000, nBytes, i;

	nBytes = N*sizeof(float);
	a_h = (float *)malloc(nBytes);

	cudaMalloc((void **)&a_d, nBytes);

	for(i=0; i < N; i++) {
		a_h[i] = i;
	}

	cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);

	int blocksize = 56;

	dim3 dimBlock(blocksize);
	dim3 dimGrid( (int)ceil(N / (float) blocksize) );

	inc_gpu<<<dimGrid, dimBlock>>>(a_d, time(NULL), N);

	cudaMemcpy(a_h, a_d, nBytes, cudaMemcpyDeviceToHost);

	for(i=0; i < N; i++) {
		printf("%f \n", a_h[i]);
	}
	
	free(a_h);
	cudaFree(a_d);

	return 0;
}

