#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda.h>
#include <math_functions.h>

__device__ inline float MoroInvCNDgpu(float P){
    const float a1 = 2.50662823884f;
    const float a2 = -18.61500062529f;
    const float a3 = 41.39119773534f;
    const float a4 = -25.44106049637f;
    const float b1 = -8.4735109309f;
    const float b2 = 23.08336743743f;
    const float b3 = -21.06224101826f;
    const float b4 = 3.13082909833f;
    const float c1 = 0.337475482272615f;
    const float c2 = 0.976169019091719f;
    const float c3 = 0.160797971491821f;
    const float c4 = 2.76438810333863E-02f;
    const float c5 = 3.8405729373609E-03f;
    const float c6 = 3.951896511919E-04f;
    const float c7 = 3.21767881768E-05f;
    const float c8 = 2.888167364E-07f;
    const float c9 = 3.960315187E-07f;
    float y, z;

    if(P <= 0 || P >= 1.0f)
        return __int_as_float(0x7FFFFFFF);

    y = P - 0.5f;
    if(fabsf(y) < 0.42f){
        z = y * y;
        z = y * (((a4 * z + a3) * z + a2) * z + a1) / ((((b4 * z + b3) * z + b2) * z + b1) * z + 1.0f);
    }else{
        if(y > 0)
            z = __logf(-__logf(1.0f - P));
        else
            z = __logf(-__logf(P));

        z = c1 + z * (c2 + z * (c3 + z * (c4 + z * (c5 + z * (c6 + z * (c7 + z * (c8 + z * c9)))))));
        if(y < 0) z = -z;
    }

    return z;
}

__device__ inline float AcklamInvCNDgpu(float P){
    const float   a1 = -39.6968302866538f;
    const float   a2 = 220.946098424521f;
    const float   a3 = -275.928510446969f;
    const float   a4 = 138.357751867269f;
    const float   a5 = -30.6647980661472f;
    const float   a6 = 2.50662827745924f;
    const float   b1 = -54.4760987982241f;
    const float   b2 = 161.585836858041f;
    const float   b3 = -155.698979859887f;
    const float   b4 = 66.8013118877197f;
    const float   b5 = -13.2806815528857f;
    const float   c1 = -7.78489400243029E-03f;
    const float   c2 = -0.322396458041136f;
    const float   c3 = -2.40075827716184f;
    const float   c4 = -2.54973253934373f;
    const float   c5 = 4.37466414146497f;
    const float   c6 = 2.93816398269878f;
    const float   d1 = 7.78469570904146E-03f;
    const float   d2 = 0.32246712907004f;
    const float   d3 = 2.445134137143f;
    const float   d4 = 3.75440866190742f;
    const float  low = 0.02425f;
    const float high = 1.0f - low;
    float z, R;

    if(P <= 0 || P >= 1.0f)
        return __int_as_float(0x7FFFFFFF);

    if(P < low){
        z = sqrtf(-2.0f * __logf(P));
        z = (((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) /
            ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0f);
    }else{
        if(P > high){
            z = sqrtf(-2.0 * __logf(1.0 - P));
            z = -(((((c1 * z + c2) * z + c3) * z + c4) * z + c5) * z + c6) /
                 ((((d1 * z + d2) * z + d3) * z + d4) * z + 1.0f);
        }else{
            z = P - 0.5f;
            R = z * z;
            z = (((((a1 * R + a2) * R + a3) * R + a4) * R + a5) * R + a6) * z /
                (((((b1 * R + b2) * R + b3) * R + b4) * R + b5) * R + 1.0f);
        }
    }

    return z;
}

__device__ long dummy;

__device__ float iv[32];
__device__ long dummy2;
__device__ long iy;

__host__ __device__ inline void Seed(long dum) { 
	iy = 0;
	dummy2 = 123456789;
	dummy = dum;  
}

__device__ inline float unirand0(void)
{
	const long im	= 2147483657;
	const float am	= (1./im);
	const long ia	= 16807;
	
	const long ntab		= 32;
	const long nwup		= 8;
	const float ndiv	= (1 + (im - 1)/ntab);
	const float eps		= 1.2e-7;
	const float rnmx	= (1.0 - eps);

	const long iq = 12773;
	const long ir = 2836;

	const long mask = 123456789;
	
	float ans;
	long k;
	
	dummy ^= mask; 

	/* avoid dummy==0 */ 

	k = dummy/iq; 

	if((dummy = ia*(dummy - k*iq) - ir*k) < 0) 
		dummy += im; 
	
	ans = am*dummy; 
	dummy ^= mask; 
	/* restore unmasked dummy */ 

	return(ans); 
}

__device__ float unirand2(float a) 
{
	const float eps		= 1.2e-7;
	const long ntab		= 32;
	const long nwup		= 8;
	const float rnmx	= (1.0 - eps);
	const long im1		= 2147483563;
	const long im2		= 2147483399;
	const float am		= (1./im1); 
	const float imm1	= (im1 - 1);
	const long ia1		= 40014;
	const long ia2		= 40692;
	const long iq1		= 53668;
	const long iq2		= 52774;
	const long ir1		= 12211;
	const long ir2		= 3791;
	const float ndiv	= (1 + imm1/ntab);

	int j;
	long k;
	// long dummy2	= 123456789;
	// long iy		= 0;
	// long iv[ntab];
	float temp;

	/* initialize the random sequence (first set of coefficients, the 
	   routine close to that in the function above */
	if(dummy<=0 || !iy) 
	{
		/* avoid negative or zero seed */
		if(dummy<0) dummy=-dummy;
		else if(dummy==0) dummy=1;
		dummy2=dummy;

		/* after NWUP warmups, initialize shuffle table */
		for(j=ntab + nwup - 1;j >= 0; j--) 
		{
			k=dummy/iq1;
			if((dummy=ia1*(dummy-k*iq1)-ir1*k)<0) dummy += im1;
		
			if(j < ntab) iv[j] = dummy;
		}

		/* first specimen from the table */
		iy = iv[0];
	}

	/* regular work: generate 2 sequences. */
	k = dummy/iq1;
	if((dummy = ia1 * (dummy - k*iq1) - ir1 * k) < 0) dummy += im1;

	k = dummy2/iq2;
	if( (dummy2 = ia2*(dummy2 - k*iq2)-ir2*k) < 0 ) dummy2 += im2;

	/* shuffle output combining 2 sequences */
	iy=iv[j=iy/ndiv]-dummy2;iv[j]=dummy;
	if(iy<1) iy+=imm1;

	/* return the result, as in the previous function */
	if((temp = am*iy) > rnmx) return(rnmx*a);
	else return(temp*a);
}

#define MUL(a, b) __umul24(a, b)

static __global__ void inc_gpu(float *a_d, long time,  int N) 
{
	// unsigned int     tid = MUL(blockDim.x, blockIdx.x) + threadIdx.x;
	// unsigned int threadN = MUL(blockDim.x, gridDim.x);
	float		   q = (float) 1.0 / (float) (N+1);

	int idx = blockIdx.x * blockDim.x + threadIdx.x;


	if(idx < N) {
		float d = (float)(idx + 1) * q;
		a_d[idx] = (float) unirand2(d);
	}

	// for(unsigned int pos = tid; pos < N; pos += threadN)
	// {
	// 	 float d = (float)(pos + 1) * q;
	// 	 a_d[pos] = (float)AcklamInvCNDgpu(d);
	// }
	
	// Seed(time);

	// for(unsigned int pos = tid; pos < N; pos += threadN)
	// {
	// 	float d = (float)(pos + 1) * q;
	// 	a_d[pos] = (float)unirand2(d);
	// }
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

	Seed(time(NULL));

	inc_gpu<<<dimGrid, dimBlock>>>(a_d, time(NULL), N);

	cudaMemcpy(a_h, a_d, nBytes, cudaMemcpyDeviceToHost);

	for(i=0; i < N; i++) {
		printf("%f \n", a_h[i]);
	}
	
	free(a_h);
	cudaFree(a_d);

	return 0;
}

