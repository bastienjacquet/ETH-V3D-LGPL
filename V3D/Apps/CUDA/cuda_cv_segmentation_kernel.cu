#include "cuda_cv_segmentation.h"

#include <cuda.h>

#ifdef _WINDOWS
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <cstdio>

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

struct CUDA_CV_Segmentation_Buffers
{
      int w, h;

      float *d_u, *d_p1, *d_p2;
      float *d_g;
      float *d_c1, *d_usum;

      float fSum;
};

#define DIM_X 16
#define DIM_Y 8

void
CUDA_CV_Segmentation::allocate(int w, int h)
{
   // Allocate additionals row to avoid a few conditionals in the kernel
   int const size = sizeof(float) * w * (h+1);

   _w = w;
   _h = h;

   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_u, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_p1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_p2, size) );

   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_g, size) );

   int const c1Size = w/DIM_X * h/DIM_Y * sizeof(float);
   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_c1, c1Size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_usum, c1Size) );
} // end CUDA_CV_Segmentation::allocate()

void
CUDA_CV_Segmentation::deallocate()
{
   CUDA_SAFE_CALL( cudaFree(_d_u) );
   CUDA_SAFE_CALL( cudaFree(_d_p1) );
   CUDA_SAFE_CALL( cudaFree(_d_p2) );
   CUDA_SAFE_CALL( cudaFree(_d_g) );
   CUDA_SAFE_CALL( cudaFree(_d_c1) );
   CUDA_SAFE_CALL( cudaFree(_d_usum) );
}

void
CUDA_CV_Segmentation::getResult(float * uDst, float * p1Dst, float * p2Dst)
{
   int const sz = _w*_h*sizeof(float);

   if (uDst) CUDA_SAFE_CALL( cudaMemcpy( uDst, _d_u, sz, cudaMemcpyDeviceToHost) );
   if (p1Dst) CUDA_SAFE_CALL( cudaMemcpy( p1Dst, _d_p1, sz, cudaMemcpyDeviceToHost) );
   if (p2Dst) CUDA_SAFE_CALL( cudaMemcpy( p2Dst, _d_p2, sz, cudaMemcpyDeviceToHost) );
}

void
CUDA_CV_Segmentation::setImageData(float const * fSrc)
{
   int const sz = _w*_h*sizeof(float);

   _fSum = 0.0f;
   for (int y = 0; y < _h; ++y)
      for (int x = 0; x < _w; ++x)
         _fSum += fSrc[y*_w + x];

   CUDA_SAFE_CALL( cudaMemcpy( _d_g, fSrc, sz, cudaMemcpyHostToDevice) );
}

void
CUDA_CV_Segmentation::initSegmentation()
{
   int const sz = _w*_h*sizeof(float);

   CUDA_SAFE_CALL( cudaMemset( _d_u, 0, sz) );
   CUDA_SAFE_CALL( cudaMemset( _d_p1, 0, sz) );
   CUDA_SAFE_CALL( cudaMemset( _d_p2, 0, sz) );
}

//**********************************************************************

#define SYNCTHREADS() __syncthreads()

static __global__ void
_updateUVolume_kernel(int w, int h, float tau, float * U, float const * P1, float const * P2, float const * G)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y*DIM_X];
   __shared__ float p2_sh[DIM_Y*DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float const g = G[curPos];
   p1_sh[ix] = P1[curPos];
   p2_sh[ix] = P2[curPos];

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? P1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? P2[curPos - w] : p2_0;

   float const div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                        ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float u_new = u - tau * (div_p + g);
   U[curPos] = max(0.0f, min(1.0f, u_new));
} // end updateUVolume()

static __global__ void
_updatePVolume_kernel(int w, int h, float alpha, float tau, float * const U, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

#if 1
   float new_p1 = p1_cur - tau * u_x;
   float new_p2 = p2_cur - tau * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha);
   //float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2));
   new_p1 /= norm;
   new_p2 /= norm;
#else
   float const tv = sqrtf(u_x*u_x + u_y*u_y);
   float const denom_p = 1.0f / (1.0f + tau * tv);
   float new_p1 = (p1_cur - tau * u_x) * denom_p;
   float new_p2 = (p2_cur - tau * u_y) * denom_p;
#endif

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end updatePVolume()

//**********************************************************************

static __global__ void
_updateUVolume_kernel(int w, int h, float alpha, float tau, float c1, float c2, float const * F,
                      float * U, float const * P1, float const * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y*DIM_X];
   __shared__ float p2_sh[DIM_Y*DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float const f = F[curPos];
   p1_sh[ix] = P1[curPos];
   p2_sh[ix] = P2[curPos];

   float const g = alpha * ((f-c1)*(f-c1) - (f-c2)*(f-c2));

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? P1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? P2[curPos - w] : p2_0;

   float const div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                        ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float u_new = u - tau * (div_p + g);
   U[curPos] = max(0.0f, min(1.0f, u_new));
} // end updateUVolume()

//**********************************************************************
// The is the codepath of Bressons approach

static __global__ void
_updateUVolume_theta(int w, int h, float alpha, float theta, float c1, float c2,
                     float const * F, float * U, float const * P1, float const * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y*DIM_X];
   __shared__ float p2_sh[DIM_Y*DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float const f = F[curPos];
   p1_sh[ix] = P1[curPos];
   p2_sh[ix] = P2[curPos];

   float const g = ((f-c1)*(f-c1) - (f-c2)*(f-c2));

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? P1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? P2[curPos - w] : p2_0;

   float const div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                        ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float const v = max(0.0f, min(1.0f, u - alpha*theta*g));
   U[curPos] = v - theta * div_p;
} // end updateUVolume()

static __global__ void
_updatePVolume_theta(int w, int h, float tau_over_theta, float * const U, float * P1, float * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int pos = Y*w + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   // Load p and q of current slice/disparity
   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

# if 1
   float new_p1 = p1_cur - tau_over_theta * u_x;
   float new_p2 = p2_cur - tau_over_theta * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2));
   new_p1 /= norm;
   new_p2 /= norm;
# else
   float const tv = sqrtf(u_x*u_x + u_y*u_y);
   float const denom_p = 1.0f / (1.0f + tau * tv);
   float new_p1 = (p1_cur - tau_over_theta * u_x) * denom_p;
   float new_p2 = (p2_cur - tau_over_theta * u_y) * denom_p;
#endif

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end updatePVolume_theta()


//**********************************************************************

static __global__ void
_computeMeans_kernel(int const w, float const * F, float const * U, float * C1, float * Usum)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int const pos = Y*w + X;
   int const dstPos = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;

   __shared__ float c1_sh[DIM_Y][DIM_X];
   __shared__ float usum_sh[DIM_Y][DIM_X];

   float const u = U[pos];
   float const f = F[pos];

   c1_sh[tidy][tidx] = u*f;
   usum_sh[tidy][tidx] = u;
   SYNCTHREADS();

   if (tidy < 4) c1_sh[tidy][tidx] += c1_sh[tidy+4][tidx];
   if (tidy < 4) usum_sh[tidy][tidx] += usum_sh[tidy+4][tidx];
   SYNCTHREADS();
   if (tidy < 2) c1_sh[tidy][tidx] += c1_sh[tidy+2][tidx];
   if (tidy < 2) usum_sh[tidy][tidx] += usum_sh[tidy+2][tidx];
   SYNCTHREADS();
   if (tidy < 1) c1_sh[tidy][tidx] += c1_sh[tidy+1][tidx];
   if (tidy < 1) usum_sh[tidy][tidx] += usum_sh[tidy+1][tidx];
   SYNCTHREADS();
   if (tidy < 2)
   {
      c1_sh[tidy][tidx] += c1_sh[tidy][tidx+8];
      c1_sh[tidy][tidx] += c1_sh[tidy][tidx+4];
      c1_sh[tidy][tidx] += c1_sh[tidy][tidx+2];
      c1_sh[tidy][tidx] += c1_sh[tidy][tidx+1];

      usum_sh[tidy][tidx] += usum_sh[tidy][tidx+8];
      usum_sh[tidy][tidx] += usum_sh[tidy][tidx+4];
      usum_sh[tidy][tidx] += usum_sh[tidy][tidx+2];
      usum_sh[tidy][tidx] += usum_sh[tidy][tidx+1];
   }
   if (tidx == 0 && tidy == 0) C1[dstPos] = c1_sh[0][0];
   if (tidx == 0 && tidy == 0) Usum[dstPos] = usum_sh[0][0];
} // end _computeMeans_kernel()

//**********************************************************************


void
CUDA_CV_Segmentation::runGeneralSegmentation(int nIterations, float alpha, float tau)
{
   dim3 gridDim(_w/DIM_X, _h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int i = 0; i < nIterations; ++i)
   {
      _updateUVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, tau, _d_u, _d_p1, _d_p2, _d_g);
      _updatePVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, 1.0f, tau, _d_u, _d_p1, _d_p2);
   }
} // end cuda_segmentation_CV

void
CUDA_CV_Segmentation::runSegmentation(float c1, float c2, int nIterations, float alpha, float tau)
{
   dim3 gridDim(_w/DIM_X, _h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int i = 0; i < nIterations; ++i)
   {
      _updateUVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, alpha, tau, c1, c2, _d_g, _d_u, _d_p1, _d_p2);
      _updatePVolume_kernel<<< gridDim, blockDim, 0 >>>(_w, _h, 1.0f, tau, _d_u, _d_p1, _d_p2);
   }
} // end CUDA_CV_Segmentation::runSegmentation()

void
CUDA_CV_Segmentation::runSegmentation_theta(float c1, float c2, int nIterations, float alpha, float tau, float theta)
{
   dim3 gridDim(_w/DIM_X, _h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int i = 0; i < nIterations; ++i)
   {
      _updateUVolume_theta<<< gridDim, blockDim, 0 >>>(_w, _h, alpha, theta, c1, c2, _d_g, _d_u, _d_p1, _d_p2);
      _updatePVolume_theta<<< gridDim, blockDim, 0 >>>(_w, _h, tau/theta, _d_u, _d_p1, _d_p2);
   }
} // end CUDA_CV_Segmentation::runSegmentation_theta()

void
CUDA_CV_Segmentation::updateMeans(float& c1, float& c2)
{
   dim3 gridDim(_w/DIM_X, _h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nC1Elems = _w/DIM_X * _h/DIM_Y;

   float * c1Array = new float[nC1Elems];
   float * uSumArray = new float[nC1Elems];

   _computeMeans_kernel<<< gridDim, blockDim, 0 >>>(_w, _d_g, _d_u, _d_c1, _d_usum);
   CUDA_SAFE_CALL( cudaMemcpy( c1Array, _d_c1, nC1Elems*sizeof(float), cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( uSumArray, _d_usum, nC1Elems*sizeof(float), cudaMemcpyDeviceToHost) );

   float uSum = 0.0f, c1sum = 0.0f;
   for (int k = 0; k < nC1Elems; ++k) c1sum += c1Array[k];
   for (int k = 0; k < nC1Elems; ++k) uSum += uSumArray[k];
   c1 = c1sum / uSum;
   c2 = (_fSum - c1sum) / (float(_w)*_h - uSum);
   //printf("c1 = %f, c2 = %f\n", c1, c2);

   delete [] c1Array;
   delete [] uSumArray;
} // end cuda_updateMeans()
