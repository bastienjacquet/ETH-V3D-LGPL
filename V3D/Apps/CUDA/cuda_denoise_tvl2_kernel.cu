#include <cuda.h>

#ifdef _WINDOWS
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <cstdio>

#include <algorithm>

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

#define DIM_X 16
#define DIM_Y 16

// namespace
// {

float *d_f, *d_u, *d_p1, *d_p2;

template <int nInnerIterations, int shift>
__global__ void
updateUP_CB(int w, int h, float const * F,
            float tau, float lambda, float tau_lambda, float rcpLambda,
            float * U, float * P1, float * P2)
{
   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
   int const X = __mul24(blockIdx_x, blockDim.x) + threadIdx.x;
   int const Y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

   int const pos = __mul24(Y, w) + X;

   float const tau_p = tau_lambda;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   float const f = F[pos];

   // Load u, p and q
   u_sh[threadIdx.y][threadIdx.x]    = U[pos];
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

   __syncthreads();

   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];
   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
//    if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = u_sh[threadIdx.y][DIM_X-1];
//    if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = u_sh[DIM_Y-1][threadIdx.x];

//    if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = 0;
//    if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = 0;
   if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = (X > 0) ? P1[pos - 1] : 0;
   if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = (Y > 0) ? P2[pos - w] : 0;

   __syncthreads();

   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
      float const u0 = u_sh[threadIdx.y][threadIdx.x];
      float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
      float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

      // Update P and Q

      float const new_p1 = p1_sh[threadIdx.y][threadIdx.x+1] + tau_p * u_x;
      float const new_p2 = p2_sh[threadIdx.y+1][threadIdx.x] + tau_p * u_y;
#if 0
      float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
      float const denom = max(1.0f, tv);
      p1_sh[threadIdx.y][threadIdx.x+1] = new_p1 / denom;
      p2_sh[threadIdx.y+1][threadIdx.x] = new_p2 / denom;
#else
      p1_sh[threadIdx.y][threadIdx.x+1] = max(-1.0f, min(1.0f, new_p1));
      p2_sh[threadIdx.y+1][threadIdx.x] = max(-1.0f, min(1.0f, new_p2));
#endif

      __syncthreads();

      // Update U
      float div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                     p1_sh[threadIdx.y][threadIdx.x] +
                     ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                     p2_sh[threadIdx.y][threadIdx.x]);
//       float div_p = (p1_sh[threadIdx.y][threadIdx.x+1] -
//                      p1_sh[threadIdx.y][threadIdx.x] +
//                      p2_sh[threadIdx.y+1][threadIdx.x] -
//                      p2_sh[threadIdx.y][threadIdx.x]);
      u_sh[threadIdx.y][threadIdx.x] = f + rcpLambda * div_p;

      __syncthreads();
   } // end for (iter)

   U[pos]  = u_sh[threadIdx.y][threadIdx.x];
   P1[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ_CB()

// } // end namespace <>

void
start_tvl2_cuda(int w, int h, float const * f)
{
   int const size = sizeof(float) * w * h;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_f, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_u, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p2, size) );

   CUDA_SAFE_CALL( cudaMemcpy( d_f, f, size, cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL( cudaMemcpy( d_u, f, size, cudaMemcpyHostToDevice) );

   CUDA_SAFE_CALL( cudaMemset( d_p1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( d_p2, 0, size) );
}

void
finish_tvl2_cuda()
{
   CUDA_SAFE_CALL( cudaFree(d_f) );
   CUDA_SAFE_CALL( cudaFree(d_u) );
   CUDA_SAFE_CALL( cudaFree(d_p1) );
   CUDA_SAFE_CALL( cudaFree(d_p2) );
}

void
run_denoise_tvl2_cuda(int w, int h, int nIterations, float tau, float lambda,
                      float * u, float * p1, float * p2)
{
   float const rcpLambda = 1.0f/lambda;

   dim3 gridDim(w/DIM_X/2, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 10;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations/2;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUP_CB<nInnerIterations, 0><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                   tau, lambda, tau*lambda, rcpLambda,
                                                                   d_u, d_p1, d_p2);
      updateUP_CB<nInnerIterations, 1><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                   tau, lambda, tau*lambda, rcpLambda,
                                                                   d_u, d_p1, d_p2);
   }

   int const size = sizeof(float) * w * h;

   CUDA_SAFE_CALL( cudaMemcpy( u, d_u, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p1, d_p1, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p2, d_p2, size, cudaMemcpyDeviceToHost) );
} // end run_denoise_tvl1_cuda()
