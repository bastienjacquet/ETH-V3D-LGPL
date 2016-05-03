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

float *d_f, *d_uA, *d_p1A, *d_p2A, *d_qA, *d_uB, *d_p1B, *d_p2B, *d_qB;

template <int nInnerIterations>
__global__ void
updateUPQ(int w, int h, float const * F,
          float tau, float lambda, float rcpLambda,
          float const * U, float const * P1, float const * P2, float const * Q,
          float * Udst, float * P1dst, float * P2dst, float * Qdst)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   float const f = F[pos];
   float       q = Q[pos];

   // Load u, p and q
   u_sh[threadIdx.y][threadIdx.x]    = U[pos];
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

   __syncthreads();

   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];

   if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = (X > 0) ? P1[pos - 1] : 0;
   if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = (Y > 0) ? P2[pos - w] : 0;

   __syncthreads();

   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
      float const u0 = u_sh[threadIdx.y][threadIdx.x];
      float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
      float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

      // Update P and Q

      float const new_p1 = p1_sh[threadIdx.y][threadIdx.x+1] + tau * u_x;
      float const new_p2 = p2_sh[threadIdx.y+1][threadIdx.x] + tau * u_y;
#if 0
      float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
      float const denom = max(1.0f, tv);
      p1_sh[threadIdx.y][threadIdx.x+1] = new_p1 / denom;
      p2_sh[threadIdx.y+1][threadIdx.x] = new_p2 / denom;
#else
      p1_sh[threadIdx.y][threadIdx.x+1] = max(-1.0f, min(1.0f, new_p1));
      p2_sh[threadIdx.y+1][threadIdx.x] = max(-1.0f, min(1.0f, new_p2));
//          p1_sh[threadIdx.y][threadIdx.x+1] = max(-rcpLambda, min(rcpLambda, new_p1));
//          p2_sh[threadIdx.y+1][threadIdx.x] = max(-rcpLambda, min(rcpLambda, new_p2));
#endif

      __syncthreads();

      // Update U
      float const div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                           p2_sh[threadIdx.y][threadIdx.x]);

#if 0
      float const new_q = q + tau * (u0 - f);
#else
      float const alpha = 0.01f;
      float const new_q = q + tau * (alpha * (u0 - f) + (1.0f - alpha)*(div_p - q));
#endif
      q = max(-lambda, min(lambda, new_q));
      //q = max(-1.0f, min(1.0f, new_q));

      u_sh[threadIdx.y][threadIdx.x] += tau * (div_p - q);
      //u_sh[threadIdx.y][threadIdx.x] += __fmul_rn(tau, (div - q_sh[threadIdx.y][threadIdx.x]));
      //u_sh[threadIdx.y][threadIdx.x] = max(0.0f, min(255.0f, u_sh[threadIdx.y][threadIdx.x]));

      __syncthreads();
   } // end for (iter)

   Udst[pos]  = u_sh[threadIdx.y][threadIdx.x];
   Qdst[pos]  = q;
   P1dst[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2dst[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ()

template <int nInnerIterations>
__global__ void
updateUPQ_sync(int w, int h, float const * F,
               int nOuterIterations, float tau, float lambda, float rcpLambda,
               float const * U, float const * P1, float const * P2, float const * Q,
               float * Udst, float * P1dst, float * P2dst, float * Qdst)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   float const f = F[pos];
   float       q = Q[pos];

   // Load u, p and q
   u_sh[threadIdx.y][threadIdx.x]    = U[pos];
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

   __syncthreads();

   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];

   if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = (X > 0) ? P1[pos - 1] : 0;
   if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = (Y > 0) ? P2[pos - w] : 0;

   __syncthreads();

   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
      float const q0 = q;
      float const u0 = u_sh[threadIdx.y][threadIdx.x];
      float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
      float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

      float const div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                           p2_sh[threadIdx.y][threadIdx.x]);

      // Update P and Q

      float const new_p1 = p1_sh[threadIdx.y][threadIdx.x+1] + tau * u_x;
      float const new_p2 = p2_sh[threadIdx.y+1][threadIdx.x] + tau * u_y;
#if 0
      float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
      float const denom = max(1.0f, tv);
      p1_sh[threadIdx.y][threadIdx.x+1] = new_p1 / denom;
      p2_sh[threadIdx.y+1][threadIdx.x] = new_p2 / denom;
#else
      p1_sh[threadIdx.y][threadIdx.x+1] = max(-1.0f, min(1.0f, new_p1));
      p2_sh[threadIdx.y+1][threadIdx.x] = max(-1.0f, min(1.0f, new_p2));
//          p1_sh[threadIdx.y][threadIdx.x+1] = max(-rcpLambda, min(rcpLambda, new_p1));
//          p2_sh[threadIdx.y+1][threadIdx.x] = max(-rcpLambda, min(rcpLambda, new_p2));
#endif

      // Update U
#if 1
      float const new_q = q + tau * (u0 - f);
#else
      float const alpha = 0.01f;
      float const new_q = q + tau * (alpha * (u0 - f) + (1.0f - alpha)*(div_p - q));
#endif
      q = max(-lambda, min(lambda, new_q));
      //q = max(-1.0f, min(1.0f, new_q));

      u_sh[threadIdx.y][threadIdx.x] += tau * (div_p - q0);

      __syncthreads();
   } // end for (iter)

   Udst[pos]  = u_sh[threadIdx.y][threadIdx.x];
   Qdst[pos]  = q;
   P1dst[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2dst[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ_sync()

__global__ void
updateU(int w, int h, float const * F,
        float tau, float lambda, float rcpLambda,
        float * U, float const * P1, float const * P2, float const * Q)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float       u = U[pos];
   float const q = Q[pos];

   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   // Load p
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

   if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = (X > 0) ? P1[pos - 1] : 0;
   if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = (Y > 0) ? P2[pos - w] : 0;

   __syncthreads();

   float const div = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                      p1_sh[threadIdx.y][threadIdx.x] +
                      ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                      p2_sh[threadIdx.y][threadIdx.x]);
   u += tau * (div - q);
   //u_sh[threadIdx.y][threadIdx.x] += __fmul_rn(tau, (div - q_sh[threadIdx.y][threadIdx.x]));
   //u_sh[threadIdx.y][threadIdx.x] = max(0.0f, min(255.0f, u_sh[threadIdx.y][threadIdx.x]));

   U[pos]  = u;
} // end updateU()

__global__ void
updatePQ(int w, int h, float const * F,
         float tau, float lambda, float rcpLambda,
         float const * U, float * P1, float * P2, float * Q)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float const f = F[pos];
   float       q = Q[pos];
   float       p1 = P1[pos];
   float       p2 = P2[pos];

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];

   // Load u
   u_sh[threadIdx.y][threadIdx.x]  = U[pos];

   __syncthreads();

   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];

   __syncthreads();

   float const u0 = u_sh[threadIdx.y][threadIdx.x];
   float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
   float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

   float const new_p1 = p1 + tau * u_x;
   float const new_p2 = p2 + tau * u_y;
#if 0
   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv);
   p1 = new_p1 / denom;
   p2 = new_p2 / denom;
#else
   p1 = max(-1.0f, min(1.0f, new_p1));
   p2 = max(-1.0f, min(1.0f, new_p2));
//          p1_sh[threadIdx.y][threadIdx.x+1] = max(-rcpLambda, min(rcpLambda, new_p1));
//          p2_sh[threadIdx.y+1][threadIdx.x] = max(-rcpLambda, min(rcpLambda, new_p2));
#endif

   float const new_q = q + tau * (u0 - f);
   q = max(-lambda, min(lambda, new_q));
   //q = max(-1.0f, min(1.0f, new_q));

   Q[pos]  = q;
   P1[pos] = p1;
   P2[pos] = p2;
} // end updatePQ()

__global__ void
updateUQ(int w, int h, float const * F,
         float tau, float lambda, float rcpLambda,
         float * U, float const * P1, float const * P2, float * Q)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float const f = F[pos];
   float u       = U[pos];
   float q       = Q[pos];

   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   // Load p
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

   if (threadIdx.x == 0) p1_sh[threadIdx.y][0] = (X > 0) ? P1[pos - 1] : 0;
   if (threadIdx.y == 0) p2_sh[0][threadIdx.x] = (Y > 0) ? P2[pos - w] : 0;

   __syncthreads();

   float const div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                        p1_sh[threadIdx.y][threadIdx.x] +
                        ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                        p2_sh[threadIdx.y][threadIdx.x]);

   //float const alpha = 0.01f;
   //float const new_q = q + tau * (alpha * (u - f) + (1.0f - alpha)*(div_p - q));
   float const alpha = 0.5f;
   float const new_q = q + tau * (u - f + alpha*(div_p - q));
   q = max(-lambda, min(lambda, new_q));
   //q = max(-1.0f, min(1.0f, new_q));

   u += tau * (div_p - q);

   U[pos] = u;
   Q[pos] = q;
} // end updateUQ()

__global__ void
updateP(int w, int h, float const * F,
        float tau, float lambda, float rcpLambda,
        float const * U, float * P1, float * P2)
{
   int const X = blockIdx.x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float       p1 = P1[pos];
   float       p2 = P2[pos];

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];

   // Load u
   u_sh[threadIdx.y][threadIdx.x]  = U[pos];

   __syncthreads();

   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];

   __syncthreads();

   float const u0 = u_sh[threadIdx.y][threadIdx.x];
   float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
   float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

   float const new_p1 = p1 + tau * u_x;
   float const new_p2 = p2 + tau * u_y;
#if 0
   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv);
   p1 = new_p1 / denom;
   p2 = new_p2 / denom;
#else
   p1 = max(-1.0f, min(1.0f, new_p1));
   p2 = max(-1.0f, min(1.0f, new_p2));
//          p1_sh[threadIdx.y][threadIdx.x+1] = max(-rcpLambda, min(rcpLambda, new_p1));
//          p2_sh[threadIdx.y+1][threadIdx.x] = max(-rcpLambda, min(rcpLambda, new_p2));
#endif

   P1[pos] = p1;
   P2[pos] = p2;
} // end updateP()

template <int nInnerIterations, int shift>
__global__ void
updateUPQ_CB(int w, int h, float const * F,
             float tau, float lambda, float rcpLambda,
             float * U, float * P1, float * P2, float * Q)
{
   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
   int const X = blockIdx_x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   //float const tau_p = tau / nInnerIterations;
   float const tau_p = tau;
   float const tau_q = tau;
   float const tau_u = tau;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   float const f = F[pos];
   float       q = Q[pos];

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
//          p1_sh[threadIdx.y][threadIdx.x+1] = max(-rcpLambda, min(rcpLambda, new_p1));
//          p2_sh[threadIdx.y+1][threadIdx.x] = max(-rcpLambda, min(rcpLambda, new_p2));
#endif

      __syncthreads();

#if 1
      float div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                     p1_sh[threadIdx.y][threadIdx.x] +
                     ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                     p2_sh[threadIdx.y][threadIdx.x]);
#else
      float div_p = (((threadIdx.x < DIM_X-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                     p1_sh[threadIdx.y][threadIdx.x] +
                     ((threadIdx.y < DIM_Y-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                     p2_sh[threadIdx.y][threadIdx.x]);
#endif
      //div_p = max(-lambda, min(lambda, div_p));

#if 0
      float const new_q = q + tau_q * (u0 - f);
#else
      float const alpha = 0.01f;
      float const new_q = q + tau_q * (alpha * (u0 - f) + (1.0f - alpha)*(div_p - q));
//       float const alpha = 0.5f; //0.01f;
//       float const new_q = q + tau_q * ((u0 - f) + alpha*(div_p - q));
#endif
      q = max(-lambda, min(lambda, new_q));
      //q = max(-1.0f, min(1.0f, new_q));

      // Update U
      u_sh[threadIdx.y][threadIdx.x] += tau_u * (div_p - q);

      __syncthreads();
   } // end for (iter)

   U[pos]  = u_sh[threadIdx.y][threadIdx.x];
   Q[pos]  = q;
   P1[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ_CB()

#define MINIMIZE_BRANCHES 1

template <int nInnerIterations, int shift>
__global__ void
updateUPQ_CB_2(int w, int h, float const * F,
               float tau, float lambda, float rcpLambda,
               float * U, float * P1, float * P2, float * Q)
{
   float const beta = 1.0f;

   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
#if 0
   int const X = blockIdx_x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;
   int const pos = Y*w + X;
#else
   int const X = __mul24(blockIdx_x, blockDim.x) + threadIdx.x;
   int const Y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
   int const pos = __mul24(Y, w) + X;
#endif

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float q_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   float const f = F[pos];

   // Load u, p and q
   u_sh[threadIdx.y][threadIdx.x]    = U[pos];
   q_sh[threadIdx.y][threadIdx.x]    = Q[pos];
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

#if defined(MINIMIZE_BRANCHES)
   float const rightBorder  = (X < w-1) ? 1.0f : 0.0f;
   float const bottomBorder = (Y < h-1) ? 1.0f : 0.0f;
//    float const leftBorder   = (X > 0)   ? 1.0f : 0.0f;
//    float const topBorder    = (Y > 0)   ? 1.0f : 0.0f;
#endif

   __syncthreads();

#if !defined(MINIMIZE_BRANCHES)
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];
   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];

   if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = (X < w-1) ? Q[pos + 1] : q_sh[threadIdx.y][DIM_X-1];
   if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? Q[pos + w] : q_sh[DIM_Y-1][threadIdx.x];

   if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = (X > 0)   ? P1[pos - 1] : 0;
   if (threadIdx.x == DIM_X-1) p1_sh[threadIdx.y][DIM_X+1] = (X < w-1) ? P1[pos + 1] : 0;

   if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = (Y > 0)   ? P2[pos - w] : 0;
   if (threadIdx.y == DIM_Y-1) p2_sh[DIM_Y+1][threadIdx.x] = (Y < h-1) ? P2[pos + w] : 0;

   if (threadIdx.y == DIM_Y-1) p1_sh[DIM_Y][threadIdx.x]   = (Y < h-1) ? P1[pos + w] : 0;
   if (threadIdx.x == DIM_X-1) p2_sh[threadIdx.y][DIM_X]   = (X < w-1) ? P2[pos + 1] : 0;
#else
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = U[pos + 1];
   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = U[pos + w];
   if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = Q[pos + 1];
   if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = Q[pos + w];
//    if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];
//    if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
//    if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = (X < w-1) ? Q[pos + 1] : q_sh[threadIdx.y][DIM_X-1];
//    if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? Q[pos + w] : q_sh[DIM_Y-1][threadIdx.x];

//    if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = leftBorder * P1[pos - 1];
   if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = (X > 0)   ? P1[pos - 1] : 0;
   if (threadIdx.x == DIM_X-1) p1_sh[threadIdx.y][DIM_X+1] = rightBorder * P1[pos + 1];

//    if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = topBorder * P2[pos - w];
   if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = (Y > 0)   ? P2[pos - w] : 0;
   if (threadIdx.y == DIM_Y-1) p2_sh[DIM_Y+1][threadIdx.x] = bottomBorder * P2[pos + w];

   if (threadIdx.y == DIM_Y-1) p1_sh[DIM_Y][threadIdx.x]   = bottomBorder * P1[pos + w];
   if (threadIdx.x == DIM_X-1) p2_sh[threadIdx.y][DIM_X]   = rightBorder * P2[pos + 1];
#endif

   __syncthreads();

#pragma unroll
   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
#if !defined(MINIMIZE_BRANCHES)
      float const u0  = u_sh[threadIdx.y][threadIdx.x];
      float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u0;
      float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u0;

      float const q0  = q_sh[threadIdx.y][threadIdx.x];
      float const q_x = q_sh[threadIdx.y][threadIdx.x+1] - q0;
      float const q_y = q_sh[threadIdx.y+1][threadIdx.x] - q0;
#else
      float const u0  = u_sh[threadIdx.y][threadIdx.x];
      float const u_x = rightBorder * (u_sh[threadIdx.y][threadIdx.x+1] - u0);
      float const u_y = bottomBorder * (u_sh[threadIdx.y+1][threadIdx.x] - u0);

      float const q0  = q_sh[threadIdx.y][threadIdx.x];
      float const q_x = rightBorder * (q_sh[threadIdx.y][threadIdx.x+1] - q0);
      float const q_y = bottomBorder * (q_sh[threadIdx.y+1][threadIdx.x] - q0);
#endif

      // Divergence at (x, y)
#if !defined(MINIMIZE_BRANCHES)
      float const div_p_00 = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                              p1_sh[threadIdx.y][threadIdx.x] +
                              ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                              p2_sh[threadIdx.y][threadIdx.x]);

      // Divergence at (x+1, y)
      float const div_p_10 = ((X == w-1) ? div_p_00 :
                              (p1_sh[threadIdx.y][threadIdx.x+2] -
                               p1_sh[threadIdx.y][threadIdx.x+1] +
                               ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x+1] : 0) -
                               p2_sh[threadIdx.y][threadIdx.x+1]));

      // Divergence at (x, y+1)
      float const div_p_01 = ((Y == h-1) ? div_p_00 :
                              ((X < w-1) ? p1_sh[threadIdx.y+1][threadIdx.x+1] : 0) -
                              p1_sh[threadIdx.y+1][threadIdx.x] +
                              p2_sh[threadIdx.y+2][threadIdx.x] -
                              p2_sh[threadIdx.y+1][threadIdx.x]);
#else
      float const div_p_00 = (p1_sh[threadIdx.y][threadIdx.x+1] -
                              p1_sh[threadIdx.y][threadIdx.x] +
                              p2_sh[threadIdx.y+1][threadIdx.x] -
                              p2_sh[threadIdx.y][threadIdx.x]);

# if 1
      // Divergence at (x+1, y)
      float const div_p_10 = ((X == w-1) ? div_p_00 :
                              (p1_sh[threadIdx.y][threadIdx.x+2] -
                               p1_sh[threadIdx.y][threadIdx.x+1] +
                               p2_sh[threadIdx.y+1][threadIdx.x+1] -
                               p2_sh[threadIdx.y][threadIdx.x+1]));

      // Divergence at (x, y+1)
      float const div_p_01 = ((Y == h-1) ? div_p_00 :
                              (p1_sh[threadIdx.y+1][threadIdx.x+1] -
                               p1_sh[threadIdx.y+1][threadIdx.x] +
                               p2_sh[threadIdx.y+2][threadIdx.x] -
                               p2_sh[threadIdx.y+1][threadIdx.x]));
# else
      // Divergence at (x+1, y)
      float const div_p_10 = (p1_sh[threadIdx.y][threadIdx.x+2] -
                              p1_sh[threadIdx.y][threadIdx.x+1] +
                              p2_sh[threadIdx.y+1][threadIdx.x+1] -
                              p2_sh[threadIdx.y][threadIdx.x+1]);

      // Divergence at (x, y+1)
      float const div_p_01 = (p1_sh[threadIdx.y+1][threadIdx.x+1] -
                              p1_sh[threadIdx.y+1][threadIdx.x] +
                              p2_sh[threadIdx.y+2][threadIdx.x] -
                              p2_sh[threadIdx.y+1][threadIdx.x]);
# endif
#endif

#if 1 || !defined(MINIMIZE_BRANCHES)
      float dp1 = u_x - beta * (q_x - div_p_10 + div_p_00);
      float dp2 = u_y - beta * (q_y - div_p_01 + div_p_00);
#else
      float dp1 = u_x - beta * (q_x - rightBorder * (div_p_10 - div_p_00));
      float dp2 = u_y - beta * (q_y - bottomBorder * (div_p_01 - div_p_00));
#endif

      // Update P and Q

      float const new_p1 = p1_sh[threadIdx.y][threadIdx.x+1] + tau * dp1;
      float const new_p2 = p2_sh[threadIdx.y+1][threadIdx.x] + tau * dp2;

      //__syncthreads();

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
#if !defined(MINIMIZE_BRANCHES)
      float const div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                           p2_sh[threadIdx.y][threadIdx.x]);
#elif 1
      float const div_p = (rightBorder * p1_sh[threadIdx.y][threadIdx.x+1] -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           bottomBorder * p2_sh[threadIdx.y+1][threadIdx.x] -
                           p2_sh[threadIdx.y][threadIdx.x]);
#else
      float const div_p = (p1_sh[threadIdx.y][threadIdx.x+1] -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           p2_sh[threadIdx.y+1][threadIdx.x] -
                           p2_sh[threadIdx.y][threadIdx.x]);
#endif

      float new_q = q0 + tau * ((u0 - f) + beta * (div_p - q0));
      new_q = max(-lambda, min(lambda, new_q));

      // Update U
      u_sh[threadIdx.y][threadIdx.x] += tau * (div_p - new_q);
      q_sh[threadIdx.y][threadIdx.x] = new_q;

      __syncthreads();
   } // end for (iter)

   U[pos]  = u_sh[threadIdx.y][threadIdx.x];
   Q[pos]  = q_sh[threadIdx.y][threadIdx.x];
   P1[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ_CB_2()

template <int nInnerIterations, int shift>
__global__ void
updateUPQ_CB_3(int w, int h, float const * F,
               float tau, float lambda, float rcpLambda,
               float * U, float * P1, float * P2, float * Q)
{
   float const beta = 1.0f;

   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
#if 0
   int const X = blockIdx_x * blockDim.x + threadIdx.x;
   int const Y = blockIdx.y * blockDim.y + threadIdx.y;
   int const pos = Y*w + X;
#else
   int const X = __mul24(blockIdx_x, blockDim.x) + threadIdx.x;
   int const Y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
   int const pos = __mul24(Y, w) + X;
#endif

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float q_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   float const f = F[pos];

   // Load u, p and q
   u_sh[threadIdx.y][threadIdx.x]    = U[pos];
   q_sh[threadIdx.y][threadIdx.x]    = Q[pos];
   p1_sh[threadIdx.y][threadIdx.x+1] = P1[pos];
   p2_sh[threadIdx.y+1][threadIdx.x] = P2[pos];

#if defined(MINIMIZE_BRANCHES)
   float const rightBorder  = (X < w-1) ? 1.0f : 0.0f;
   float const bottomBorder = (Y < h-1) ? 1.0f : 0.0f;
//    float const leftBorder   = (X > 0)   ? 1.0f : 0.0f;
//    float const topBorder    = (Y > 0)   ? 1.0f : 0.0f;
#endif

   __syncthreads();

#if !defined(MINIMIZE_BRANCHES)
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];
   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];

   if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = (X < w-1) ? Q[pos + 1] : q_sh[threadIdx.y][DIM_X-1];
   if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? Q[pos + w] : q_sh[DIM_Y-1][threadIdx.x];

   if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = (X > 0)   ? P1[pos - 1] : 0;
   if (threadIdx.x == DIM_X-1) p1_sh[threadIdx.y][DIM_X+1] = (X < w-1) ? P1[pos + 1] : 0;

   if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = (Y > 0)   ? P2[pos - w] : 0;
   if (threadIdx.y == DIM_Y-1) p2_sh[DIM_Y+1][threadIdx.x] = (Y < h-1) ? P2[pos + w] : 0;

   if (threadIdx.y == DIM_Y-1) p1_sh[DIM_Y][threadIdx.x]   = (Y < h-1) ? P1[pos + w] : 0;
   if (threadIdx.x == DIM_X-1) p2_sh[threadIdx.y][DIM_X]   = (X < w-1) ? P2[pos + 1] : 0;
#else
   if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = U[pos + 1];
   if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = U[pos + w];
   if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = Q[pos + 1];
   if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = Q[pos + w];
//    if (threadIdx.x == DIM_X-1) u_sh[threadIdx.y][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[threadIdx.y][DIM_X-1];
//    if (threadIdx.y == DIM_Y-1) u_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][threadIdx.x];
//    if (threadIdx.x == DIM_X-1) q_sh[threadIdx.y][DIM_X] = (X < w-1) ? Q[pos + 1] : q_sh[threadIdx.y][DIM_X-1];
//    if (threadIdx.y == DIM_Y-1) q_sh[DIM_Y][threadIdx.x] = (Y < h-1) ? Q[pos + w] : q_sh[DIM_Y-1][threadIdx.x];

//    if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = leftBorder * P1[pos - 1];
   if (threadIdx.x == 0)       p1_sh[threadIdx.y][0]       = (X > 0)   ? P1[pos - 1] : 0;
   if (threadIdx.x == DIM_X-1) p1_sh[threadIdx.y][DIM_X+1] = rightBorder * P1[pos + 1];

//    if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = topBorder * P2[pos - w];
   if (threadIdx.y == 0)       p2_sh[0][threadIdx.x]       = (Y > 0)   ? P2[pos - w] : 0;
   if (threadIdx.y == DIM_Y-1) p2_sh[DIM_Y+1][threadIdx.x] = bottomBorder * P2[pos + w];

   if (threadIdx.y == DIM_Y-1) p1_sh[DIM_Y][threadIdx.x]   = bottomBorder * P1[pos + w];
   if (threadIdx.x == DIM_X-1) p2_sh[threadIdx.y][DIM_X]   = rightBorder * P2[pos + 1];
#endif

   __syncthreads();

#pragma unroll
   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
#if !defined(MINIMIZE_BRANCHES)
      float const div_p = (((X < w-1) ? p1_sh[threadIdx.y][threadIdx.x+1] : 0) -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x] : 0) -
                           p2_sh[threadIdx.y][threadIdx.x]);
#else
      float const div_p = (rightBorder * p1_sh[threadIdx.y][threadIdx.x+1] -
                           p1_sh[threadIdx.y][threadIdx.x] +
                           bottomBorder * p2_sh[threadIdx.y+1][threadIdx.x] -
                           p2_sh[threadIdx.y][threadIdx.x]);
#endif
      float u = u_sh[threadIdx.y][threadIdx.x];

      // Use u_sh temporarily for div_p
      u_sh[threadIdx.y][threadIdx.x] = div_p;

      float q0  = q_sh[threadIdx.y][threadIdx.x];
      float new_q = q0 + tau * ((u - f) + beta * (div_p - q0));
      new_q = max(-lambda, min(lambda, new_q));

      // Update U
      u += tau * (div_p - new_q);
      q_sh[threadIdx.y][threadIdx.x] = new_q;

      __syncthreads();

      // Divergence at (x, y)
      float const div_p_00 = u_sh[threadIdx.y][threadIdx.x];
      float div_p_10, div_p_01;
#if !defined(MINIMIZE_BRANCHES)
      // Divergence at (x+1, y)
      if (threadIdx.x == DIM_X-1)
      {
         div_p_10 = ((X == w-1) ? div_p_00 :
                     (p1_sh[threadIdx.y][threadIdx.x+2] -
                      p1_sh[threadIdx.y][threadIdx.x+1] +
                      ((Y < h-1) ? p2_sh[threadIdx.y+1][threadIdx.x+1] : 0) -
                      p2_sh[threadIdx.y][threadIdx.x+1]));
      }
      else
         div_p_10 = u_sh[threadIdx.y][threadIdx.x+1];

      if (threadIdx.y == DIM_Y-1)
      {
         // Divergence at (x, y+1)
         div_p_01 = ((Y == h-1) ? div_p_00 :
                     ((X < w-1) ? p1_sh[threadIdx.y+1][threadIdx.x+1] : 0) -
                     p1_sh[threadIdx.y+1][threadIdx.x] +
                     p2_sh[threadIdx.y+2][threadIdx.x] -
                     p2_sh[threadIdx.y+1][threadIdx.x]);
      }
      else
         div_p_01 = u_sh[threadIdx.y+1][threadIdx.x];
#else
      // Divergence at (x+1, y)
      if (threadIdx.x == DIM_X-1)
      {
         div_p_10 = ((X == w-1) ? div_p_00 :
                     (p1_sh[threadIdx.y][threadIdx.x+2] -
                      p1_sh[threadIdx.y][threadIdx.x+1] +
                      p2_sh[threadIdx.y+1][threadIdx.x+1] -
                      p2_sh[threadIdx.y][threadIdx.x+1]));
      }
      else
         div_p_10 = u_sh[threadIdx.y][threadIdx.x+1];

      if (threadIdx.y == DIM_Y-1)
      {
         // Divergence at (x, y+1)
         div_p_01 = ((Y == h-1) ? div_p_00 :
                     (p1_sh[threadIdx.y+1][threadIdx.x+1] -
                      p1_sh[threadIdx.y+1][threadIdx.x] +
                      p2_sh[threadIdx.y+2][threadIdx.x] -
                      p2_sh[threadIdx.y+1][threadIdx.x]));
      }
      else
         div_p_01 = u_sh[threadIdx.y+1][threadIdx.x];
#endif
      __syncthreads();

      u_sh[threadIdx.y][threadIdx.x] = u;
      __syncthreads();

#if !defined(MINIMIZE_BRANCHES)
      float const u_x = u_sh[threadIdx.y][threadIdx.x+1] - u;
      float const u_y = u_sh[threadIdx.y+1][threadIdx.x] - u;

      float const q_x = q_sh[threadIdx.y][threadIdx.x+1] - q0;
      float const q_y = q_sh[threadIdx.y+1][threadIdx.x] - q0;
#else
      float const u_x = rightBorder * (u_sh[threadIdx.y][threadIdx.x+1] - u);
      float const u_y = bottomBorder * (u_sh[threadIdx.y+1][threadIdx.x] - u);

      q0  = q_sh[threadIdx.y][threadIdx.x];
      float const q_x = rightBorder * (q_sh[threadIdx.y][threadIdx.x+1] - q0);
      float const q_y = bottomBorder * (q_sh[threadIdx.y+1][threadIdx.x] - q0);
#endif

#if !defined(MINIMIZE_BRANCHES)
      float dp1 = u_x - beta * (q_x - div_p_10 + div_p_00);
      float dp2 = u_y - beta * (q_y - div_p_01 + div_p_00);
#else
      float dp1 = u_x - beta * (q_x - rightBorder * (div_p_10 - div_p_00));
      float dp2 = u_y - beta * (q_y - bottomBorder * (div_p_01 - div_p_00));
#endif

      // Update P
      float const new_p1 = p1_sh[threadIdx.y][threadIdx.x+1] + tau * dp1;
      float const new_p2 = p2_sh[threadIdx.y+1][threadIdx.x] + tau * dp2;

      //__syncthreads();

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
   } // end for (iter)

   U[pos]  = u_sh[threadIdx.y][threadIdx.x];
   Q[pos]  = q_sh[threadIdx.y][threadIdx.x];
   P1[pos] = p1_sh[threadIdx.y][threadIdx.x+1];
   P2[pos] = p2_sh[threadIdx.y+1][threadIdx.x];
} // end updateUPQ_CB_3()

// } // end namespace <>

void
start_tvl1_cuda(int w, int h, float const * f)
{
   int const size = sizeof(float) * w * h;
   int const sizeU = size + w;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_f, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uA, sizeU) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p1A, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p2A, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_qA, sizeU) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_uB, sizeU) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p1B, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p2B, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_qB, sizeU) );

   CUDA_SAFE_CALL( cudaMemset( d_uA, 0, sizeU) );
   CUDA_SAFE_CALL( cudaMemset( d_uB, 0, sizeU) );
   CUDA_SAFE_CALL( cudaMemset( d_qA, 0, sizeU) );
   CUDA_SAFE_CALL( cudaMemset( d_qB, 0, sizeU) );

   CUDA_SAFE_CALL( cudaMemcpy( d_f, f, size, cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL( cudaMemcpy( d_uA, f, size, cudaMemcpyHostToDevice) );

   CUDA_SAFE_CALL( cudaMemset( d_p1A, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( d_p2A, 0, size) );
}

void
finish_tvl1_cuda()
{
   CUDA_SAFE_CALL( cudaFree(d_f) );
   CUDA_SAFE_CALL( cudaFree(d_uA) );
   CUDA_SAFE_CALL( cudaFree(d_p1A) );
   CUDA_SAFE_CALL( cudaFree(d_p2A) );
   CUDA_SAFE_CALL( cudaFree(d_qA) );
   CUDA_SAFE_CALL( cudaFree(d_uB) );
   CUDA_SAFE_CALL( cudaFree(d_p1B) );
   CUDA_SAFE_CALL( cudaFree(d_p2B) );
   CUDA_SAFE_CALL( cudaFree(d_qB) );
}

void
run_denoise_tvl1_cuda(int w, int h, int nIterations, float tau, float lambda,
                      float * u, float * p1, float * p2, float * q)
{
   float const rcpLambda = 1.0f/lambda;

#if 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 2;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ<nInnerIterations><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                              tau, lambda, rcpLambda,
                                                              d_uA, d_p1A, d_p2A, d_qA,
                                                              d_uB, d_p1B, d_p2B, d_qB);
      std::swap(d_uA, d_uB);
      std::swap(d_p1A, d_p1B);
      std::swap(d_p2A, d_p2B);
      std::swap(d_qA, d_qB);
   }
#elif 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 5;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ_sync<nInnerIterations><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                   nOuterIterations, tau, lambda, rcpLambda,
                                                                   d_uA, d_p1A, d_p2A, d_qA,
                                                                   d_uB, d_p1B, d_p2B, d_qB);
      std::swap(d_uA, d_uB);
      std::swap(d_p1A, d_p1B);
      std::swap(d_p2A, d_p2B);
      std::swap(d_qA, d_qB);
   }
#elif 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int iter = 0; iter < nIterations; ++iter)
   {
      updateU<<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                          tau, lambda, rcpLambda,
                                          d_uA, d_p1A, d_p2A, d_qA);
      updatePQ<<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                           tau, lambda, rcpLambda,
                                           d_uA, d_p1A, d_p2A, d_qA);
   }
#elif 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int iter = 0; iter < nIterations; ++iter)
   {
      updateUQ<<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                           tau, lambda, rcpLambda,
                                           d_uA, d_p1A, d_p2A, d_qA);
      updateP<<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                          tau, lambda, rcpLambda,
                                          d_uA, d_p1A, d_p2A);
   }
#elif 0
   dim3 gridDim(w/DIM_X/2, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 5;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations/2;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ_CB<nInnerIterations, 0><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                    tau, lambda, rcpLambda,
                                                                    d_uA, d_p1A, d_p2A, d_qA);
      updateUPQ_CB<nInnerIterations, 1><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                    tau, lambda, rcpLambda,
                                                                    d_uA, d_p1A, d_p2A, d_qA);
   }
#else
   dim3 gridDim(w/DIM_X/2, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 10;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations/2;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ_CB_2<nInnerIterations, 0><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                      tau, lambda, rcpLambda,
                                                                      d_uA, d_p1A, d_p2A, d_qA);
      updateUPQ_CB_2<nInnerIterations, 1><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                      tau, lambda, rcpLambda,
                                                                      d_uA, d_p1A, d_p2A, d_qA);
   }
#endif

   int const size = sizeof(float) * w * h;

   CUDA_SAFE_CALL( cudaMemcpy( u, d_uA, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p1, d_p1A, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p2, d_p2A, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( q, d_qA, size, cudaMemcpyDeviceToHost) );
} // end run_denoise_tvl1_cuda()
