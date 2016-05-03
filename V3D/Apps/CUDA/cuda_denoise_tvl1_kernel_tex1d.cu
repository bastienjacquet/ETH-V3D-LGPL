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

float *d_f, *d_u, *d_p1, *d_p2, *d_q;
float *d_U, *d_P1, *d_P2, *d_Q;

texture<float, 1, cudaReadModeElementType> u_tex;
texture<float, 1, cudaReadModeElementType> p1_tex;
texture<float, 1, cudaReadModeElementType> p2_tex;
texture<float, 1, cudaReadModeElementType> q_tex;

#define USE_SYNC_VERSION 1
#define MINIMIZE_BRANCHES 1

template <int nInnerIterations, int shift>
__global__ void
updateUPQ_CB_2(int w, int h, float const * F,
               float tau, float lambda, float const maxU,
               float * U, float * P1, float * P2, float * Q)
{
   // Time step 0.22
//    float const beta_p = 1.0f;
//    float const beta_q = 1.0f;
   // Timestep 0.2
   float const beta_p = 0.5f;
   float const beta_q = 0.5f;

//    float const beta_p = 0.1f;
//    float const beta_q = 0.1f;

   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
#if 0
   int const X = blockIdx_x * blockDim.x + tidx;
   int const Y = blockIdx.y * blockDim.y + tidy;
   int const pos = Y*w + X;
#else
   int const X = __mul24(blockIdx_x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;
#endif

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float q_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   float const f = F[pos];

   // Load u, p and q
   u_sh[tidy][tidx]    = tex1Dfetch(u_tex, pos);
   q_sh[tidy][tidx]    = tex1Dfetch(q_tex, pos);
   p1_sh[tidy][tidx+1] = tex1Dfetch(p1_tex, pos);
   p2_sh[tidy+1][tidx] = tex1Dfetch(p2_tex, pos);

#if defined(MINIMIZE_BRANCHES)
   float const rightBorder  = ((tidx < DIM_X-1) || (X < w-1)) ? 1.0f : 0.0f;
   float const bottomBorder = ((tidy < DIM_Y-1) || (Y < h-1)) ? 1.0f : 0.0f;
//    float const rightBorder  = 1.0f;
//    float const bottomBorder = 1.0f;
//    float const leftBorder   = (X > 0)   ? 1.0f : 0.0f;
//    float const topBorder    = (Y > 0)   ? 1.0f : 0.0f;
#endif

   //__syncthreads();

#if !defined(MINIMIZE_BRANCHES)
   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? tex1Dfetch(u_tex, pos+1) : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? tex1Dfetch(u_tex, pos+w) : u_sh[DIM_Y-1][tidx];

   if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? tex1Dfetch(q_tex, pos+1) : q_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? tex1Dfetch(q_tex, pos+w) : q_sh[DIM_Y-1][tidx];

   if (tidx == 0)       p1_sh[tidy][0]       = (X > 0)   ? tex1Dfetch(p1_tex, pos-1) : 0;
   if (tidx == DIM_X-1) p1_sh[tidy][DIM_X+1] = (X < w-1) ? tex1Dfetch(p1_tex, pos+1) : 0;

   if (tidy == 0)       p2_sh[0][tidx]       = (Y > 0)   ? tex1Dfetch(p2_tex, pos-w) : 0;
   if (tidy == DIM_Y-1) p2_sh[DIM_Y+1][tidx] = (Y < h-1) ? tex1Dfetch(p2_tex, pos+w) : 0;

   if (tidy == DIM_Y-1) p1_sh[DIM_Y][tidx]   = (Y < h-1) ? tex1Dfetch(p1_tex, pos+w) : 0;
   if (tidx == DIM_X-1) p2_sh[tidy][DIM_X]   = (X < w-1) ? tex1Dfetch(p2_tex, pos+1) : 0;
#else
   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = tex1Dfetch(u_tex, pos+1);
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = tex1Dfetch(u_tex, pos+w);
   if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = tex1Dfetch(q_tex, pos+1);
   if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = tex1Dfetch(q_tex, pos+w);
//    if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos + 1] : u_sh[tidy][DIM_X-1];
//    if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos + w] : u_sh[DIM_Y-1][tidx];
//    if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? Q[pos + 1] : q_sh[tidy][DIM_X-1];
//    if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? Q[pos + w] : q_sh[DIM_Y-1][tidx];

//    if (tidx == 0)       p1_sh[tidy][0]       = leftBorder * P1[pos - 1];
   if (tidx == 0)       p1_sh[tidy][0]       = (X > 0)   ? tex1Dfetch(p1_tex, pos-1) : 0;
   //if (tidx == 0)       p1_sh[tidy][0]       = tex1Dfetch(p1_tex, pos-1);
   if (tidx == DIM_X-1) p1_sh[tidy][DIM_X+1] = rightBorder * tex1Dfetch(p1_tex, pos+1);

//    if (tidy == 0)       p2_sh[0][tidx]       = topBorder * P2[pos - w];
   //if (tidy == 0)       p2_sh[0][tidx]       = (Y > 0)   ? tex1Dfetch(p2_tex, pos-w) : 0;
   if (tidy == 0)       p2_sh[0][tidx]       = tex1Dfetch(p2_tex, pos-w);
   if (tidy == DIM_Y-1) p2_sh[DIM_Y+1][tidx] = bottomBorder * tex1Dfetch(p2_tex, pos+w);

   if (tidy == DIM_Y-1) p1_sh[DIM_Y][tidx]   = bottomBorder * tex1Dfetch(p1_tex, pos+w);
   if (tidx == DIM_X-1) p2_sh[tidy][DIM_X]   = rightBorder * tex1Dfetch(p2_tex, pos+1);
#endif

   __syncthreads();

#pragma unroll
   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
#if !defined(MINIMIZE_BRANCHES)
      float const u0  = u_sh[tidy][tidx];
      float const u_x = u_sh[tidy][tidx+1] - u0;
      float const u_y = u_sh[tidy+1][tidx] - u0;

      float const q0  = q_sh[tidy][tidx];
      float const q_x = q_sh[tidy][tidx+1] - q0;
      float const q_y = q_sh[tidy+1][tidx] - q0;
#else
      float const u0  = u_sh[tidy][tidx];
      float const u_x = rightBorder * (u_sh[tidy][tidx+1] - u0);
      float const u_y = bottomBorder * (u_sh[tidy+1][tidx] - u0);

      float const q0  = q_sh[tidy][tidx];
      float const q_x = rightBorder * (q_sh[tidy][tidx+1] - q0);
      float const q_y = bottomBorder * (q_sh[tidy+1][tidx] - q0);
#endif

      // Divergence at (x, y)
#if !defined(MINIMIZE_BRANCHES)
      float const div_p_00 = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                              p1_sh[tidy][tidx] +
                              ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                              p2_sh[tidy][tidx]);

      // Divergence at (x+1, y)
      float const div_p_10 = ((X == w-1) ? div_p_00 :
                              (p1_sh[tidy][tidx+2] -
                               p1_sh[tidy][tidx+1] +
                               ((Y < h-1) ? p2_sh[tidy+1][tidx+1] : 0) -
                               p2_sh[tidy][tidx+1]));

      // Divergence at (x, y+1)
      float const div_p_01 = ((Y == h-1) ? div_p_00 :
                              ((X < w-1) ? p1_sh[tidy+1][tidx+1] : 0) -
                              p1_sh[tidy+1][tidx] +
                              p2_sh[tidy+2][tidx] -
                              p2_sh[tidy+1][tidx]);
#else
      float const div_p_00 = (p1_sh[tidy][tidx+1] -
                              p1_sh[tidy][tidx] +
                              p2_sh[tidy+1][tidx] -
                              p2_sh[tidy][tidx]);

# if 1
      // Divergence at (x+1, y)
      float const div_p_10 = ((X == w-1) ? div_p_00 :
                              (p1_sh[tidy][tidx+2] -
                               p1_sh[tidy][tidx+1] +
                               p2_sh[tidy+1][tidx+1] -
                               p2_sh[tidy][tidx+1]));

      // Divergence at (x, y+1)
      float const div_p_01 = ((Y == h-1) ? div_p_00 :
                              (p1_sh[tidy+1][tidx+1] -
                               p1_sh[tidy+1][tidx] +
                               p2_sh[tidy+2][tidx] -
                               p2_sh[tidy+1][tidx]));
# else
      // Divergence at (x+1, y)
      float const div_p_10 = (p1_sh[tidy][tidx+2] -
                              p1_sh[tidy][tidx+1] +
                              p2_sh[tidy+1][tidx+1] -
                              p2_sh[tidy][tidx+1]);

      // Divergence at (x, y+1)
      float const div_p_01 = (p1_sh[tidy+1][tidx+1] -
                              p1_sh[tidy+1][tidx] +
                              p2_sh[tidy+2][tidx] -
                              p2_sh[tidy+1][tidx]);
# endif
#endif

#if 1 || !defined(MINIMIZE_BRANCHES)
      float dp1 = u_x - beta_p * (q_x - div_p_10 + div_p_00);
      float dp2 = u_y - beta_p * (q_y - div_p_01 + div_p_00);
#else
      float dp1 = u_x - beta_p * (q_x - rightBorder * (div_p_10 - div_p_00));
      float dp2 = u_y - beta_p * (q_y - bottomBorder * (div_p_01 - div_p_00));
#endif

      // Update P and Q

      float const new_p1 = p1_sh[tidy][tidx+1] + tau * dp1;
      float const new_p2 = p2_sh[tidy+1][tidx] + tau * dp2;

      __syncthreads();

#if 1
      float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
      float const denom = max(1.0f, tv);
      p1_sh[tidy][tidx+1] = new_p1 / denom;
      p2_sh[tidy+1][tidx] = new_p2 / denom;
#else
      p1_sh[tidy][tidx+1] = max(-1.0f, min(1.0f, new_p1));
      p2_sh[tidy+1][tidx] = max(-1.0f, min(1.0f, new_p2));
#endif

      __syncthreads();
#if !defined(MINIMIZE_BRANCHES)
      float const div_p = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                           p1_sh[tidy][tidx] +
                           ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                           p2_sh[tidy][tidx]);
#elif 0
      float const div_p = (rightBorder * p1_sh[tidy][tidx+1] -
                           p1_sh[tidy][tidx] +
                           bottomBorder * p2_sh[tidy+1][tidx] -
                           p2_sh[tidy][tidx]);
#else
      float const div_p = (p1_sh[tidy][tidx+1] -
                           p1_sh[tidy][tidx] +
                           p2_sh[tidy+1][tidx] -
                           p2_sh[tidy][tidx]);
#endif

      float new_q = q0 + tau * ((u0 - f) + beta_q * (div_p - q0));
      new_q = max(-lambda, min(lambda, new_q));

      // Update U
      float const new_u = u0 + tau * (div_p - new_q);
      u_sh[tidy][tidx] = max(0.0f, min(maxU, new_u));
      q_sh[tidy][tidx] = new_q;

      __syncthreads();
   } // end for (iter)

   U[pos]  = u_sh[tidy][tidx];
   Q[pos]  = q_sh[tidy][tidx];
   P1[pos] = p1_sh[tidy][tidx+1];
   P2[pos] = p2_sh[tidy+1][tidx];
} // end updateUPQ_CB_2()


__global__ void
updateU_sync(int w, int h, float tau, float const maxU,
             float const * U, float const * P1, float const * P2, float const* Q,
             float * Udst)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = blockIdx.x * blockDim.x + tidx;
   int const Y = blockIdx.y * blockDim.y + tidy;
   int const pos = Y*w + X;

   __shared__ float p1_sh[DIM_Y][DIM_X+1];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   p1_sh[tidy][tidx+1] = P1[pos];
   p2_sh[tidy+1][tidx] = P2[pos];

   if (tidx == 0) p1_sh[tidy][0] = (X > 0) ? P1[pos-1] : 0;
   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? P2[pos-w] : 0;

   float const q  = Q[pos];
   float const u0 = U[pos];

   __syncthreads();

   float const div_p = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                        p1_sh[tidy][tidx] +
                        ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                        p2_sh[tidy][tidx]);

   float const new_u = u0 + tau * (div_p - q);
   Udst[pos] = max(0.0f, min(maxU, new_u));
} // end updateU_sync()

__global__ void
updatePQ_sync(int w, int h, float const * F,
              float tau, float lambda,
              float const * U, float const * P1, float const * P2, float const * Q,
              float * P1dst, float * P2dst, float * Qdst)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = blockIdx.x * blockDim.x + tidx;
   int const Y = blockIdx.y * blockDim.y + tidy;
   int const pos = Y*w + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];

   u_sh[tidy][tidx] = U[pos];

   __syncthreads();

   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

   __syncthreads();

   float const u0  = u_sh[tidy][tidx];
   float const u_x = u_sh[tidy][tidx+1] - u0;
   float const u_y = u_sh[tidy+1][tidx] - u0;

   float const new_p1 = P1[pos] + tau * u_x;
   float const new_p2 = P2[pos] + tau * u_y;

#if 1
   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv);
   //float const denom = max(1.0f, tv * lambda);
   P1dst[pos] = new_p1 / denom;
   P2dst[pos] = new_p2 / denom;
#else
   P1dst[pos] = max(-1.0f, min(1.0f, new_p1));
   P2dst[pos] = max(-1.0f, min(1.0f, new_p2));
#endif

   float new_q = Q[pos] + tau * (u0 - F[pos]);
   Qdst[pos] = max(-lambda, min(lambda, new_q));
   //Qdst[pos] = max(-1.0f, min(1.0f, new_q));
} // end updatePQ_sync()

// __constant__ float const beta_sync2 = 0.0f;
__constant__ float const beta_sync2 = 0.1f;
// __constant__ float const beta_sync2 = 1.0f;

__global__ void
updateUQ_sync2(int w, int h, float const * F,
               float tau, float lambda, float const maxU,
               float const * U, float const * P1, float const * P2, float const * Q,
               float * Udst, float * Qdst)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;

   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   float const f = F[pos];
   float const q0 = Q[pos];
   float const u0 = U[pos];

   // Load p
   p1_sh[tidy][tidx+1] = P1[pos];
   p2_sh[tidy+1][tidx] = P2[pos];

   if (tidx == 0)       p1_sh[tidy][0]       = (X > 0)   ? P1[pos-1] : 0;
   if (tidx == DIM_X-1) p1_sh[tidy][DIM_X+1] = (X < w-1) ? P1[pos+1] : 0;

   if (tidy == 0)       p2_sh[0][tidx]       = (Y > 0)   ? P2[pos-w] : 0;
   if (tidy == DIM_Y-1) p2_sh[DIM_Y+1][tidx] = (Y < h-1) ? P2[pos+w] : 0;

   if (tidy == DIM_Y-1) p1_sh[DIM_Y][tidx]   = (Y < h-1) ? P1[pos+w] : 0;
   if (tidx == DIM_X-1) p2_sh[tidy][DIM_X]   = (X < w-1) ? P2[pos+1] : 0;

   __syncthreads();

   float const div_p = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                        p1_sh[tidy][tidx] +
                        ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                        p2_sh[tidy][tidx]);

   float new_q = q0 + tau * ((u0 - f) + beta_sync2 * (div_p - q0));
   new_q = max(-lambda, min(lambda, new_q));
   //new_q = max(-1.0f, min(1.0f, new_q));

   // Update U
   float new_u = u0 + tau * (div_p - new_q);
   new_u = max(0.0f, min(maxU, new_u));

   Udst[pos] = new_u;
   Qdst[pos] = new_q;
} // end updateUQ_sync2()

#if 1
__global__ void
updateP_sync2(int w, int h, float tau, float lambda,
              float const * U, float const * P1, float const * P2, float const * Q,
              float * P1dst, float * P2dst)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float q_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   // Load u, p and q
   u_sh[tidy][tidx]    = U[pos];
   q_sh[tidy][tidx]    = Q[pos];
   p1_sh[tidy][tidx+1] = P1[pos];
   p2_sh[tidy+1][tidx] = P2[pos];

   //__syncthreads();

   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

   if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? Q[pos+1] : q_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? Q[pos+w] : q_sh[DIM_Y-1][tidx];

   if (tidx == 0)       p1_sh[tidy][0]       = (X > 0)   ? P1[pos-1] : 0;
   if (tidx == DIM_X-1) p1_sh[tidy][DIM_X+1] = (X < w-1) ? P1[pos+1] : 0;

   if (tidy == 0)       p2_sh[0][tidx]       = (Y > 0)   ? P2[pos-w] : 0;
   if (tidy == DIM_Y-1) p2_sh[DIM_Y+1][tidx] = (Y < h-1) ? P2[pos+w] : 0;

   if (tidy == DIM_Y-1) p1_sh[DIM_Y][tidx]   = (Y < h-1) ? P1[pos+w] : 0;
   if (tidx == DIM_X-1) p2_sh[tidy][DIM_X]   = (X < w-1) ? P2[pos+1] : 0;

   __syncthreads();

   float const u0  = u_sh[tidy][tidx];
   float const u_x = u_sh[tidy][tidx+1] - u0;
   float const u_y = u_sh[tidy+1][tidx] - u0;

   float const q0  = q_sh[tidy][tidx];
   float const q_x = q_sh[tidy][tidx+1] - q0;
   float const q_y = q_sh[tidy+1][tidx] - q0;

   // Divergence at (x, y)
   float const div_p_00 = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                           p1_sh[tidy][tidx] +
                           ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                           p2_sh[tidy][tidx]);

   // Divergence at (x+1, y)
   float const div_p_10 = ((X == w-1) ? div_p_00 :
                           (p1_sh[tidy][tidx+2] -
                            p1_sh[tidy][tidx+1] +
                            ((Y < h-1) ? p2_sh[tidy+1][tidx+1] : 0) -
                            p2_sh[tidy][tidx+1]));

   // Divergence at (x, y+1)
   float const div_p_01 = ((Y == h-1) ? div_p_00 :
                           ((X < w-1) ? p1_sh[tidy+1][tidx+1] : 0) -
                           p1_sh[tidy+1][tidx] +
                           p2_sh[tidy+2][tidx] -
                           p2_sh[tidy+1][tidx]);
   float dp1 = u_x - beta_sync2 * (q_x - div_p_10 + div_p_00);
   float dp2 = u_y - beta_sync2 * (q_y - div_p_01 + div_p_00);

   // Update P and Q

   float new_p1 = p1_sh[tidy][tidx+1] + tau * dp1;
   float new_p2 = p2_sh[tidy+1][tidx] + tau * dp2;

   //__syncthreads();

#if 1
   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv);
   //float const denom = max(1.0f, tv * lambda);
   new_p1 = new_p1 / denom;
   new_p2 = new_p2 / denom;
#else
   new_p1 = max(-1.0f, min(1.0f, new_p1));
   new_p2 = max(-1.0f, min(1.0f, new_p2));
#endif

   P1dst[pos] = new_p1;
   P2dst[pos] = new_p2;
} // end updateP_sync2()
#else
__global__ void
updateP_sync2(int w, int h, float tau, float lambda,
              float const * U, float const * P1, float const * P2, float const * Q,
              float * P1dst, float * P2dst)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X];
   __shared__ float q_sh[DIM_Y+1][DIM_X];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   // Load u, p and q
   u_sh[tidy][tidx]    = U[pos];
   q_sh[tidy][tidx]    = Q[pos];
   p1_sh[tidy][tidx+1] = P1[pos];
   p2_sh[tidy+1][tidx] = P2[pos];

   //__syncthreads();

   //if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

   //if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? Q[pos+1] : q_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? Q[pos+w] : q_sh[DIM_Y-1][tidx];

   if (tidx == 0)       p1_sh[tidy][0]       = (X > 0)   ? P1[pos-1] : 0;
   if (tidx == DIM_X-1) p1_sh[tidy][DIM_X+1] = (X < w-1) ? P1[pos+1] : 0;

   if (tidy == 0)       p2_sh[0][tidx]       = (Y > 0)   ? P2[pos-w] : 0;
   if (tidy == DIM_Y-1) p2_sh[DIM_Y+1][tidx] = (Y < h-1) ? P2[pos+w] : 0;

   if (tidy == DIM_Y-1) p1_sh[DIM_Y][tidx]   = (Y < h-1) ? P1[pos+w] : 0;
   if (tidx == DIM_X-1) p2_sh[tidy][DIM_X]   = (X < w-1) ? P2[pos+1] : 0;

   __syncthreads();

   float u1 = (tidx < DIM_X-1) ? u_sh[tidy][tidx+1] : u_sh[tidy][tidx];
   u1 = (X < w-1) ? U[pos+1] : u1;
   float const u_x = u1 - u_sh[tidy][tidx];
   float const u_y = u_sh[tidy+1][tidx] - u_sh[tidy][tidx];

   float q1 = (tidx < DIM_X-1) ? q_sh[tidy][tidx+1] : q_sh[tidy][tidx];
   q1 = (X < w-1) ? Q[pos+1] : q1;
   float const q_x = q1 - q_sh[tidy][tidx];
   float const q_y = q_sh[tidy+1][tidx] - q_sh[tidy][tidx];

   // Divergence at (x, y)
   float const div_p_00 = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
                           p1_sh[tidy][tidx] +
                           ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
                           p2_sh[tidy][tidx]);

   // Divergence at (x+1, y)
   float const div_p_10 = ((X == w-1) ? div_p_00 :
                           (p1_sh[tidy][tidx+2] -
                            p1_sh[tidy][tidx+1] +
                            ((Y < h-1) ? p2_sh[tidy+1][tidx+1] : 0) -
                            p2_sh[tidy][tidx+1]));

   // Divergence at (x, y+1)
   float const div_p_01 = ((Y == h-1) ? div_p_00 :
                           ((X < w-1) ? p1_sh[tidy+1][tidx+1] : 0) -
                           p1_sh[tidy+1][tidx] +
                           p2_sh[tidy+2][tidx] -
                           p2_sh[tidy+1][tidx]);
   float dp1 = u_x - beta_sync2 * (q_x - div_p_10 + div_p_00);
   float dp2 = u_y - beta_sync2 * (q_y - div_p_01 + div_p_00);

   // Update P and Q

   float new_p1 = p1_sh[tidy][tidx+1] + tau * dp1;
   float new_p2 = p2_sh[tidy+1][tidx] + tau * dp2;

#if 1
   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv);
   //float const denom = max(1.0f, tv * lambda);
   new_p1 = new_p1 / denom;
   new_p2 = new_p2 / denom;
#else
   new_p1 = max(-1.0f, min(1.0f, new_p1));
   new_p2 = max(-1.0f, min(1.0f, new_p2));
#endif

   P1dst[pos] = new_p1;
   P2dst[pos] = new_p2;
} // end updateP_sync2()
#endif

//**********************************************************************

__constant__ float const theta = 8*0.1f;

__global__ void
updateUVolume_theta(int w, int h, float alpha, float maxU, float const * F,
                    float * U, float const * P1, float const * P2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y][DIM_X];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   p1_sh[tidy][tidx]   = P1[curPos];
   p2_sh[tidy+1][tidx] = P2[curPos];

   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? P2[curPos-w] : 0.0f;

   float u = U[curPos];
   float const f = F[curPos];

   float const d = f - u;
   float v = (d >= alpha*theta) ? (u+alpha*theta) : ((d <= -alpha*theta) ? (u-alpha*theta) : f);

   v = max(0.0f, min(maxU, v));

   __syncthreads();

   float p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? P1[curPos - 1] : p1_0;
   float p2_0 = p2_sh[tidy][tidx];
   float div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
                  ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   u = v + theta * div_p;

   U[curPos] = u;
} // end updateUVolume_theta()

__global__ void
updatePVolume_theta(int w, int h, float alpha, float tau, float * const U, float * P1, float * P2)
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
   __syncthreads();

   // Load p and q of current slice/disparity
   float const p1_cur = P1[pos];
   float const p2_cur = P2[pos];

   float       u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float const u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1              = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float const u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

# if 1
   float new_p1 = p1_cur + tau * u_x;
   float new_p2 = p2_cur + tau * u_y;

   //float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha);
   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2));
   new_p1 /= norm;
   new_p2 /= norm;
# else
   float const tv = sqrtf(u_x*u_x + u_y*u_y);
   float const denom_p = 1.0f / (1.0f + tau * tv);
   float new_p1 = (p1_cur - tau * u_x) * denom_p;
   float new_p2 = (p2_cur - tau * u_y) * denom_p;
#endif

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end updatePVolume_theta()

// __global__ void
// updateQVolume_theta(int w, int h, float const * F, float const * U, float * Q)
// {
//    int const tidx = threadIdx.x;
//    int const tidy = threadIdx.y;

//    int const ix = __mul24(tidy, DIM_X) + tidx;

//    int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
//    int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
//    //int const pos = __mul24(Y, w) + X;
//    int pos = Y*w + X;

//    float const f = F[pos];
//    float const u = U[pos];

//    Q[pos] = (u >= f) ? 1.0f : -1.0f;
// } // end updateQVolume_theta()

//**********************************************************************

// } // end namespace <>

void
start_tvl1_cuda(int w, int h, float const * f)
{
   int const size = sizeof(float) * w * h;
   int const sizeU = size;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_f, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_u, sizeU) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_p2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_q, sizeU) );

#if defined(USE_SYNC_VERSION)
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_U, sizeU) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_P1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_P2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Q, sizeU) );
#endif

   CUDA_SAFE_CALL( cudaMemset( d_u, 0, sizeU) );
   CUDA_SAFE_CALL( cudaMemset( d_q, 0, sizeU) );

   CUDA_SAFE_CALL( cudaMemcpy( d_f, f, size, cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL( cudaMemcpy( d_u, f, size, cudaMemcpyHostToDevice) );

   CUDA_SAFE_CALL( cudaMemset( d_p1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( d_p2, 0, size) );

#if !defined(USE_SYNC_VERSION)
   CUDA_SAFE_CALL( cudaBindTexture(0, u_tex, d_u, sizeU) );
   CUDA_SAFE_CALL( cudaBindTexture(0, q_tex, d_q, sizeU) );
   CUDA_SAFE_CALL( cudaBindTexture(0, p1_tex, d_p1, size) );
   CUDA_SAFE_CALL( cudaBindTexture(0, p2_tex, d_p2, size) );
#endif
}

void
finish_tvl1_cuda()
{
#if !defined(USE_SYNC_VERSION)
   CUDA_SAFE_CALL( cudaUnbindTexture(u_tex) );
   CUDA_SAFE_CALL( cudaUnbindTexture(q_tex) );
   CUDA_SAFE_CALL( cudaUnbindTexture(p1_tex) );
   CUDA_SAFE_CALL( cudaUnbindTexture(p2_tex) );
#endif

   CUDA_SAFE_CALL( cudaFree(d_f) );
   CUDA_SAFE_CALL( cudaFree(d_u) );
   CUDA_SAFE_CALL( cudaFree(d_p1) );
   CUDA_SAFE_CALL( cudaFree(d_p2) );
   CUDA_SAFE_CALL( cudaFree(d_q) );

#if defined(USE_SYNC_VERSION)
   CUDA_SAFE_CALL( cudaFree(d_U) );
   CUDA_SAFE_CALL( cudaFree(d_P1) );
   CUDA_SAFE_CALL( cudaFree(d_P2) );
   CUDA_SAFE_CALL( cudaFree(d_Q) );
#endif
}

void
run_denoise_tvl1_cuda(int w, int h, int nIterations, float tau, float lambda, float maxU,
                      float * u, float * p1, float * p2, float * q)
{
   int const size = sizeof(float) * w * h;

#if !defined(USE_SYNC_VERSION)
   dim3 gridDim(w/DIM_X/2, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 5;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations/2;

   CUDA_SAFE_CALL( cudaMemcpy( d_u, u, size, cudaMemcpyHostToDevice) );

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ_CB_2<nInnerIterations, 0><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                      tau, lambda, maxU,
                                                                      d_u, d_p1, d_p2, d_q);
      updateUPQ_CB_2<nInnerIterations, 1><<< gridDim, blockDim, 0 >>>(w, h, d_f,
                                                                      tau, lambda, maxU,
                                                                      d_u, d_p1, d_p2, d_q);
   }

   CUDA_SAFE_CALL( cudaMemcpy( u, d_u, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p1, d_p1, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p2, d_p2, size, cudaMemcpyDeviceToHost) );
   //CUDA_SAFE_CALL( cudaMemcpy( q, d_q, size, cudaMemcpyDeviceToHost) );
#elif 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int iter = 0; iter < nIterations; ++iter)
   {
      updatePVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, lambda, tau, d_u, d_p1, d_p2);
      updateUVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, lambda, maxU, d_f, d_u, d_p1, d_p2);
   }
   //updateQVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, d_f, d_u, d_q);

   CUDA_SAFE_CALL( cudaMemcpy( u, d_u, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p1, d_p1, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p2, d_p2, size, cudaMemcpyDeviceToHost) );
   //CUDA_SAFE_CALL( cudaMemcpy( q, d_q, size, cudaMemcpyDeviceToHost) );
#else
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int iter = 0; iter < nIterations; ++iter)
   {
# if 0
      updateU_sync<<< gridDim, blockDim, 0 >>>(w, h, tau, maxU, d_u, d_p1, d_p2, d_q, d_U);
      std::swap(d_u, d_U);
      updatePQ_sync<<< gridDim, blockDim, 0 >>>(w, h, d_f, tau, lambda, d_u, d_p1, d_p2, d_q, d_P1, d_P2, d_Q);
      std::swap(d_p1, d_P1);
      std::swap(d_p2, d_P2);
      std::swap(d_q, d_Q);
# else
      updateP_sync2<<< gridDim, blockDim, 0 >>>(w, h, tau, lambda, d_u, d_p1, d_p2, d_q, d_P1, d_P2);
      std::swap(d_p1, d_P1);
      std::swap(d_p2, d_P2);
      updateUQ_sync2<<< gridDim, blockDim, 0 >>>(w, h, d_f, tau, lambda, maxU, d_u, d_p1, d_p2, d_q, d_U, d_Q);
      std::swap(d_u, d_U);
      std::swap(d_q, d_Q);
# endif
   }

   CUDA_SAFE_CALL( cudaMemcpy( u, d_u, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p1, d_p1, size, cudaMemcpyDeviceToHost) );
   CUDA_SAFE_CALL( cudaMemcpy( p2, d_p2, size, cudaMemcpyDeviceToHost) );
   //CUDA_SAFE_CALL( cudaMemcpy( q, d_q, size, cudaMemcpyDeviceToHost) );
#endif
} // end run_denoise_tvl1_cuda()
