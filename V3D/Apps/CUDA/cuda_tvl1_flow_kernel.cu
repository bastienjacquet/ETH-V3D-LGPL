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

//#define ENABLE_BETA_TERM

static float const beta = 0.5f;

struct CUDA_Flow_Buffers
{
      float *d_u, *d_v, *d_pu1, *d_pu2, *d_pv1, *d_pv2, *d_q;

      float *d_a1, *d_a2, *d_c;
};

void *
allocateBuffers(int w, int h)
{
   // Allocate additionals row to avoid a few conditionals in the kernel
   int const size = sizeof(float) * w * (h+1);

   CUDA_Flow_Buffers * res = new CUDA_Flow_Buffers;

   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_u, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_v, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_pu1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_pu2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_pv1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_pv2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_q, size) );

   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_a1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_a2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_c, size) );

   return res;
}

void
deallocateBuffers(void * buffers)
{
   CUDA_Flow_Buffers * res = (CUDA_Flow_Buffers *)buffers;

   CUDA_SAFE_CALL( cudaFree(res->d_u) );
   CUDA_SAFE_CALL( cudaFree(res->d_v) );
   CUDA_SAFE_CALL( cudaFree(res->d_pu1) );
   CUDA_SAFE_CALL( cudaFree(res->d_pu2) );
   CUDA_SAFE_CALL( cudaFree(res->d_pv1) );
   CUDA_SAFE_CALL( cudaFree(res->d_pv2) );
   CUDA_SAFE_CALL( cudaFree(res->d_q) );

   CUDA_SAFE_CALL( cudaFree(res->d_a1) );
   CUDA_SAFE_CALL( cudaFree(res->d_a2) );
   CUDA_SAFE_CALL( cudaFree(res->d_c) );
}

void
getBuffers(int w, int h, void * buffers, float * u, float * v)
{
   int const size = sizeof(float) * w * h;
   CUDA_Flow_Buffers * src = (CUDA_Flow_Buffers *)buffers;

   if (u) CUDA_SAFE_CALL( cudaMemcpy( u, src->d_u, size, cudaMemcpyDeviceToHost) );
   if (v) CUDA_SAFE_CALL( cudaMemcpy( v, src->d_v, size, cudaMemcpyDeviceToHost) );
}

#define DIM_X 16
#define DIM_Y 8

#define SYNCTHREADS() __syncthreads()
//#define SYNCTHREADS()

//**********************************************************************

template <int nInnerIterations, int shift>
__global__ void
updateUPQ_CB(int w, int h, float tau, float lambda,
             float const * A1, float const * A2, float const * C,
             float * U, float * V, float * PU1, float * PU2, float * PV1, float * PV2, float * Q)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
   int const X = __mul24(blockIdx_x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;

   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float v_sh[DIM_Y+1][DIM_X+1];
   __shared__ float pu1_sh[DIM_Y][DIM_X+1];
   __shared__ float pu2_sh[DIM_Y+1][DIM_X];
   __shared__ float pv1_sh[DIM_Y][DIM_X+1];
   __shared__ float pv2_sh[DIM_Y+1][DIM_X];

   // Load u, p and q
   u_sh[tidy][tidx]     = U[pos];
   v_sh[tidy][tidx]     = V[pos];
   pu1_sh[tidy][tidx+1] = PU1[pos];
   pu2_sh[tidy+1][tidx] = PU2[pos];
   pv1_sh[tidy][tidx+1] = PV1[pos];
   pv2_sh[tidy+1][tidx] = PV2[pos];

   //__syncthreads();

   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

   if (tidx == DIM_X-1) v_sh[tidy][DIM_X] = (X < w-1) ? V[pos+1] : v_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) v_sh[DIM_Y][tidx] = (Y < h-1) ? V[pos+w] : v_sh[DIM_Y-1][tidx];

   if (tidx == 0)       pu1_sh[tidy][0]       = (X > 0)   ? PU1[pos-1] : 0;
   if (tidy == 0)       pu2_sh[0][tidx]       = (Y > 0)   ? PU2[pos-w] : 0;
   if (tidx == 0)       pv1_sh[tidy][0]       = (X > 0)   ? PV1[pos-1] : 0;
   if (tidy == 0)       pv2_sh[0][tidx]       = (Y > 0)   ? PV2[pos-w] : 0;

   __syncthreads();

   float q_cur    = Q[pos];
   float const a1 = A1[pos];
   float const a2 = A2[pos];
   float const c  = C[pos];

#pragma unroll
   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
      float const u0  = u_sh[tidy][tidx];
      float const u_x = u_sh[tidy][tidx+1] - u0;
      float const u_y = u_sh[tidy+1][tidx] - u0;

      float const v0  = v_sh[tidy][tidx];
      float const v_x = v_sh[tidy][tidx+1] - v0;
      float const v_y = v_sh[tidy+1][tidx] - v0;

      float const new_pu1 = pu1_sh[tidy][tidx+1] - tau * u_x;
      float const new_pu2 = pu2_sh[tidy+1][tidx] - tau * u_y;

      float const new_pv1 = pv1_sh[tidy][tidx+1] - tau * v_x;
      float const new_pv2 = pv2_sh[tidy+1][tidx] - tau * v_y;

      __syncthreads();

      float tv = sqrtf(new_pu1*new_pu1 + new_pu2*new_pu2);
      //float denom = max(1.0f, tv);
      float denom = max(1.0f, tv * lambda);
      pu1_sh[tidy][tidx+1] = new_pu1 / denom;
      pu2_sh[tidy+1][tidx] = new_pu2 / denom;

      tv = sqrtf(new_pv1*new_pv1 + new_pv2*new_pv2);
      //denom = max(1.0f, tv);
      denom = max(1.0f, tv * lambda);
      pv1_sh[tidy][tidx+1] = new_pv1 / denom;
      pv2_sh[tidy+1][tidx] = new_pv2 / denom;

      __syncthreads();

      // Update Q

      float const new_q = q_cur - tau * (a1*u0 + a2*v0 + c);
      //q_cur = max(-lambda, min(lambda, new_q));
      q_cur = max(-1.0f, min(1.0f, new_q));

      // Update U and V

      float const div_pu = (((X < w-1) ? pu1_sh[tidy][tidx+1] : 0) -
                            pu1_sh[tidy][tidx] +
                            ((Y < h-1) ? pu2_sh[tidy+1][tidx] : 0) -
                            pu2_sh[tidy][tidx]);

      u_sh[tidy][tidx] = u0 - tau * (div_pu - a1*q_cur);

      float const div_pv = (((X < w-1) ? pv1_sh[tidy][tidx+1] : 0) -
                            pv1_sh[tidy][tidx] +
                            ((Y < h-1) ? pv2_sh[tidy+1][tidx] : 0) -
                            pv2_sh[tidy][tidx]);

      v_sh[tidy][tidx] = v0 - tau * (div_pv - a2*q_cur);

      __syncthreads();
   } // end for (iter)

   U[pos]  = u_sh[tidy][tidx];
   V[pos]  = v_sh[tidy][tidx];
   Q[pos]  = q_cur;
   PU1[pos] = pu1_sh[tidy][tidx+1];
   PU2[pos] = pu2_sh[tidy+1][tidx];
   PV1[pos] = pv1_sh[tidy][tidx+1];
   PV2[pos] = pv2_sh[tidy+1][tidx];
} // end updateUPQ_CB_2()

//**********************************************************************

// template <int nInnerIterations, int shift>
// __global__ void
// updateUPQ_CB_2(int w, int h, float tau, float lambda,
//                float const * A1, float const * A2, float const * C,
//                float * U, float * V, float * PU1, float * PU2, float * PV1, float * PV2, float * Q)
// {
//    // Time step 0.22
// //    float const beta_p = 1.0f;
// //    float const beta_q = 1.0f;
//    // Timestep 0.2
//    float const beta_p = 0.5f;
//    float const beta_q = 0.5f;

//    int const tidx = threadIdx.x;
//    int const tidy = threadIdx.y;

//    int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
//    int const X = __mul24(blockIdx_x, blockDim.x) + tidx;
//    int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
//    int const pos = __mul24(Y, w) + X;

//    __shared__ float u_sh[DIM_Y+1][DIM_X+1];
//    __shared__ float v_sh[DIM_Y+1][DIM_X+1];
//    __shared__ float q_sh[DIM_Y+1][DIM_X+1];
//    __shared__ float pu1_sh[DIM_Y+1][DIM_X+2];
//    __shared__ float pu2_sh[DIM_Y+2][DIM_X+1];
//    __shared__ float pv1_sh[DIM_Y+1][DIM_X+2];
//    __shared__ float pv2_sh[DIM_Y+2][DIM_X+1];

//    float const f = F[pos];

//    // Load u, p and q
//    u_sh[tidy][tidx]    = U[pos];
//    q_sh[tidy][tidx]    = Q[pos];
//    pu1_sh[tidy][tidx+1] = PU1[pos];
//    pu2_sh[tidy+1][tidx] = PU2[pos];
//    pv1_sh[tidy][tidx+1] = PV1[pos];
//    pv2_sh[tidy+1][tidx] = PV2[pos];

//    //__syncthreads();

//    if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
//    if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

//    if (tidx == DIM_X-1) v_sh[tidy][DIM_X] = (X < w-1) ? V[pos+1] : v_sh[tidy][DIM_X-1];
//    if (tidy == DIM_Y-1) v_sh[DIM_Y][tidx] = (Y < h-1) ? V[pos+w] : v_sh[DIM_Y-1][tidx];

//    if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? Q[pos+1] : q_sh[tidy][DIM_X-1];
//    if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? Q[pos+w] : q_sh[DIM_Y-1][tidx];

//    if (tidx == 0)       pu1_sh[tidy][0]       = (X > 0)   ? PU1[pos-1] : 0;
//    if (tidx == DIM_X-1) pu1_sh[tidy][DIM_X+1] = (X < w-1) ? PU1[pos+1] : 0;

//    if (tidy == 0)       pu2_sh[0][tidx]       = (Y > 0)   ? PU2[pos-w] : 0;
//    if (tidy == DIM_Y-1) pu2_sh[DIM_Y+1][tidx] = (Y < h-1) ? PU2[pos+w] : 0;

//    if (tidy == DIM_Y-1) pu1_sh[DIM_Y][tidx]   = (Y < h-1) ? PU1[pos+w] : 0;
//    if (tidx == DIM_X-1) pu2_sh[tidy][DIM_X]   = (X < w-1) ? PU2[pos+1] : 0;

//    if (tidx == 0)       pv1_sh[tidy][0]       = (X > 0)   ? PV1[pos-1] : 0;
//    if (tidx == DIM_X-1) pv1_sh[tidy][DIM_X+1] = (X < w-1) ? PV1[pos+1] : 0;

//    if (tidy == 0)       pv2_sh[0][tidx]       = (Y > 0)   ? PV2[pos-w] : 0;
//    if (tidy == DIM_Y-1) pv2_sh[DIM_Y+1][tidx] = (Y < h-1) ? PV2[pos+w] : 0;

//    if (tidy == DIM_Y-1) pv1_sh[DIM_Y][tidx]   = (Y < h-1) ? PV1[pos+w] : 0;
//    if (tidx == DIM_X-1) pv2_sh[tidy][DIM_X]   = (X < w-1) ? PV2[pos+1] : 0;

//    __syncthreads();

// #pragma unroll
//    for (int iter = 0; iter < nInnerIterations; ++iter)
//    {
//       float const u0  = u_sh[tidy][tidx];
//       float const u_x = u_sh[tidy][tidx+1] - u0;
//       float const u_y = u_sh[tidy+1][tidx] - u0;

//       float const q0  = q_sh[tidy][tidx];
//       float const q_x = q_sh[tidy][tidx+1] - q0;
//       float const q_y = q_sh[tidy+1][tidx] - q0;

//       // Divergence at (x, y)
//       float const div_p_00 = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
//                               p1_sh[tidy][tidx] +
//                               ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
//                               p2_sh[tidy][tidx]);

//       // Divergence at (x+1, y)
//       float const div_p_10 = ((X == w-1) ? div_p_00 :
//                               (p1_sh[tidy][tidx+2] -
//                                p1_sh[tidy][tidx+1] +
//                                ((Y < h-1) ? p2_sh[tidy+1][tidx+1] : 0) -
//                                p2_sh[tidy][tidx+1]));

//       // Divergence at (x, y+1)
//       float const div_p_01 = ((Y == h-1) ? div_p_00 :
//                               ((X < w-1) ? p1_sh[tidy+1][tidx+1] : 0) -
//                               p1_sh[tidy+1][tidx] +
//                               p2_sh[tidy+2][tidx] -
//                               p2_sh[tidy+1][tidx]);

//       float dp1 = u_x - beta_p * (q_x - div_p_10 + div_p_00);
//       float dp2 = u_y - beta_p * (q_y - div_p_01 + div_p_00);

//       // Update P

//       float const new_p1 = p1_sh[tidy][tidx+1] + tau * dp1;
//       float const new_p2 = p2_sh[tidy+1][tidx] + tau * dp2;

//       __syncthreads();

//       float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
//       float const denom = max(1.0f, tv);
//       p1_sh[tidy][tidx+1] = new_p1 / denom;
//       p2_sh[tidy+1][tidx] = new_p2 / denom;

//       __syncthreads();

//       // Update Q
//       float const div_p = (((X < w-1) ? p1_sh[tidy][tidx+1] : 0) -
//                            p1_sh[tidy][tidx] +
//                            ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) -
//                            p2_sh[tidy][tidx]);

//       float new_q = q0 + tau * ((u0 - f) + beta_q * (div_p - q0));
//       new_q = max(-lambda, min(lambda, new_q));

//       // Update U
//       float const new_u = u0 + tau * (div_p - new_q);
//       u_sh[tidy][tidx] = max(0.0f, min(maxU, new_u));
//       q_sh[tidy][tidx] = new_q;

//       __syncthreads();
//    } // end for (iter)

//    U[pos]  = u_sh[tidy][tidx];
//    Q[pos]  = q_sh[tidy][tidx];
//    P1[pos] = p1_sh[tidy][tidx+1];
//    P2[pos] = p2_sh[tidy+1][tidx];
// } // end updateUPQ_CB_2()

//**********************************************************************

__global__ void
updateUVolume(int w, int h, float tau, float * U, float const * P1, float const * P2, float const * Q, float const * A)
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
   float const q = Q[curPos];
   float const a = A[curPos];
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

   float u_new = u - tau * (div_p - q*a);
   U[curPos] = u_new;
} // end updateUVolume()

__global__ void
updatePVolume(int w, int h, float alpha, float tau, float * const U, float * P1, float * P2, float const * Q, float const * A)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int pos = Y*w + X;

#if 1 || !defined(ENABLE_BETA_TERM)
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
   float new_p1 = p1_cur - tau * u_x;
   float new_p2 = p2_cur - tau * u_y;

   float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha);
   //float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2));
   new_p1 /= norm;
   new_p2 /= norm;
# else
   float const tv = sqrtf(u_x*u_x + u_y*u_y);
   float const denom_p = 1.0f / (1.0f + tau * tv);
   float new_p1 = (p1_cur - tau * u_x) * denom_p;
   float new_p2 = (p2_cur - tau * u_y) * denom_p;
# endif
#else
   __shared__ float u_sh[DIM_Y+1][DIM_X+1];
   __shared__ float q_sh[DIM_Y+1][DIM_X+1];
   __shared__ float a_sh[DIM_Y+1][DIM_X+1];
   __shared__ float p1_sh[DIM_Y+1][DIM_X+2];
   __shared__ float p2_sh[DIM_Y+2][DIM_X+1];

   // Load u, p and q
   u_sh[tidy][tidx]    = U[pos];
   q_sh[tidy][tidx]    = Q[pos];
   a_sh[tidy][tidx]    = A[pos];
   p1_sh[tidy][tidx+1] = P1[pos];
   p2_sh[tidy+1][tidx] = P2[pos];

   //__syncthreads();

   if (tidx == DIM_X-1) u_sh[tidy][DIM_X] = (X < w-1) ? U[pos+1] : u_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) u_sh[DIM_Y][tidx] = (Y < h-1) ? U[pos+w] : u_sh[DIM_Y-1][tidx];

   if (tidx == DIM_X-1) q_sh[tidy][DIM_X] = (X < w-1) ? Q[pos+1] : q_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) q_sh[DIM_Y][tidx] = (Y < h-1) ? Q[pos+w] : q_sh[DIM_Y-1][tidx];

   if (tidx == DIM_X-1) a_sh[tidy][DIM_X] = (X < w-1) ? A[pos+1] : a_sh[tidy][DIM_X-1];
   if (tidy == DIM_Y-1) a_sh[DIM_Y][tidx] = (Y < h-1) ? A[pos+w] : a_sh[DIM_Y-1][tidx];

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

#if 0
   float const q0  = q_sh[tidy][tidx];
   float const q_x = q_sh[tidy][tidx+1] - q0;
   float const q_y = q_sh[tidy+1][tidx] - q0;
#elif 1
   float const q0  = q_sh[tidy][tidx]*a_sh[tidy][tidx];
   float const q_x = q_sh[tidy][tidx+1]*a_sh[tidy][tidx+1] - q0;
   float const q_y = q_sh[tidy+1][tidx]*a_sh[tidy+1][tidx] - q0;
#elif 0
   float const q0  = q_sh[tidy][tidx]*a_sh[tidy][tidx];
   float const q_x = q_sh[tidy][tidx+1]*a_sh[tidy][tidx] - q0;
   float const q_y = q_sh[tidy+1][tidx]*a_sh[tidy][tidx] - q0;
#else
   float const q0  = q_sh[tidy][tidx];
   float q_x = q_sh[tidy][tidx+1] - q0;
   float q_y = q_sh[tidy+1][tidx] - q0;

   float const a0  = a_sh[tidy][tidx];
   float const a_x = a_sh[tidy][tidx+1] - a0;
   float const a_y = a_sh[tidy+1][tidx] - a0;

   q_x = a0 * q_x + q0 * a_x;
   q_y = a0 * q_y + q0 * a_y;
#endif

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
   float dp1 = -u_x - beta * (q_x - div_p_10 + div_p_00);
   float dp2 = -u_y - beta * (q_y - div_p_01 + div_p_00);

   float new_p1 = p1_sh[tidy][tidx+1] + tau * dp1;
   float new_p2 = p2_sh[tidy+1][tidx] + tau * dp2;

   float const tv = sqrtf(new_p1*new_p1 + new_p2*new_p2);
   float const denom = max(1.0f, tv * alpha);
   new_p1 = new_p1 / denom;
   new_p2 = new_p2 / denom;
#endif

   P1[pos] = new_p1;
   P2[pos] = new_p2;
} // end updatePVolume()

__global__ void
updateQVolume(int w, int h, float tau, float alpha,
              float const * U, float const * V, float * const A1, float * A2, float const * C,
              float const * PU1, float const * PU2, float const * PV1, float const * PV2, float * Q)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

#if defined(ENABLE_BETA_TERM)
   __shared__ float p1_sh[DIM_Y*DIM_X];
   __shared__ float p2_sh[DIM_Y*DIM_X];

   p1_sh[ix] = PU1[curPos];
   p2_sh[ix] = PU2[curPos];
   SYNCTHREADS();

   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PU1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? PU2[curPos - w] : p2_0;

   float const div_pu = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                         ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   p1_sh[ix] = PV1[curPos];
   p2_sh[ix] = PV2[curPos];
   SYNCTHREADS();

   p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PV1[curPos - 1] : p1_0;

   p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? PV2[curPos - w] : p2_0;

   float const div_pv = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                         ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);
#endif

   float const q = Q[curPos];
   float const u = U[curPos];
   float const v = V[curPos];
   float const a1 = A1[curPos];
   float const a2 = A2[curPos];
   float const c  = C[curPos];

   float dq = -(a1*u + a2*v + c);
#if defined(ENABLE_BETA_TERM)
   dq -= beta * (a1*(q*a1 - div_pu) + a2*(q*a2 - div_pv));
#endif

   float new_q = q + tau * dq;
   Q[curPos] = max(-1.0f, min(1.0f, new_q));
   //Q[curPos] = max(-alpha, min(alpha, new_q));
}

__global__ void
updateUVQVolume(int w, int h, float tau, 
                float * U, float * V, float * const A1, float * A2, float const * C,
                float const * PU1, float const * PU2, float const * PV1, float const * PV2, float * Q)
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
   float const q  = Q[curPos];
   float const a1 = A1[curPos];
   p1_sh[ix] = PU1[curPos];
   p2_sh[ix] = PU2[curPos];

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PU1[curPos - 1] : p1_0;

   float p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? PU2[curPos - w] : p2_0;

   float div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
                  ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float u_new = u - tau * (div_p - q*a1);
   U[curPos] = u_new;

   float v = V[curPos];
   float const a2 = A2[curPos];
   p1_sh[ix] = PV1[curPos];
   p2_sh[ix] = PV2[curPos];

   SYNCTHREADS();

   // Update v
   p1_0 = (tidx > 0) ? p1_sh[ix-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PV1[curPos - 1] : p1_0;

   p2_0 = (tidy > 0) ? p2_sh[ix-DIM_X] : 0.0f;
   p2_0 = (tidy == 0 && Y > 0) ? PV2[curPos - w] : p2_0;

   div_p = (((X < w-1) ? p1_sh[ix] : 0) - p1_0 +
            ((Y < h-1) ? p2_sh[ix] : 0) - p2_0);

   float v_new = v - tau * (div_p - q*a2);
   V[curPos] = v_new;

   // Update q
   float const c  = C[curPos];

   float dq = -(a1*u + a2*v + c);

   float new_q = q + tau * dq;
   Q[curPos] = max(-1.0f, min(1.0f, new_q));
} // end updateUVQVolume()


//**********************************************************************

__global__ void
updateUVolume_theta(int w, int h, float alpha, float const theta,
                    float const * A1, float const * A2, float const * C,
                    float * U, float * V, float const * PU1, float const * PU2, float const * PV1, float const * PV2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   //int const pos = __mul24(Y, w) + X;
   int curPos = Y*w + X;

   __shared__ float p1_sh[DIM_Y][DIM_X];
   __shared__ float p2_sh[DIM_Y+1][DIM_X];

   // Load u, p and q of current slice/disparity
   float u = U[curPos];
   float v = V[curPos];
   float const a1 = A1[curPos];
   float const a2 = A2[curPos];
   float const c  = C[curPos];

   p1_sh[tidy][tidx]   = PU1[curPos];
   p2_sh[tidy+1][tidx] = PU2[curPos];

   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? PU2[curPos-w] : 0.0f;

   float D = c + a1*u + a2*v;

   float R2 = a1*a1 + a2*a2;
   float lam_R2 = alpha * theta * R2;
   float step = (D + lam_R2 < 0) ? alpha*theta : ((D - lam_R2 > 0) ? -alpha*theta : (-D/(R2+0.001f)));

   u = u + step * a1;
   v = v + step * a2;

   SYNCTHREADS();

   // Update u
   float p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PU1[curPos - 1] : p1_0;
   float p2_0 = p2_sh[tidy][tidx];
   float div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
                  ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   U[curPos] = u - theta * div_p;

   // Update v
   p1_sh[tidy][tidx]   = PV1[curPos];
   p2_sh[tidy+1][tidx] = PV2[curPos];
   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? PV2[curPos-w] : 0.0f;
   SYNCTHREADS();

   p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? PV1[curPos - 1] : p1_0;
   p2_0 = p2_sh[tidy][tidx];
   div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
            ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   V[curPos] = v - theta * div_p;
} // end updateUVolume_theta()

__global__ void
updatePVolume_theta(int w, int h, float tau_over_theta, float * const U, float * P1, float * P2)
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

   //float norm = max(1.0f, sqrtf(new_p1*new_p1 + new_p2*new_p2) * alpha);
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


#define COEFFS_DIM_X 16
#define COEFFS_DIM_Y 8

//typedef unsigned char PyrPixelType;
typedef float PyrPixelType;

texture<PyrPixelType, 2, cudaReadModeElementType> I0_tex;
texture<PyrPixelType, 2, cudaReadModeElementType> I1_tex;

__global__ void
computeCoeffs(int w, int h, float alpha, float const * U, float const * V, float * A1, float * A2, float * C)
{
   int const X = blockIdx.x*blockDim.x + threadIdx.x;
   int const Y = blockIdx.y*blockDim.y + threadIdx.y;

   int const pos = Y*w + X;

   float const u = U[pos];
   float const v = V[pos];

   float I0 = tex2D(I0_tex, X, Y);
   float I1 = tex2D(I1_tex, X+u, Y+v);
   float I1x = tex2D(I1_tex, X+u+0.5f, Y+v) - tex2D(I1_tex, X+u-0.5f, Y+v);
   float I1y = tex2D(I1_tex, X+u, Y+v+0.5f) - tex2D(I1_tex, X+u, Y+v-0.5f);

   I0 /= 255.0f; I1 /= 255.0f; I1x /= 255.0f; I1y /= 255.0f;

   float const eps = 0.01f;
   I1x = (I1x < 0.0f) ? min(-eps, I1x) : max(eps, I1x);
   I1y = (I1y < 0.0f) ? min(-eps, I1y) : max(eps, I1y);

   A1[pos] = alpha*I1x;
   A2[pos] = alpha*I1y;
   C[pos]  = alpha*(I1 - I1x*u - I1y*v - I0);
} // end computeCoeffs()

void
computeCoefficients(int w, int h, float alpha, PyrPixelType const * I0, PyrPixelType const * I1, void * bufs)
{
   CUDA_Flow_Buffers * buffers = (CUDA_Flow_Buffers *)bufs;

   dim3 gridDim(w/COEFFS_DIM_X, h/COEFFS_DIM_Y, 1);
   dim3 blockDim(COEFFS_DIM_X, COEFFS_DIM_Y, 1);

   int const size = w*h*sizeof(PyrPixelType);

   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<PyrPixelType>();
   cudaArray * d_I0Array;
   cudaArray * d_I1Array;

   CUDA_SAFE_CALL( cudaMallocArray(&d_I0Array, &channelDesc, w, h));
   CUDA_SAFE_CALL( cudaMemcpyToArray(d_I0Array, 0, 0, I0, size, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL( cudaBindTextureToArray(I0_tex, d_I0Array, channelDesc) );

   CUDA_SAFE_CALL( cudaMallocArray(&d_I1Array, &channelDesc, w, h));
   CUDA_SAFE_CALL( cudaMemcpyToArray(d_I1Array, 0, 0, I1, size, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL( cudaBindTextureToArray(I1_tex, d_I1Array, channelDesc) );

   I0_tex.addressMode[0] = cudaAddressModeClamp;
   I0_tex.addressMode[1] = cudaAddressModeClamp;
   I0_tex.filterMode = cudaFilterModeLinear;
   I0_tex.normalized = false;

   I1_tex.addressMode[0] = cudaAddressModeClamp;
   I1_tex.addressMode[1] = cudaAddressModeClamp;
   I1_tex.filterMode = cudaFilterModeLinear;
   I1_tex.normalized = false;

   computeCoeffs<<< gridDim, blockDim, 0>>>(w, h, alpha, buffers->d_u, buffers->d_v, buffers->d_a1, buffers->d_a2, buffers->d_c);

   CUDA_SAFE_CALL( cudaUnbindTexture(&I0_tex) );
   CUDA_SAFE_CALL( cudaUnbindTexture(&I1_tex) );
   CUDA_SAFE_CALL( cudaFreeArray(d_I0Array) );
   CUDA_SAFE_CALL( cudaFreeArray(d_I1Array) );
}

//**********************************************************************


#define UPSAMPLE_DIM_X 16

__global__ void
upsampleBuffer(int w, int h, float const factor, float const * Usrc, float * Udst)
{
   int const tidx = threadIdx.x;

   int const X0 = blockIdx.x*blockDim.x;
   int const Y  = blockIdx.y;

   __shared__ float u_sh[UPSAMPLE_DIM_X];

   u_sh[tidx] = factor * Usrc[Y*w + X0 + tidx];

   float u0 = u_sh[tidx/2];
   Udst[2*X0 + tidx + (2*Y+0)*2*w] = u0;
   Udst[2*X0 + tidx + (2*Y+1)*2*w] = u0;

   float u1 = u_sh[tidx/2 + UPSAMPLE_DIM_X/2];
   Udst[2*X0 + tidx + UPSAMPLE_DIM_X + (2*Y+0)*2*w] = u1;
   Udst[2*X0 + tidx + UPSAMPLE_DIM_X + (2*Y+1)*2*w] = u1;
}

void
upsampleBuffers(int w, int h, void * srcBuffers, void * dstBuffers)
{
   int const dstSize = 4*w*h*sizeof(float);

   CUDA_Flow_Buffers * src = (CUDA_Flow_Buffers *)srcBuffers;
   CUDA_Flow_Buffers * dst = (CUDA_Flow_Buffers *)dstBuffers;

   dim3 gridDim(w/UPSAMPLE_DIM_X, h, 1);
   dim3 blockDim(UPSAMPLE_DIM_X, 1, 1);
   
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_u, dst->d_u);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_v, dst->d_v);

#if 1
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src->d_q, dst->d_q);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src->d_pu1, dst->d_pu1);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src->d_pu2, dst->d_pu2);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src->d_pv1, dst->d_pv1);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 1.0f, src->d_pv2, dst->d_pv2);
#else
//    CUDA_SAFE_CALL( cudaMemset( dst->d_u, 0, dstSize) );
//    CUDA_SAFE_CALL( cudaMemset( dst->d_v, 0, dstSize) );

   CUDA_SAFE_CALL( cudaMemset( dst->d_pu1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pu2, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pv1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pv2, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_q, 0, dstSize) );
#endif
}

void
setZeroBuffers(int w, int h, void * buffers_)
{
   int const size = w*h*sizeof(float);

   CUDA_Flow_Buffers * buffers = (CUDA_Flow_Buffers *)buffers_;

   CUDA_SAFE_CALL( cudaMemset( buffers->d_u, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_v, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_pu1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_pu2, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_pv1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_pv2, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_q, 0, size) );
}

void
tv_flow_cuda(int w, int h, int nIterations, float alpha, float const theta, float tau, void * buffers_)
{
   //timeval tv;
   //gettimeofday(&tv, 0);
   //double startTime = tv.tv_sec * 1000000.0 + tv.tv_usec;

   CUDA_Flow_Buffers * buffers = (CUDA_Flow_Buffers *)buffers_;

   float * d_a1 = buffers->d_a1;
   float * d_a2 = buffers->d_a2;
   float * d_c  = buffers->d_c;

   float * d_u = buffers->d_u;
   float * d_v = buffers->d_v;
   float * d_q = buffers->d_q;
   float * d_pu1 = buffers->d_pu1;
   float * d_pu2 = buffers->d_pu2;
   float * d_pv1 = buffers->d_pv1;
   float * d_pv2 = buffers->d_pv2;

#if 0
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int i = 0; i < nIterations; ++i)
   {
      updatePVolume<<< gridDim, blockDim, 0 >>>(w, h, alpha, tau,
                                                buffers->d_u, buffers->d_pu1, buffers->d_pu2,
                                                buffers->d_q, buffers->d_a1);
      updatePVolume<<< gridDim, blockDim, 0 >>>(w, h, alpha, tau,
                                                buffers->d_v, buffers->d_pv1, buffers->d_pv2,
                                                buffers->d_q, buffers->d_a2);
# if 0
      updateQVolume<<< gridDim, blockDim, 0 >>>(w, h, tau, alpha,
                                                buffers->d_u, buffers->d_v, d_a1, d_a2, d_c,
                                                buffers->d_pu1, buffers->d_pu2, buffers->d_pv1, buffers->d_pv2,
                                                buffers->d_q);

      updateUVolume<<< gridDim, blockDim, 0 >>>(w, h, tau, buffers->d_u, buffers->d_pu1, buffers->d_pu2,
                                                buffers->d_q, d_a1);
      updateUVolume<<< gridDim, blockDim, 0 >>>(w, h, tau, buffers->d_v, buffers->d_pv1, buffers->d_pv2,
                                                buffers->d_q, d_a2);
# else
      updateUVQVolume<<< gridDim, blockDim, 0 >>>(w, h, tau,
                                                  buffers->d_u, buffers->d_v, d_a1, d_a2, d_c,
                                                  buffers->d_pu1, buffers->d_pu2, buffers->d_pv1, buffers->d_pv2,
                                                  buffers->d_q);
# endif
   }
#elif 1
   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   for (int i = 0; i < nIterations; ++i)
   {
      updatePVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, tau/theta,
                                                      buffers->d_u, buffers->d_pu1, buffers->d_pu2);
      updatePVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, tau/theta,
                                                      buffers->d_v, buffers->d_pv1, buffers->d_pv2);

      updateUVolume_theta<<< gridDim, blockDim, 0 >>>(w, h, alpha, theta, d_a1, d_a2, d_c,
                                                      d_u, d_v, d_pu1, d_pu2, d_pv1, d_pv2);
   }
#else
   dim3 gridDim(w/DIM_X/2, h/DIM_Y, 1);
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nInnerIterations = 2;
   int const nOuterIterations = (nIterations+nInnerIterations-1)/nInnerIterations;

   for (int iter = 0; iter < nOuterIterations; ++iter)
   {
      updateUPQ_CB<nInnerIterations, 0><<< gridDim, blockDim, 0 >>>(w, h, tau, alpha, d_a1, d_a2, d_c,
                                                                    d_u, d_v, d_pu1, d_pu2, d_pv1, d_pv2, d_q);
      updateUPQ_CB<nInnerIterations, 1><<< gridDim, blockDim, 0 >>>(w, h, tau, alpha, d_a1, d_a2, d_c,
                                                                    d_u, d_v, d_pu1, d_pu2, d_pv1, d_pv2, d_q);
   }
#endif

   CUDA_SAFE_CALL( cudaThreadSynchronize() );

   //gettimeofday(&tv, 0);
   //double endTime = tv.tv_sec * 1000000.0 + tv.tv_usec;

   //printf("Done, timer = %lf sec.\n", (endTime - startTime) / 1000000.0);
}
