#include <cuda.h>

#ifdef _WINDOWS
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <cstdio>

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                            \
      cudaError err = call;                                             \
      if( cudaSuccess != err) {                                         \
         fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
         exit(EXIT_FAILURE);                                            \
      } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                    \
      CUDA_SAFE_CALL_NO_SYNC(call);                                     \
      cudaError err = cudaThreadSynchronize();                          \
      if( cudaSuccess != err) {                                         \
         fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",  \
                 __FILE__, __LINE__, cudaGetErrorString( err) );        \
         exit(EXIT_FAILURE);                                            \
      } } while (0)

#define ENABLE_U_CHECKER_PATTERN 1

struct CUDA_Flow_Buffers
{
      float *d_u, *d_v, *d_pu1, *d_pu2, *d_pv1, *d_pv2;
      float *d_qu, *d_qv;
      float *d_bu1, *d_bu2, *d_bv1, *d_bv2, *d_du, *d_dv;

      float *d_a1, *d_a2, *d_c, *d_div;
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
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_qu, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_qv, size) );

   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_bu1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_bu2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_bv1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_bv2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_du, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_dv, size) );

   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_a1, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_a2, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_c, size) );
   CUDA_SAFE_CALL( cudaMalloc( (void**) &res->d_div, size) );

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
   CUDA_SAFE_CALL( cudaFree(res->d_qu) );
   CUDA_SAFE_CALL( cudaFree(res->d_qv) );

   CUDA_SAFE_CALL( cudaFree(res->d_bu1) );
   CUDA_SAFE_CALL( cudaFree(res->d_bu2) );
   CUDA_SAFE_CALL( cudaFree(res->d_bv1) );
   CUDA_SAFE_CALL( cudaFree(res->d_bv2) );
   CUDA_SAFE_CALL( cudaFree(res->d_du) );
   CUDA_SAFE_CALL( cudaFree(res->d_dv) );

   CUDA_SAFE_CALL( cudaFree(res->d_a1) );
   CUDA_SAFE_CALL( cudaFree(res->d_a2) );
   CUDA_SAFE_CALL( cudaFree(res->d_c) );
   CUDA_SAFE_CALL( cudaFree(res->d_div) );
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

static __global__ void
_computeDivPB(int w, int h, float const * P1, float const * P2, float const * B1, float const * B2,
              float const * Q, float const * D, float * Div)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const curPos = __mul24(Y, w) + X;

   __shared__ float p1_sh[DIM_Y][DIM_X]; // Holds really p1+b1
   __shared__ float p2_sh[DIM_Y+1][DIM_X]; // Holds really p2+b2

   p1_sh[tidy][tidx]   = -P1[curPos];
   p2_sh[tidy+1][tidx] = -P2[curPos];

   p1_sh[tidy][tidx]   += B1[curPos];
   p2_sh[tidy+1][tidx] += B2[curPos];

   if (tidy == 0) p2_sh[0][tidx] = (Y > 0) ? -P2[curPos-w] : 0.0f;
   if (tidy == 0) p2_sh[0][tidx] += (Y > 0) ? B2[curPos-w] : 0.0f;

   SYNCTHREADS();

   float p1_0 = (tidx > 0) ? p1_sh[tidy][tidx-1] : 0.0f;
   p1_0 = (tidx == 0 && X > 0) ? -P1[curPos - 1] + B1[curPos-1] : p1_0;
   float p2_0 = p2_sh[tidy][tidx];
   float div_p = (((X < w-1) ? p1_sh[tidy][tidx] : 0) - p1_0 +
                  ((Y < h-1) ? p2_sh[tidy+1][tidx] : 0) - p2_0);

   Div[curPos] = div_p + Q[curPos] - D[curPos];
} // end _computeDivPB()

template <int nInnerIterations, int shift>
__global__ void
_updateU(int w, int h, float const * Div, float * U)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

#if defined(ENABLE_U_CHECKER_PATTERN)
   int const blockIdx_x = 2*blockIdx.x + ((blockIdx.y & 1) ? shift : (1-shift));
#else
   int const blockIdx_x = blockIdx.x;
#endif
   int const X = __mul24(blockIdx_x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const curPos = __mul24(Y, w) + X;

   __shared__ float u_sh[DIM_Y][DIM_X];

   float const div_p = Div[curPos];

   u_sh[tidy][tidx] = U[curPos];

   SYNCTHREADS();

   for (int iter = 0; iter < nInnerIterations; ++iter)
   {
      float num = 0.0f, u1;

      // West
      u1 = (tidx > 0) ? u_sh[tidy][tidx-1] : 0.0f;
      u1 = (tidx == 0 && X > 0) ? U[curPos - 1] : u1;
      num += u1;
      // East
      u1 = (tidx < DIM_X-1) ? u_sh[tidy][tidx+1] : 0.0f;
      u1 = (X < w-1) ? U[curPos + 1] : u1;
      num += u1;
      // North
      u1 = (tidy > 0) ? u_sh[tidy-1][tidx] : 0.0f;
      u1 = (tidy == 0 && Y > 0) ? U[curPos - w] : u1;
      num += u1;
      // South
      u1 = (tidy < DIM_Y-1) ? u_sh[tidy+1][tidx] : 0.0f;
      u1 = (Y < h-1) ? U[curPos + w] : u1;
      num += u1;

      float denom = 1.0f;
      denom += (X > 0) ? 1.0f : 0.0f;
      denom += (X < w-1) ? 1.0f : 0.0f;
      denom += (Y > 0) ? 1.0f : 0.0f;
      denom += (Y < h-1) ? 1.0f : 0.0f;

      u_sh[tidy][tidx] = (num + div_p) / denom;
   } // end for (iter)
   U[curPos] = u_sh[tidy][tidx];
} // end _updateU()

static __global__ void
_updatePB(int w, int h, float rcpMu, float const * U, float * P1, float * P2, float * B1, float * B2)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const ix = __mul24(tidy, DIM_X) + tidx;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int const pos = __mul24(Y, w) + X;

   __shared__ float u_sh[DIM_Y*DIM_X];

   u_sh[ix] = U[pos];
   SYNCTHREADS();

   float const b1 = B1[pos];
   float const b2 = B2[pos];

   float u1  = (tidx < DIM_X-1) ? u_sh[ix+1] : U[pos + 1];
   float u_x = (X < w-1) ? (u1 - u_sh[ix]) : 0.0f;
   u1        = (tidy < DIM_Y-1) ? u_sh[ix+DIM_X] : U[pos + w];
   float u_y = (Y < h-1) ? (u1 - u_sh[ix]) : 0.0f;

   float p1 = u_x + b1;
   float p2 = u_y + b2;

   float const rcpNorm = rsqrtf(p1*p1 + p2*p2);
   float const c = max(0.0f, 1.0f - rcpMu * rcpNorm);

   p1 *= c; p2 *= c;

   P1[pos] = p1;
   P2[pos] = p2;
   B1[pos] = b1 + u_x - p1;
   B2[pos] = b2 + u_y - p2;
} // end _updatePB()

static __global__ void
_updateQD(int w, int h, float lambda_over_mu,
          float const * U, float const * V, float const * Au, float const * Av, float const * C,
          float * Qu, float * Qv, float * Du, float * Dv)
{
   int const tidx = threadIdx.x;
   int const tidy = threadIdx.y;

   int const X = __mul24(blockIdx.x, blockDim.x) + tidx;
   int const Y = __mul24(blockIdx.y, blockDim.y) + tidy;
   int curPos = __mul24(Y, w) + X;

   float const u = U[curPos];
   float const v = V[curPos];
   float const Ix = Au[curPos];
   float const Iy = Av[curPos];
   float const c  = C[curPos];
   float const uu = u + Du[curPos];
   float const vv = v + Dv[curPos];

   float const D = c + Ix*uu + Iy*vv;
   float const a2 = Ix*Ix + Iy*Iy;
   float const lam_a2 = lambda_over_mu * a2;
   float const step = (D + lam_a2 < 0) ? lambda_over_mu : ((D - lam_a2 > 0) ? -lambda_over_mu : (-D/(a2+0.001f)));

   Qu[curPos] = uu + step*Ix;
   Qv[curPos] = vv + step*Iy;

   Du[curPos] = -step*Ix;
   Dv[curPos] = -step*Iy;
} // end _updateQD()

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
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_qu, dst->d_qu);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_qv, dst->d_qv);

#if 1
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_pu1, dst->d_pu1);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_pu2, dst->d_pu2);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_pv1, dst->d_pv1);
   upsampleBuffer<<< gridDim, blockDim, 0 >>>(w, h, 2.0f, src->d_pv2, dst->d_pv2);
#else
   CUDA_SAFE_CALL( cudaMemset( dst->d_pu1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pu2, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pv1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_pv2, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_qu, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_qv, 0, dstSize) );
#endif
   CUDA_SAFE_CALL( cudaMemset( dst->d_bu1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_bu2, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_bv1, 0, dstSize) );
   CUDA_SAFE_CALL( cudaMemset( dst->d_bv2, 0, dstSize) );
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
   CUDA_SAFE_CALL( cudaMemset( buffers->d_qu, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_qv, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_bu1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_bu2, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_bv1, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_bv2, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_du, 0, size) );
   CUDA_SAFE_CALL( cudaMemset( buffers->d_dv, 0, size) );
}

void
tv_flow_cuda(int w, int h, int nIterations, float const lambda, float const mu, void * buffers_)
{
   //timeval tv;
   //gettimeofday(&tv, 0);
   //double startTime = tv.tv_sec * 1000000.0 + tv.tv_usec;

   CUDA_Flow_Buffers * buffers = (CUDA_Flow_Buffers *)buffers_;

   float * d_a1  = buffers->d_a1;
   float * d_a2  = buffers->d_a2;
   float * d_c   = buffers->d_c;
   float * d_div = buffers->d_div;

   float * d_u = buffers->d_u;
   float * d_v = buffers->d_v;
   float * d_qu = buffers->d_qu;
   float * d_qv = buffers->d_qv;
   float * d_pu1 = buffers->d_pu1;
   float * d_pu2 = buffers->d_pu2;
   float * d_pv1 = buffers->d_pv1;
   float * d_pv2 = buffers->d_pv2;

   float * d_bu1 = buffers->d_bu1;
   float * d_bu2 = buffers->d_bu2;
   float * d_bv1 = buffers->d_bv1;
   float * d_bv2 = buffers->d_bv2;
   float * d_du  = buffers->d_du;
   float * d_dv  = buffers->d_dv;

   dim3 gridDim(w/DIM_X, h/DIM_Y, 1);
#if defined(ENABLE_U_CHECKER_PATTERN)
   dim3 gridDimCB(w/DIM_X/2, h/DIM_Y, 1);
#else
   dim3 gridDimCB(w/DIM_X, h/DIM_Y, 1);
#endif
   dim3 blockDim(DIM_X, DIM_Y, 1);

   int const nU_Updates = 1;
   int const NN = 2;

   for (int i = 0; i < nIterations; ++i)
   {
      _computeDivPB<<< gridDim, blockDim, 0 >>>(w, h, d_pu1, d_pu2, d_bu1, d_bu2, d_qu, d_du, d_div);
      for (int k = 0; k < nU_Updates; ++k)
      {
         _updateU<NN, 0><<< gridDimCB, blockDim, 0 >>>(w, h, d_div, d_u);
         _updateU<NN, 1><<< gridDimCB, blockDim, 0 >>>(w, h, d_div, d_u);
      }

      _computeDivPB<<< gridDim, blockDim, 0 >>>(w, h, d_pv1, d_pv2, d_bv1, d_bv2, d_qv, d_dv, d_div);
      for (int k = 0; k < nU_Updates; ++k)
      {
         _updateU<NN, 0><<< gridDimCB, blockDim, 0 >>>(w, h, d_div, d_v);
         _updateU<NN, 1><<< gridDimCB, blockDim, 0 >>>(w, h, d_div, d_v);
      }

      _updateQD<<< gridDim, blockDim, 0 >>>(w, h, lambda/mu, d_u, d_v, d_a1, d_a2, d_c, d_qu, d_qv, d_du, d_dv);

      _updatePB<<< gridDim, blockDim, 0 >>>(w, h, 1.0f/mu, d_u, d_pu1, d_pu2, d_bu1, d_bu2);
      _updatePB<<< gridDim, blockDim, 0 >>>(w, h, 1.0f/mu, d_v, d_pv1, d_pv2, d_bv1, d_bv2);
   } // end for (i)

   CUDA_SAFE_CALL( cudaThreadSynchronize() );

   //gettimeofday(&tv, 0);
   //double endTime = tv.tv_sec * 1000000.0 + tv.tv_usec;

   //printf("Done, timer = %lf sec.\n", (endTime - startTime) / 1000000.0);
}
