#define NOMINMAX 

#include "Base/v3d_image.h"
#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace V3D;

namespace
{

   //typedef unsigned char PyrPixelType;
   typedef float PyrPixelType;

   typedef unsigned char byte;

   inline void
   reduce4(int w, int h, PyrPixelType* input,  PyrPixelType* output)
   {
      for (int y = 0; y < h; ++y)
      {
         int const yy = 2*y;
         int const yy0 = (y > 0) ? (yy-1) : 1;
         int const yy1 = yy;
         //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
         int const yy2 = yy+1;
         int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

         int const rowOfs = y*w;
         PyrPixelType const * row0 = input + yy0*2*w; PyrPixelType const * row1 = input + yy1*2*w;
         PyrPixelType const * row2 = input + yy2*2*w; PyrPixelType const * row3 = input + yy3*2*w;

         for (int x = 1; x < w-1; ++x)
         {
            int const xx = 2*x;
            PyrPixelType value = 0;

            value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+2];
            value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+2];
            value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+2];
            value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+2];

            output[x + rowOfs] = (value+32) / (1 << 6);
         } // end for (x)
      } // end for (y)

      // Column x = 0
      for (int y = 0; y < h; ++y)
      {
         int const yy = 2*y;
         int const yy0 = (y > 0) ? (yy-1) : 1;
         int const yy1 = yy;
         //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
         int const yy2 = yy+1;
         int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

         int const rowOfs = y*w;
         PyrPixelType const * row0 = input + yy0*2*w; PyrPixelType const * row1 = input + yy1*2*w;
         PyrPixelType const * row2 = input + yy2*2*w; PyrPixelType const * row3 = input + yy3*2*w;

         int const x = 0;
         int const xx = 2*x;
         PyrPixelType value = 0;

         value += 1*row0[xx+1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+2];
         value += 3*row1[xx+1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+2];
         value += 3*row2[xx+1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+2];
         value += 1*row3[xx+1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+2];

         output[x + rowOfs] = (value+32) / (1 << 6);
      } // end for (y)

//       // Column x = w-2
//       for (int y = 0; y < h; ++y)
//       {
//          int const yy = 2*y;
//          int const yy0 = (y > 0) ? (yy-1) : 1;
//          int const yy1 = yy;
//          //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
//          int const yy2 = yy+1;
//          int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

//          int const rowOfs = y*w;
//          byte const * row0 = input + yy0*2*w; byte const * row1 = input + yy1*2*w;
//          byte const * row2 = input + yy2*2*w; byte const * row3 = input + yy3*2*w;

//          int const x = w-2;
//          int const xx = 2*x;
//          int value = 0;

//          value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx+0];
//          value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx+0];
//          value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx+0];
//          value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx+0];

//          output[x + rowOfs] = (value+32) >> 6;
//       } // end for (y)

      // Column x = w-1
      for (int y = 0; y < h; ++y)
      {
         int const yy = 2*y;
         int const yy0 = (y > 0) ? (yy-1) : 1;
         int const yy1 = yy;
         //int const yy2 = (y < h-1) ? (yy+1) : (yy-1);
         int const yy2 = yy+1;
         int const yy3 = (y < h-2) ? (yy+2) : (yy-2);

         int const rowOfs = y*w;
         PyrPixelType const * row0 = input + yy0*2*w; PyrPixelType const * row1 = input + yy1*2*w;
         PyrPixelType const * row2 = input + yy2*2*w; PyrPixelType const * row3 = input + yy3*2*w;

         int const x = w-1;
         int const xx = 2*x;
         PyrPixelType value = 0;

         value += 1*row0[xx-1] + 3*row0[xx+0] + 3*row0[xx+1] + 1*row0[xx-1];
         value += 3*row1[xx-1] + 9*row1[xx+0] + 9*row1[xx+1] + 3*row1[xx-1];
         value += 3*row2[xx-1] + 9*row2[xx+0] + 9*row2[xx+1] + 3*row2[xx-1];
         value += 1*row3[xx-1] + 3*row3[xx+0] + 3*row3[xx+1] + 1*row3[xx-1];

         output[x + rowOfs] = (value+32) / (1 << 6);
      } // end for (y)
   } // end reduce4()

   // build Image Pyramid
   inline void
   buildPyramid(Image<unsigned char> const& source, vector<Image<PyrPixelType> >& pyramid)
   {
      size_t const nLevels = pyramid.size();

      // first level
      int const w = source.width();
      int const h = source.height();
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            pyramid[0](x, y) = PyrPixelType(source(x, y));

      // next levels
      for (size_t i = 1; i < nLevels; ++i)
         reduce4(pyramid[i].width(), pyramid[i].height(), &pyramid[i-1](0, 0), &pyramid[i](0, 0));
   } // end buildPyramid

} // end namespace <>

void * allocateBuffers(int w, int h);
void deallocateBuffers(void * buffers);
void getBuffers(int w, int h, void * buffers, float * u, float * v);
void computeCoefficients(int w, int h, float alpha, PyrPixelType const * I0, PyrPixelType const * I1, void * bufs);
void upsampleBuffers(int w, int h, void * srcBuffers, void * dstBuffers);
void setZeroBuffers(int w, int h, void * buffers_);
void tv_flow_cuda(int w, int h, int nIterations, float const lambda, float const mu, void * buffers_);

int main(int argc, char * argv[])
{
   if (argc != 6)
   {
      cerr << "Usage: " << argv[0] << " <left image> <right image> <nIterations> <alpha> <mu>" << endl;
      return -1;
   }

   Image<unsigned char> leftImage, rightImage;
   loadImageFile(argv[1], leftImage);
   loadImageFile(argv[2], rightImage);

   int const nIterations = atoi(argv[3]);
   float const alpha = atof(argv[4]);
   float const mu    = atof(argv[5]);

   int const w = leftImage.width();
   int const h = leftImage.height();

   int const nLevels = 3;
   vector<Image<PyrPixelType> > leftPyr(nLevels), rightPyr(nLevels);
   vector<void *> cudaBuffers(nLevels);

   for (int l = 0; l < nLevels; ++l)
   {
      int const W = w >> l;
      int const H = h >> l;

      cout << "W = " << W << " H = " << H << endl;

      leftPyr[l].resize(W, H, 1);
      rightPyr[l].resize(W, H, 1);

      cudaBuffers[l] = allocateBuffers(W, H);
   }

   buildPyramid(leftImage, leftPyr);
   buildPyramid(rightImage, rightPyr);

   cout << "Done building pyramids." << endl;

   Timer t;

   char name[200];

   for (int k = 0; k < 1; ++k)
   {
      t.start();
      for (int l = nLevels-1; l >= 0; --l)
      {
         int const W = w >> l;
         int const H = h >> l;
         Image<float> u(W, H, 1), v(W, H, 1);

         float const alphaLevel = alpha;
         //float const alphaLevel = alpha * sqrtf((1 << l));
         //float const alphaLevel = alpha * ((1 << l));

         int const nLevelIterations = nIterations;

         if (l == nLevels-1)
            setZeroBuffers(W, H, cudaBuffers[l]);
         else
            upsampleBuffers(W/2, H/2, cudaBuffers[l+1], cudaBuffers[l]);

         computeCoefficients(W, H, alphaLevel, &leftPyr[l](0, 0), &rightPyr[l](0, 0), cudaBuffers[l]);
         //computeCoefficients(W, H, 1.0f, &leftPyr[l](0, 0), &rightPyr[l](0, 0), cudaBuffers[l]);

         tv_flow_cuda(W, H, nLevelIterations, 1.0f, mu, cudaBuffers[l]);
         //tv_flow_cuda(W, H, nLevelIterations, alphaLevel, mu, cudaBuffers[l]);

         getBuffers(W, H, cudaBuffers[l], &u(0, 0), &v(0, 0));

#if 1
         Image<unsigned char> flowImage(W, H, 3);

         //float const scale = 0.125f/2 * (1 << l);
         float const scale = 0.2f * (1 << l);

         sprintf(name, "flow-level-%i.png", l);
         saveImageFile(getVisualImageForFlowField(u, v, scale, true), name);
#endif
      } // end for (l)
      t.stop();
   } // end for (k)

   t.print();

   if (0)
   {
      Image<float> u(w, h, 1), v(w, h, 1);

      getBuffers(w, h, cudaBuffers[0], &u(0, 0), &v(0, 0));

      float const scale = 0.2f;

      saveImageFile(getVisualImageForFlowField(u, v, scale, true), "flow-level-0.png");
   }

   if (0)
   {
      int const w = 128;
      float const maxR = sqrtf(2.0f) * w;

      Image<float> u(w, w, 1), v(w, w, 1);
      for (int y = 0; y < w; ++y)
         for (int x = 0; x < w; ++x)
         {
            u(x, y) = (x - w/2.0f) / maxR;
            v(x, y) = (y - w/2.0f) / maxR;
         }
      saveImageFile(getVisualImageForFlowField(u, v, 2.0f), "flow-colors.png");
      //saveImageFile(makeColorWheelImage(), "flow-wheel.png");
   }

   for (int l = 0; l < nLevels; ++l)
      deallocateBuffers(cudaBuffers[l]);

   return 0;
}
