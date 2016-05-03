#include "cuda_cv_segmentation.h"

#include "Base/v3d_image.h"
#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace V3D;

//#define REPORT_ENERGIES 1

namespace
{

   template <typename T>
   inline T sqr(T x) { return x*x; }

   void
   saveFloatImage(Image<float> const& im, int channel, char const * name)
   {
      int const w = im.width();
      int const h = im.height();

      float minVal = 1e30;
      float maxVal = -1e30;

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            minVal = std::min<float>(minVal, im(x, y, channel));
            maxVal = std::max<float>(maxVal, im(x, y, channel));
         }

      //cout << "maxVal = " << maxVal << endl;

      float const len = maxVal - minVal;
      //cout << "len = " << len << endl;

      Image<unsigned char> byteIm(w, h, 1);

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            byteIm(x, y) = int(255.0f * (im(x, y, channel) - minVal) / len);

      saveImageFile(byteIm, name);
   } // end saveFloatImage()

   void
   saveFloatImage(Image<float> const& im, int channel, float minVal, float maxVal, char const * name)
   {
      int const w = im.width();
      int const h = im.height();

      float const len = maxVal - minVal;
      //cout << "len = " << len << endl;

      Image<unsigned char> byteIm(w, h, 1);

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            byteIm(x, y) = int(255.0f * (im(x, y, channel) - minVal) / len);

      saveImageFile(byteIm, name);
   } // end saveFloatImage()

   double
   computePrimalEnergy(float c1, float c2, float alpha, Image<float> const& f, Image<float> const& u)
   {
      double E_data = 0.0, E_smooth = 0.0;

      int const w = f.width();
      int const h = f.height();

      for (int y = 0; y < h; ++y)
      {
         int const Y1 = (y < h-1) ? (y+1) : (h-1);
         for (int x = 0; x < w; ++x)
         {
            int const X1 = (x < w-1) ? (x+1) : (w-1);
            float u_x = u(X1, y, 0) - u(x, y, 0);
            float u_y = u(x, Y1, 0) - u(x, y, 0);
            //E_smooth += fabsf(u_x) + fabsf(u_y);
            E_smooth += sqrtf(u_x*u_x + u_y*u_y);

            //E_data += alpha * (u(x, y) * sqr(f(x, y) - c1) + (1-u(x, y)) * sqr(f(x, y) - c2));
            E_data += alpha * ((u(x, y)) * sqr(f(x, y) - c1) + (-u(x, y)) * sqr(f(x, y) - c2));
         }
      } // end for (y)
      //cout << "E_data = " << E_data << " E_smooth = " << E_smooth << endl;
      return E_data + E_smooth;
   } // end computePrimalEnergy()

   double
   computeDualEnergy(float c1, float c2, float alpha, Image<float> const& f, Image<float> const& p)
   {
      double E = 0.0;

      int const w = f.width();
      int const h = f.height();

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                         ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

            float r = alpha * (sqr(f(x, y) - c1) - sqr(f(x, y) - c2));

            div = std::min<float>(0.0f, div + r);

            E += div;
         }
      return E;
   } // end computeDualEnergy()

} // end namespace <>

int main(int argc, char * argv[])
{
   if (argc != 7)
   {
      cerr << "Usage: " << argv[0] << " <input image> <c1> <c2> <nRounds> <nIterations> <alpha>" << endl;
      return -1;
   }

   Image<unsigned char> srcImage;
   loadImageFile(argv[1], srcImage);

   float c1 = atof(argv[2])/255.0f;
   float c2 = atof(argv[3])/255.0f;

   int const nRounds = atoi(argv[4]);
   int const nIterations = atoi(argv[5]);
   float const alpha = atof(argv[6]);

   int const w = srcImage.width();
   int const h = srcImage.height();

   Image<float> f(w, h, 1);
   for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
         f(x, y) = srcImage(x, y) / 255.0f;

   Image<float> g(w, h, 1);
   for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
         g(x, y) = alpha * (sqr(c1 - f(x, y)) - sqr(c2 - f(x, y)));

   Image<float> u(w, h, 1);
   Image<float> p(w, h, 2);

   CUDA_CV_Segmentation cvSegmentation;
   cvSegmentation.allocate(w, h);

   Timer t;
   t.start();

   float c1_0 = c1;
   float c2_0 = c2;

   cvSegmentation.setImageData(&f(0, 0));
   cvSegmentation.initSegmentation();

#if !defined(REPORT_ENERGIES)
   for (int round = 0; round < nRounds; ++round)
   {
# if 1
      float const tau = 0.7f;
      //cuda_segmentation_CV(w, h, nIterations, 1.0f, tau, &g(0, 0), cudaBuffers, &u(0, 0));
      cvSegmentation.runSegmentation(c1, c2, nIterations, alpha, tau);
# else
      float const tau = 0.249f;
      float const theta = 0.05f;
      cvSegmentation.runSegmentation_theta(c1, c2, nIterations, alpha, tau, theta);
# endif
      cvSegmentation.updateMeans(c1, c2);
      cout << "c1 = " << c1 << ", c2 = " << c2 << endl;
   } // end for (round)
#else
   for (int round = 0; round < nRounds; ++round)
   {
# if 0
      float const tau = 0.7f;
      //cuda_segmentation_CV(w, h, nIterations, 1.0f, tau, &g(0, 0), cudaBuffers, &u(0, 0));
      cvSegmentation.runSegmentation(c1, c2, nIterations, alpha, tau);
# else
      float const tau = 0.249f;
      float const theta = 0.02f;
      cvSegmentation.runSegmentation_theta(c1, c2, nIterations, alpha, tau, theta);
# endif
      cvSegmentation.getResult(&u(0, 0), &p(0, 0, 0), &p(0, 0, 1));
      //cout << "E_primal = " << computePrimalEnergy(c1_0, c2_0, alpha, f, u) << endl;
      //cout << "E_dual = " << computeDualEnergy(c1_0, c2_0, alpha, f, p) << endl;
      cout << (round+1)*nIterations << " " << computePrimalEnergy(c1_0, c2_0, alpha, f, u)
           << " " << computeDualEnergy(c1_0, c2_0, alpha, f, p) << endl;
   } // end for (round)
#endif

   t.stop();
   t.print();

#if !defined(REPORT_ENERGIES)
   cout << "c1 = " << c1 << ", c2 = " << c2 << endl;
   cvSegmentation.getResult(&u(0, 0), &p(0, 0, 0), &p(0, 0, 1));
   cout << "E_primal = " << computePrimalEnergy(c1_0, c2_0, alpha, f, u) << endl;
   cout << "E_dual = " << computeDualEnergy(c1_0, c2_0, alpha, f, p) << endl;
#endif

   cvSegmentation.deallocate();

   saveFloatImage(u, 0, 0.0f, 1.0f, "u_segmentation.png");

   return 0;
}
