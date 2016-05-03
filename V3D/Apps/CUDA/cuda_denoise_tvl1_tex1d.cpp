#include "Base/v3d_image.h"
#include "Base/v3d_timer.h"

#include <iostream>
#include <cmath>

using namespace V3D;
using namespace std;

extern void start_tvl1_cuda(int w, int h, float const * f);
extern void finish_tvl1_cuda();
extern void run_denoise_tvl1_cuda(int w, int h, int nIterations, float tau, float lambda, float maxU,
                                  float * u, float * p1, float * p2, float * q);

#define NORMALIZE_F 1
#define NORMALIZATION_FACTOR 8.0f
//#define NORMALIZATION_FACTOR 0.25f
//#define NORMALIZATION_FACTOR 1.0f

namespace
{

   inline float
   uniformRand()
   {
      return (rand() / (RAND_MAX + 1.0f));
   }

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
            minVal = std::min(minVal, im(x, y, channel));
            maxVal = std::max(maxVal, im(x, y, channel));
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
   computePrimalEnergy(Image<float> const& f, Image<float> const& u, float lambda)
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
#if 1
            //E_smooth += fabsf(u_x) + fabsf(u_y);
            E_smooth += sqrtf(u_x*u_x + u_y*u_y);
            E_data += lambda * fabsf(f(x, y) - u(x, y));
#else
            E_smooth += sqrtf(u_x*u_x + u_y*u_y) / lambda;
            E_data += fabsf(f(x, y) - u(x, y));
#endif
         }
      } // end for (y)
      //cout << "E_data = " << E_data << " E_smooth = " << E_smooth << endl;
      return (E_data + E_smooth) * NORMALIZATION_FACTOR;
   } // end computePrimalEnergy()

   double
   computeDualEnergy(Image<float> const& f, Image<float>& p, Image<float> const& q, float lambda)
   {
      double E = 0.0;
      double E_diff = 0.0;

      int const w = f.width();
      int const h = f.height();
#if !defined(NORMALIZE_F)
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float const z = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                             ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

            if (z > lambda)
            {
               E -= (255.0f*(z-lambda)+lambda*f(x, y));
               //E -= z*f(x, y);
            }
            else if (z < -lambda)
            {
               E += (lambda*f(x, y));
            }
            else
            {
               E -= z*f(x, y);
            }
         }
#else
      //lambda = 1.0f;
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float const z = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                             ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

            if (z > lambda)
               E -= (255.0f/NORMALIZATION_FACTOR)*(z-lambda)+lambda*f(x, y);
            else if (z < -lambda)
               E -= -lambda*f(x, y);
            else
               E -= z*f(x, y);
         }
#endif

//       double E_div = 0.0f;
//       Image<float> im(w, h, 1);
//       Image<float> divIm(w, h, 1);

//       for (int y = 0; y < h; ++y)
//          for (int x = 0; x < w; ++x)
//          {
//             float const z = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
//                              ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));
//             divIm(x, y) = z;
//             im(x, y) = fabs(z - q(x, y));
//             E_div += fabs(z - q(x, y));
//          }
//       cout << "E_div = " << E_div << endl;

//       saveFloatImage(im, 0, "E_div.png");
//       saveFloatImage(q, 0, -lambda, lambda, "q.png");
//       saveFloatImage(divIm, 0, "div_p.png");

      return NORMALIZATION_FACTOR * E;
   } // end computeDualEnergy()

} // end namespace <>

int main(int argc, char * argv[])
{
   if (argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <image> <lambda> <tau> <rel. gap (percent)>" << endl;
      return 1;
   }

   Image<unsigned char> srcImage;
   loadImageFile(argv[1], srcImage);

   int const w = srcImage.width();
   int const h = srcImage.height();

   int const nChannels = srcImage.numChannels();

   float  const lambda = atof(argv[2]);
   //double const dualityGap = atof(argv[3]);
   float const tau = atof(argv[3]);
   float const relGap = atof(argv[4]) / 100.0f;

   //float const tau = 0.5;
   int const nIterations = 100;
   int const maxIterations = 2000;

   Image<unsigned char> finalImage(w, h, nChannels);

   for (int chan = 0; chan < nChannels; ++chan)
   {
      Image<float> f(w, h, 1);
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
#if !defined(NORMALIZE_F)
            f(x, y) = float(srcImage(x, y, chan));
#else
            f(x, y) = float(srcImage(x, y, chan)) / NORMALIZATION_FACTOR;
#endif
         }

#if !defined(NORMALIZE_F)
      float const maxU = 255.0f;
#else
      float const maxU = 255.0f / NORMALIZATION_FACTOR;
#endif

      start_tvl1_cuda(w, h, &f(0, 0));

      Image<float> u(w, h, 1, 0.0f); // Denoised image
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            //u(x, y) = f(x, y);
            u(x, y) = 255.0f * uniformRand() / NORMALIZATION_FACTOR;
         }

      Image<float> p(w, h, 2, 0.0f);
      Image<float> q(w, h, 1, 0.0f);

#if 0
      for (int y = 0; y < h; ++y)
      {
         int const Y1 = (y < h-1) ? (y+1) : (h-1);
         for (int x = 0; x < w; ++x)
         {
            int const X1 = (x < w-1) ? (x+1) : (w-1);
            float f_x = f(X1, y, 0) - f(x, y, 0);
            float f_y = f(x, Y1, 0) - f(x, y, 0);

            p(x, y, 0) = std::max(-1.0f, std::min(1.0f, f_x));
            p(x, y, 1) = std::max(-1.0f, std::min(1.0f, f_y));
         }
      }
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                         ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));
            q(x, y) = std::max(-lambda, std::min(lambda, div));
            
         }
#endif

      double E_primal = computePrimalEnergy(f, u, lambda);
      double E_dual = computeDualEnergy(f, p, q, lambda);

      double curGap = (E_primal-E_dual) / E_primal;

      cout << "E_primal = " << E_primal << " E_dual = " << E_dual << " gap = "
           << E_primal-E_dual << endl;

      cerr << "0 " << E_primal << " " << E_dual << " " << 100.0 * curGap << endl;

      Timer t;
      t.start();

      int iter = 0;
      while (curGap > relGap && iter < maxIterations)
         //while (iter < maxIterations)
      {
         run_denoise_tvl1_cuda(w, h, nIterations, tau, lambda, maxU, &u(0, 0),
                               &p(0, 0, 0), &p(0, 0, 1), &q(0, 0));

         iter += nIterations;
         cout << "iter: " << iter << endl;

//       float min_u = 1e30;
//       float max_u = -1e30;
//       for (int y = 0; y < h; ++y)
//          for (int x = 0; x < w; ++x)
//          {
//             min_u = std::min(min_u, u(x, y));
//             max_u = std::max(max_u, u(x, y));
//          }

//       cout << "min_u = " << min_u << " max_u = " << max_u << endl;

         E_primal = computePrimalEnergy(f, u, lambda);
         E_dual = computeDualEnergy(f, p, q, lambda);

         curGap = (E_primal-E_dual) / E_primal;

         cout << "E_primal = " << E_primal << " E_dual = " << E_dual << " gap = "
              << E_primal-E_dual << " rel. gap = " << 100.0 * curGap << "%" << endl;

         cerr << iter << " " << E_primal << " " << E_dual << " " << 100.0 * curGap << endl;
      }

      t.stop();
      t.print();

      finish_tvl1_cuda();

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
#if !defined(NORMALIZE_F)
            finalImage(x, y, chan) = int(std::max(0.0f, std::min(255.0f, u(x, y, 0))));
#else
            finalImage(x, y, chan) = int(std::max(0.0f, std::min(255.0f, NORMALIZATION_FACTOR * u(x, y, 0))));
#endif
         }
   } // end for (chan)

   saveImageFile(finalImage, "u_tvl1-pd-2.png");

   if (0)
   {
      cout << "Generating edge image..." << endl;

      Image<float> edgeImage(w, h, 1, 0.0f);

      for (int y = 0; y < h; ++y)
      {
         int const y0 = (y > 0) ? y-1 : y+1;
         int const y1 = (y < h-1) ? y+1 : y-1;

         for (int x = 0; x < w; ++x)
         {
            int const x0 = (x > 0) ? x-1 : x+1;
            int const x1 = (x < w-1) ? x+1 : x-1;

            float len = 0;
            for (int ch = 0; ch < finalImage.numChannels(); ++ch)
            {
#if 0
               float Ix = 0.5f*(finalImage(x1, y, ch) - finalImage(x0, y, ch));
               float Iy = 0.5f*(finalImage(x, y1, ch) - finalImage(x, y0, ch));
#else
               float Ix = finalImage(x1, y, ch) - finalImage(x, y, ch);
               float Iy = finalImage(x, y1, ch) - finalImage(x, y, ch);
#endif

               len += sqrtf(Ix*Ix + Iy*Iy);
            }
            edgeImage(x, y) = -len;
         }
      } // end for (y)
      saveFloatImage(edgeImage, 0, "edges.ppm");

//       for (int y = 0; y < h; ++y)
//          for (int x = 0; x < w; ++x)
//             edgeImage(x, y) = (fabsf(edgeImage(x, y)) > 20) ? 0.0f : 1.0f;
//       saveFloatImage(edgeImage, 0, "edges-binary.ppm");
   } // end if

   return 0;
} // end main()
