#include "Base/v3d_image.h"
#include "Base/v3d_timer.h"

#include <iostream>
#include <cmath>

using namespace V3D;
using namespace std;

extern void start_tvl1_cuda(int w, int h, float const * f);
extern void finish_tvl1_cuda();
extern void run_denoise_tvl1_cuda(int w, int h, int nIterations, float tau, float lambda,
                                  float * u, float * p1, float * p2, float * q);

namespace
{

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
            E_smooth += fabsf(u_x) + fabsf(u_y);
            //E_smooth += sqrtf(u_x*u_x + u_y*u_y);

            E_data += lambda * fabsf(f(x, y) - u(x, y));
         }
      } // end for (y)
      //cout << "E_data = " << E_data << " E_smooth = " << E_smooth << endl;
      return E_data + E_smooth;
   } // end computePrimalEnergy()

   void
   reprojectP(Image<float>& p, float lambda)
   {
      int const w = p.width();
      int const h = p.height();

      //cout << "excess = ";

      for (int iter = 0; iter < 3; ++iter)
      {
         for (int y = h-1; y >= 0; --y)
            for (int x = w-1; x >= 0; --x)
            {
               float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                            ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

               float const abs_div = fabsf(div);
               if (abs_div > lambda)
               {
                  int affectedPos = 0;
                  if (x < w-1) ++affectedPos;
                  if (x > 0)   ++affectedPos;
                  if (y < h-1) ++affectedPos;
                  if (y > 0)   ++affectedPos;

                  float dp = 0.0f;
                  if (div > 0)
                     dp = (div-lambda)/affectedPos;
                  else
                     dp = (lambda+div)/affectedPos;

                  //dp *= 0.5f;

                  if (x < w-1) p(x, y, 0) -= dp;
                  if (x > 0) p(x-1, y, 0) += dp;
                  if (y < h-1) p(x, y, 1) -= dp;
                  if (y > 0) p(x, y-1, 0) += dp;
               }
            }

//          double excess = 0.0;
//          for (int y = 0; y < h; ++y)
//             for (int x = 0; x < w; ++x)
//             {
//                float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
//                             ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

//                excess += std::max(0.0f, fabsf(div) - lambda);
//             }
//          cout << excess << " ";
      } // end for (iter)
      //cout << endl;
   } // end reprojectP()

   double
   computeDualEnergy(Image<float> const& f, Image<float>& p, Image<float> const& q, float lambda)
   {
      double E = 0.0;
      double E_diff = 0.0;

      int const w = f.width();
      int const h = f.height();

      reprojectP(p, lambda);

//       Image<float> diffIm(w, h, 1, 0.0f);

//       float maxDiv = -1e30;
//       for (int y = 0; y < h; ++y)
//          for (int x = 0; x < w; ++x)
//          {
//             float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
//                          ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));
//             diffIm(x, y) = std::max(0.0f, fabsf(div) - lambda);
//             maxDiv = std::max(maxDiv, fabsf(div));
//          }

// //       double multiplier = 1.0;
// //       if (maxDiv > lambda) multiplier = lambda/maxDiv;
//       cout << "maxDiv = " << maxDiv << endl;

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                         ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

            //div = std::max(-lambda, std::min(lambda, div));
            div = std::max<float>(-lambda, div);
            //div *= multiplier;

            E_diff += fabsf(q(x, y) - div);

            //diffIm(x, y) = fabsf(q(x, y) - div);

            E += -div * f(x, y);
            //E += -div * f(x, y) - fabsf(q(x, y)-div) / lambda;
            //E += -q(x, y) * f(x, y);
         }

      //cout << "E_diff = " << E_diff << endl;

      //saveFloatImage(diffIm, 0, 0.0f, 2*lambda, "diff.jpg");

      return E;
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
   int const nIterations = 1000;
   int const maxIterations = 10000;

   Image<unsigned char> finalImage(w, h, nChannels);

   for (int chan = 0; chan < nChannels; ++chan)
   {
      Image<float> f(w, h, 1);
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            f(x, y) = float(srcImage(x, y, chan));

      start_tvl1_cuda(w, h, &f(0, 0));

      Image<float> u(w, h, 1); // Denoised image
      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            u(x, y) = f(x, y);

      Image<float> p(w, h, 2, 0.0f);
      Image<float> q(w, h, 1, 0.0f);

      double E_primal = computePrimalEnergy(f, u, lambda);
      double E_dual = computeDualEnergy(f, p, q, lambda);

      double curGap = (E_primal-E_dual) / E_primal;

      cout << "E_primal = " << E_primal << " E_dual = " << E_dual << " gap = "
           << E_primal-E_dual << endl;

      Timer t;
      t.start();

      int iter = 0;
      while (curGap > relGap && iter < maxIterations)
         //while (iter < maxIterations)
      {
         run_denoise_tvl1_cuda(w, h, nIterations, tau, lambda, &u(0, 0),
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
      }

      t.stop();
      t.print();

      finish_tvl1_cuda();

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
            finalImage(x, y, chan) = int(std::max<float>(0.0f, std::min<float>(255.0f, u(x, y, 0))));
   }

   saveImageFile(finalImage, "u_tvl1-pd.png");

   return 0;
} // end main()
