#include "Base/v3d_image.h"
#include "Base/v3d_timer.h"

#include <iostream>
#include <cmath>

using namespace V3D;
using namespace std;

extern void start_tvl2_cuda(int w, int h, float const * f);
extern void finish_tvl2_cuda();
extern void run_denoise_tvl2_cuda(int w, int h, int nIterations, float tau, float lambda,
                                  float * u, float * p1, float * p2);

namespace
{

   template <typename T>
   inline T sqr(T x) { return x*x; }

   double
   computePrimalEnergy(Image<float> const& f, Image<float> const& u, float lambda)
   {
      double const lambda_2 = lambda/2.0;

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
            //E += sqrtf(u_x*u_x + u_y*u_y);

            E_data += lambda_2 * sqr(f(x, y) - u(x, y));
         }
      } // end for (y)
      //cout << "E_data = " << E_data << " E_smooth = " << E_smooth << endl;
      return E_data + E_smooth;
   } // end computePrimalEnergy()

   double
   computeDualEnergy(Image<float> const& f, Image<float> const& p, float lambda)
   {
      double const theta = 1.0 / (2*lambda);

      double E = 0.0;

      int const w = f.width();
      int const h = f.height();

      for (int y = 0; y < h; ++y)
         for (int x = 0; x < w; ++x)
         {
            float div = (((x < w-1) ? p(x, y, 0) : 0) - ((x > 0) ? p(x-1, y, 0) : 0) +
                         ((y < h-1) ? p(x, y, 1) : 0) - ((y > 0) ? p(x, y-1, 1) : 0));

            E += -div * f(x, y);
            E += -theta * sqr(div);
         }
      return E;
   } // end computeDualEnergy()

} // end namespace <>

int main(int argc, char * argv[])
{
   if (argc != 4)
   {
      cerr << "Usage: " << argv[0] << " <image> <lambda> <tau>" << endl;
      return 1;
   }

   Image<unsigned char> srcImage;
   loadImageFile(argv[1], srcImage);

   int const w = srcImage.width();
   int const h = srcImage.height();

   if (srcImage.numChannels() != 1)
   {
      cerr << "Warning: grayscale image expected; using only red channel." << endl;
   }

   float  const lambda = atof(argv[2]);
   //double const dualityGap = atof(argv[3]);
   float const tau = atof(argv[3]);

   //float const tau = 0.5;
   int const nIterations = 2000;
   int const maxIterations = 2000;

   Image<float> f(w, h, 1);
   for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
         f(x, y) = float(srcImage(x, y));

   start_tvl2_cuda(w, h, &f(0, 0));

   Image<float> u(w, h, 1); // Denoised image
   for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
         u(x, y) = f(x, y);

   Image<float> p(w, h, 2, 0.0f);

   double E_primal = computePrimalEnergy(f, u, lambda);
   double E_dual = computeDualEnergy(f, p, lambda);

   //double const maxGap = E_primal * dualityGap;

   cout << "E_primal = " << E_primal << " E_dual = " << E_dual << " gap = "
        << E_primal-E_dual << endl;

   int iter = 0;

   Timer t;
   t.start();

   //while (E_primal-E_dual > maxGap && iter < maxIterations)
   while (iter < maxIterations)
   {
      run_denoise_tvl2_cuda(w, h, nIterations, tau, lambda, &u(0, 0),
                            &p(0, 0, 0), &p(0, 0, 1));

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
      E_dual = computeDualEnergy(f, p, lambda);
      cout << "E_primal = " << E_primal << " E_dual = " << E_dual << " gap = "
           << E_primal-E_dual << " rel. gap = " << 100.0 * (E_primal-E_dual) / E_primal << "%" << endl;

      iter += nIterations;
   }
   t.stop();

   finish_tvl2_cuda();

   t.print();

   Image<unsigned char> finalImage(w, h, 1);
   for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
         finalImage(x, y, 0) = int(std::max<float>(0.0f, std::min<float>(255.0f, u(x, y, 0))));
   saveImageFile(finalImage, "u_tvl2-gpu.pgm");

   return 0;
} // end main()
