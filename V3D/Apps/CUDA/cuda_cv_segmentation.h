// -*- C++ -*-
#ifndef V3D_CUDA_CV_SEGMENTATION_H
#define V3D_CUDA_CV_SEGMENTATION_H

struct CUDA_CV_Segmentation
{
      void allocate(int w, int h);
      void deallocate();

      void setImageData(float const * fSrc);
      void initSegmentation();
      void getResult(float * uDst, float * p1Dst = 0, float * p2Dst = 0);

      void runGeneralSegmentation(int nIterations, float alpha, float tau);
      void runSegmentation(float c1, float c2, int nIterations, float alpha, float tau);
      void runSegmentation_theta(float c1, float c2, int nIterations, float alpha, float tau, float theta);

      void updateMeans(float& c1, float& c2);

   private:
      int _w, _h;

      float *_d_u, *_d_p1, *_d_p2;
      float *_d_g;
      float *_d_c1, *_d_usum;

      float _fSum;
}; // end struct CUDA_CV_Segmentation

#endif
