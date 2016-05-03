// -*- C++ -*-
#ifndef V3D_GPU_BINARY_SEGMENTATION_H
#define V3D_GPU_BINARY_SEGMENTATION_H

# if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "v3d_gpubase.h"

namespace V3D_GPU
{

   struct BinarySegmentationBase
   {
         BinarySegmentationBase()
            : _width(-1), _height(-1), _tau_primal(0.5f), _tau_dual(0.5f)
         { }

         void setTimesteps(float tau_primal, float tau_dual)
         {
            _tau_primal = tau_primal;
            _tau_dual = tau_dual;
         }

      protected:
         int _width, _height;
         float _tau_primal, _tau_dual;
   }; // end struct BinarySegmentationBase

   struct BinarySegmentationUzawa : public BinarySegmentationBase
   {
         BinarySegmentationUzawa(bool const useHighPrecision = false)
         {
            char const * uTexSpec = useHighPrecision ? "r=32f enableTextureRG" : "r=16f enableTextureRG";
            char const * pTexSpec = useHighPrecision ? "rg=32f enableTextureRG" : "rg=16f enableTextureRG";

            for (int i = 0; i < 2; ++i)
            {
               _uBuffer[i] = new RTT_Buffer(uTexSpec, "BinarySegmentationUzawa::uBuffer");
               _pBuffer[i] = new RTT_Buffer(pTexSpec, "BinarySegmentationUzawa::pBuffer");
            }
         }

         ~BinarySegmentationUzawa()
         {
            delete _uBuffer[0];
            delete _uBuffer[1];
            delete _pBuffer[0];
            delete _pBuffer[1];
         }

         void allocate(int w, int h);
         void deallocate();

         void initializeIterations();
         void setInitialValue(float const * u0)
         {
            _uBuffer[1]->getTexture().overwriteWith(u0, 1);
         }
         void iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBuffer[1]; }

      protected:
         RTT_Buffer * _uBuffer[2]; // Two buffers/textures for ping-pong rendering
         RTT_Buffer * _pBuffer[2]; // ditto
   }; // end struct BinarySegmentationUzawa

   struct BinarySegmentationPD : public BinarySegmentationBase
   {
         BinarySegmentationPD(bool const useHighPrecision = false)
         {
            char const * uTexSpec = useHighPrecision ? "rg=32f enableTextureRG" : "rg=16f enableTextureRG";
            char const * pTexSpec = useHighPrecision ? "rg=32f enableTextureRG" : "rg=16f enableTextureRG";

            for (int i = 0; i < 2; ++i)
            {
               _uBuffer[i] = new RTT_Buffer(uTexSpec, "BinarySegmentationPD::uBuffer");
               _pBuffer[i] = new RTT_Buffer(pTexSpec, "BinarySegmentationPD::pBuffer");
            }
         }

         void allocate(int w, int h);
         void deallocate();

         void initializeIterations();
         void setInitialValue(float const * u0)
         {
            _uBuffer[1]->getTexture().overwriteWith(u0, 1);
         }
         void iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBuffer[1]; }

      protected:
         RTT_Buffer * _uBuffer[2]; // Two buffers/textures for ping-pong rendering
         RTT_Buffer * _pBuffer[2]; // ditto
   }; // end struct BinarySegmentationPD

   // Note: this has very slow convergence!
   struct BinarySegmentationFwBw : public BinarySegmentationBase
   {
         BinarySegmentationFwBw(bool const useHighPrecision = false)
            : epsilon(0.05f), tau(0.24f)
         {
            char const * uTexSpec = useHighPrecision ? "r=32f enableTextureRG" : "r=16f enableTextureRG";

            for (int i = 0; i < 2; ++i)
               _uBuffer[i] = new RTT_Buffer(uTexSpec, "BinarySegmentationFwBw::uBuffer");
         }

         void allocate(int w, int h);
         void deallocate();

         void initializeIterations();
         void setInitialValue(float const * u0)
         {
            _uBuffer[1]->getTexture().overwriteWith(u0, 1);
         }
         void iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBuffer[1]; }

         float epsilon, tau;

      protected:
         RTT_Buffer * _uBuffer[2]; // Two buffers/textures for ping-pong rendering
   }; // end struct BinarySegmentationFwBw

   // This one used the quadratic relaxation approach
   struct BinarySegmentationRelaxed : public BinarySegmentationBase
   {
         BinarySegmentationRelaxed(bool const useHighPrecision = false)
            : tau(0.24f), theta(0.1f)
         {
            char const * uTexSpec = useHighPrecision ? "r=32f enableTextureRG" : "r=16f enableTextureRG";
            char const * pTexSpec = useHighPrecision ? "rg=32f enableTextureRG" : "rg=16f enableTextureRG";

            for (int i = 0; i < 2; ++i)
            {
               _uBuffer[i] = new RTT_Buffer(uTexSpec, "BinarySegmentationUzawa::uBuffer");
               _pBuffer[i] = new RTT_Buffer(pTexSpec, "BinarySegmentationUzawa::pBuffer");
            }
         }

         void allocate(int w, int h);
         void deallocate();

         void initializeIterations();
         void setInitialValue(float const * u0)
         {
            _uBuffer[1]->getTexture().overwriteWith(u0, 1);
         }
         void iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations);

         RTT_Buffer& getResultBuffer() { return *_uBuffer[1]; }

         float tau, theta;

      protected:
         RTT_Buffer * _uBuffer[2]; // Two buffers/textures for ping-pong rendering
         RTT_Buffer * _pBuffer[2]; // ditto
   }; // end struct BinarySegmentationRelaxed

} // end namespace V3D_GPU

# endif // defined(V3DLIB_GPGPU_ENABLE_CG)

#endif // !defined(V3D_GPU_BINARY_SEGMENTATION_H)
