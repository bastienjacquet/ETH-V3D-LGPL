// -*- C++ -*-

#ifndef V3D_GPU_VARIATIONAL_STEREO_H
#define V3D_GPU_VARIATIONAL_STEREO_H

# if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "GL/v3d_gpubase.h"
#include "GL/v3d_gpupyramid.h"
#include "Math/v3d_linear.h"

namespace V3D_GPU
{

   struct Variational_L1_StereoBase
   {
         Variational_L1_StereoBase(int nLevels)
            : _warpedBufferHighPrecision(true),
              _uvBufferHighPrecision(true),
              _pBufferHighPrecision(false), // fp16 is usually enough for p
              _nOuterIterations(1), _nInnerIterations(50), _startLevel(0), _nLevels(nLevels),
              _width(-1), _height(-1)
         { }

         void setLambda(float lambda)        { _lambda = lambda; }
         void setOuterIterations(int nIters) { _nOuterIterations = nIters; }
         void setInnerIterations(int nIters) { _nInnerIterations = nIters; }
         void setStartLevel(int startLevel)  { _startLevel = startLevel; }

         // Must be called before allocate() to have an effect.
         void configurePrecision(bool warpedBufferHighPrecision,
                                 bool uvBufferHighPrecision,
                                 bool pBufferHighPrecision)
         {
            _warpedBufferHighPrecision = warpedBufferHighPrecision;
            _uvBufferHighPrecision     = uvBufferHighPrecision;
            _pBufferHighPrecision      = pBufferHighPrecision;
         }

         void allocate(int w, int h);
         void deallocate();

         RTT_Buffer * getWarpedBuffer(int level) { return _warpedBufferPyramid[level]; }

      protected:
         bool _warpedBufferHighPrecision, _uvBufferHighPrecision, _pBufferHighPrecision;

         std::vector<RTT_Buffer *> _warpedBufferPyramid;

         int _nOuterIterations, _nInnerIterations;

         float _lambda;
         int _startLevel, _nLevels;
         int _width, _height;
   }; // end struct Variational_L1_StereoBase

//----------------------------------------------------------------------

   struct Structure_L1_Stereo_PD : public Variational_L1_StereoBase
   {
      public:
         struct Config
         {
               Config(float tau_primal = 0.7f, float tau_dual = 0.7f,
                      float eta = 1.0f, float mu = 1.0f)
                  : _tau_primal(tau_primal), _tau_dual(tau_dual), _eta(eta), _mu(mu)
               { }

               float _tau_primal, _tau_dual, _eta, _mu;
         };

         Structure_L1_Stereo_PD(int nLevels)
            : Variational_L1_StereoBase(nLevels)
         {
            _shader_u = _shader_alpha = 0;
            _shader_p = _shader_q = 0;

            _dictH[0] = -1.0f; _dictH[1] = -1.0f/3; _dictH[2] = 1.0f/3; _dictH[3] = 1.0f;
            _dictV = _dictH;
         }

         ~Structure_L1_Stereo_PD() { }

         void configure(Config const& cfg) { _cfg = cfg; }
         void setDictionaries(V3D::Vector4f const& dictH, V3D::Vector4f const& dictV)
         {
            _dictH = dictH;
            _dictV = dictV;
         }

         void allocate(int w, int h);
         void deallocate();

         void run(unsigned int I0_TexID, unsigned int I1_TexID);

         unsigned int getDisparityTextureID() { return _uBuffer2Pyramid[_startLevel]->textureID(); }
         unsigned int getAlphaTextureID()     { return _alphaTex2Pyramid[_startLevel]->textureID(); }

         unsigned int getDisparityTextureID(int const level) { return _uBuffer2Pyramid[level]->textureID(); }

      protected:
         Config _cfg;

         V3D::Vector4f _dictH, _dictV;

         Cg_FragmentProgram * _shader_u;
         Cg_FragmentProgram * _shader_alpha;
         Cg_FragmentProgram * _shader_p;
         Cg_FragmentProgram * _shader_q;

         std::vector<RTT_Buffer *> _uBuffer1Pyramid, _uBuffer2Pyramid;

         std::vector<ImageTexture2D *> _alphaTex1Pyramid, _alphaTex2Pyramid;

         // We do not need to double the beta buffers, since there is no ping-pong rendering applied on those
         std::vector<ImageTexture2D *> _betaTexPyramid;
         std::vector<FrameBufferObject *> _alphaFbo1Pyramid, _alphaFbo2Pyramid;

         std::vector<RTT_Buffer *> _pBuffer1Pyramid, _pBuffer2Pyramid;
         std::vector<RTT_Buffer *> _qhBuffer1Pyramid, _qhBuffer2Pyramid;
         std::vector<RTT_Buffer *> _qvBuffer1Pyramid, _qvBuffer2Pyramid;
   }; // end struct Structure_L1_Stereo_PD

//----------------------------------------------------------------------

   void warpImageWithDisparities(unsigned int u_tex, unsigned int I0_tex,
                                 unsigned int I1_tex, int level, RTT_Buffer& dest);

} // end namespace V3D_GPU

# endif

#endif // defined(V3D_GPU_VARIATIONAL_STEREO_H)
