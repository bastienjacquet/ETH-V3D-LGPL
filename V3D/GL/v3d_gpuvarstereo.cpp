#if defined(V3DLIB_GPGPU_ENABLE_CG)

#include "Base/v3d_utilities.h"
#include "GL/v3d_gpuvarstereo.h"
#include "GL/v3d_gpuflow.h"

#include <iostream>
#include <cmath>
#include <GL/glew.h>

using namespace std;
using namespace V3D_GPU;

namespace
{

   void
   upsampleDisparities(unsigned uvSrcTex, unsigned pSrcTex, float pScale,
                       RTT_Buffer& ubuffer, RTT_Buffer& pbuffer)
   {
      static Cg_FragmentProgram * upsampleShader = 0;

      if (upsampleShader == 0)
      {
         upsampleShader = new Cg_FragmentProgram("v3d_gpuflow::upsampleDisparities::upsampleShader");

         char const * source =
            "void main(uniform sampler2D src_tex : TEXTURE0, \n"
            "                  float2 st0 : TEXCOORD0, \n"
            "                  float4 st3 : TEXCOORD3, \n"
            "              out float4 res_out : COLOR0) \n"
            "{ \n"
            "   res_out = st3 * tex2D(src_tex, st0); \n"
            "} \n";
         upsampleShader->setProgram(source);
         upsampleShader->compile();
         checkGLErrorsHere0();
      } // end if

      setupNormalizedProjection();
      ubuffer.activate();
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, uvSrcTex);
      glEnable(GL_TEXTURE_2D);
      upsampleShader->enable();
      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      glMultiTexCoord4f(GL_TEXTURE3_ARB, 2, 2, 1, 1);
      //glMultiTexCoord4f(GL_TEXTURE3_ARB, 0, 0, 0, 0);
      renderNormalizedQuad();
      //upsampleShader->disable();
      glDisable(GL_TEXTURE_2D);
      checkGLErrorsHere0();

      pbuffer.activate();
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, pSrcTex);
      glEnable(GL_TEXTURE_2D);
      //upsampleShader->enable();
      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      glMultiTexCoord4f(GL_TEXTURE3_ARB, pScale, pScale, pScale, pScale);
      renderNormalizedQuad();
      upsampleShader->disable();
      glDisable(GL_TEXTURE_2D);
      checkGLErrorsHere0();
   } // upsampleDisparities()

   void
   upsampleBuffer(unsigned srcTex, float scale, FrameBufferObject& dstFbo)
   {
      static Cg_FragmentProgram * upsampleShader = 0;

      if (upsampleShader == 0)
      {
         upsampleShader = new Cg_FragmentProgram("v3d_gpuflow::upsampleBuffer::upsampleShader");

         char const * source =
            "void main(uniform sampler2D src_tex : TEXTURE0, \n"
            "                  float2 st0 : TEXCOORD0, \n"
            "                  float4 st3 : TEXCOORD3, \n"
            "              out float4 res_out : COLOR0) \n"
            "{ \n"
            "   res_out = st3 * tex2D(src_tex, st0); \n"
            "} \n";
         upsampleShader->setProgram(source);
         upsampleShader->compile();
         checkGLErrorsHere0();
      } // end if

      setupNormalizedProjection();
      dstFbo.activate();
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, srcTex);
      glEnable(GL_TEXTURE_2D);
      upsampleShader->enable();
      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      glMultiTexCoord4f(GL_TEXTURE3_ARB, scale, scale, scale, scale);
      renderNormalizedQuad();
      upsampleShader->disable();
      glDisable(GL_TEXTURE_2D);
      checkGLErrorsHere0();
   } // upsampleBuffer()

   void
   upsampleBuffer(unsigned srcTex, V3D::Vector4f const& scale, FrameBufferObject& dstFbo)
   {
      static Cg_FragmentProgram * upsampleShader = 0;

      if (upsampleShader == 0)
      {
         upsampleShader = new Cg_FragmentProgram("v3d_gpuflow::upsampleBuffer::upsampleShader");

         char const * source =
            "void main(uniform sampler2D src_tex : TEXTURE0, \n"
            "                  float2 st0 : TEXCOORD0, \n"
            "                  float4 st3 : TEXCOORD3, \n"
            "              out float4 res_out : COLOR0) \n"
            "{ \n"
            "   res_out = st3 * tex2D(src_tex, st0); \n"
            "} \n";
         upsampleShader->setProgram(source);
         upsampleShader->compile();
         checkGLErrorsHere0();
      } // end if

      setupNormalizedProjection();
      dstFbo.activate();
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, srcTex);
      glEnable(GL_TEXTURE_2D);
      upsampleShader->enable();
      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      glMultiTexCoord4f(GL_TEXTURE3_ARB, scale[0], scale[1], scale[2], scale[3]);
      renderNormalizedQuad();
      upsampleShader->disable();
      glDisable(GL_TEXTURE_2D);
      checkGLErrorsHere0();
   } // upsampleBuffer()

   void
   upsampleBuffers(unsigned src1Tex, unsigned src2Tex, float scale1, float scale2,
                   FrameBufferObject& dstFbo)
   {
      static Cg_FragmentProgram * upsampleShader = 0;

      if (upsampleShader == 0)
      {
         upsampleShader = new Cg_FragmentProgram("v3d_gpuflow::upsampleBuffer::upsampleShader");

         char const * source =
            "void main(uniform sampler2D src1_tex : TEXTURE0, \n"
            "          uniform sampler2D src2_tex : TEXTURE1, \n"
            "                  float2 st0 : TEXCOORD0, \n"
            "                  float4 st3 : TEXCOORD3, \n"
            "                  float4 st4 : TEXCOORD4, \n"
            "              out float4 res1_out : COLOR0, \n"
            "              out float4 res2_out : COLOR1) \n"
            "{ \n"
            "   res1_out = st3 * tex2D(src1_tex, st0); \n"
            "   res1_out = st4 * tex2D(src2_tex, st0); \n"
            "} \n";
         upsampleShader->setProgram(source);
         upsampleShader->compile();
         checkGLErrorsHere0();
      } // end if

      setupNormalizedProjection();
      dstFbo.activate();

      GLenum const targetBuffers[2] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };
      glDrawBuffersARB(2, targetBuffers);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, src1Tex);
      glEnable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, src2Tex);
      glEnable(GL_TEXTURE_2D);
      upsampleShader->enable();
      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      // Texcoords 0-2 are assigned by renderNormalizedQuad().
      glMultiTexCoord4f(GL_TEXTURE3_ARB, scale1, scale1, scale1, scale1);
      glMultiTexCoord4f(GL_TEXTURE4_ARB, scale2, scale2, scale2, scale2);
      renderNormalizedQuad();
      upsampleShader->disable();
      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE1);
      glDisable(GL_TEXTURE_2D);

      glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

      checkGLErrorsHere0();
   } // upsampleBuffers()

} // end namespace

//----------------------------------------------------------------------

namespace V3D_GPU
{

   void
   Variational_L1_StereoBase::allocate(int W, int H)
   {
      _width = W;
      _height = H;

      char const * texSpec = _warpedBufferHighPrecision ? "rgba=32f tex2D" : "rgba=16f tex2D";

      _warpedBufferPyramid.resize(_nLevels);
      for (int level = 0; level < _nLevels; ++level)
      {
         int const w = _width / (1 << level);
         int const h = _height / (1 << level);

         _warpedBufferPyramid[level] = new RTT_Buffer(texSpec, "_warpedBufferPyramid[]");
         _warpedBufferPyramid[level]->allocate(w, h);
      }
   } // end Variational_L1_StereoBase::allocate()

   void
   Variational_L1_StereoBase::deallocate()
   {
      for (int level = 0; level < _nLevels; ++level)
         _warpedBufferPyramid[level]->deallocate();
   }

//----------------------------------------------------------------------

   void
   Structure_L1_Stereo_PD::allocate(int W, int H)
   {
      Variational_L1_StereoBase::allocate(W, H);

      _shader_u = new Cg_FragmentProgram("struct_l1_stereo_pd_update_u");
      _shader_u->setProgramFromFile("VarStereo/struct_l1_stereo_update_u.cg");
      _shader_u->compile();

      _shader_alpha = new Cg_FragmentProgram("struct_stereo_pd_update_alpha");
      _shader_alpha->setProgramFromFile("VarStereo/struct_l1_stereo_update_alpha.cg");
      _shader_alpha->compile();

      _shader_p = new Cg_FragmentProgram("struct_stereo_pd_update_p");
      //_shader_p->setProgramFromFile("VarStereo/struct_l1_stereo_update_p.cg");
      _shader_p->setProgramFromFile("OpticalFlow/struct_l1_flow_update_p.cg");
      _shader_p->compile();

      _shader_q = new Cg_FragmentProgram("struct_stereo_pd_update_p");
      _shader_q->setProgramFromFile("VarStereo/struct_l1_stereo_update_q.cg");
      _shader_q->compile();

      char const * uTexSpec = _uvBufferHighPrecision ? "rgb=32f tex2D" : "rgb=16f tex2D";
      //char const * alphaTexSpec = _uvBufferHighPrecision ? "rgba=32f tex2D" : "rgba=16f tex2D";
      char const * alphaTexSpec = "rgba=32f tex2D"; // the alphas always have to be full precision
      char const * pTexSpec  = _pBufferHighPrecision ? "rgba=32f tex2D" : "rgba=16f tex2D";
      char const * qTexSpec  = _pBufferHighPrecision ? "rgba=32f tex2D" : "rgba=16f tex2D";

      allocateRTTPyramid(_nLevels, _width, _height, uTexSpec, "ubuffer1", _uBuffer1Pyramid);
      allocateRTTPyramid(_nLevels, _width, _height, uTexSpec, "ubuffer2", _uBuffer2Pyramid);
      allocateTexturePyramid(_nLevels, _width, _height, alphaTexSpec, "alphatex1", _alphaTex1Pyramid);
      allocateTexturePyramid(_nLevels, _width, _height, alphaTexSpec, "alphatex2", _alphaTex2Pyramid);
      allocateTexturePyramid(_nLevels, _width, _height, alphaTexSpec, "betaTex", _betaTexPyramid);

      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "pbuffer1", _pBuffer1Pyramid);
      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "pbuffer2", _pBuffer2Pyramid);

      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "qhbuffer1", _qhBuffer1Pyramid);
      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "qhbuffer2", _qhBuffer2Pyramid);
      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "qvbuffer1", _qvBuffer1Pyramid);
      allocateRTTPyramid(_nLevels, _width, _height, pTexSpec, "qvbuffer2", _qvBuffer2Pyramid);

      _alphaFbo1Pyramid.resize(_nLevels);
      _alphaFbo2Pyramid.resize(_nLevels);

      for (int level = 0; level < _nLevels; ++level)
      {
         int const w = _width / (1 << level);
         int const h = _height / (1 << level);

         _alphaFbo1Pyramid[level] = new FrameBufferObject("alpha1Fbo1");
         _alphaFbo1Pyramid[level]->allocate();
         _alphaFbo2Pyramid[level] = new FrameBufferObject("alpha1Fbo2");
         _alphaFbo2Pyramid[level]->allocate();

         _alphaFbo1Pyramid[level]->makeCurrent();
         _alphaFbo1Pyramid[level]->attachTexture2D(*_alphaTex1Pyramid[level], GL_COLOR_ATTACHMENT0_EXT, 0);
         _alphaFbo1Pyramid[level]->attachTexture2D(*_betaTexPyramid[level], GL_COLOR_ATTACHMENT1_EXT, 0);

         _alphaFbo2Pyramid[level]->makeCurrent();
         _alphaFbo2Pyramid[level]->attachTexture2D(*_alphaTex2Pyramid[level], GL_COLOR_ATTACHMENT0_EXT, 0);
         _alphaFbo2Pyramid[level]->attachTexture2D(*_betaTexPyramid[level], GL_COLOR_ATTACHMENT1_EXT, 0);
      } // end for (level)
   } // end Structure_L1_Stereo_PD::allocate()

   void
   Structure_L1_Stereo_PD::deallocate()
   {
      Variational_L1_StereoBase::deallocate();

      for (int level = 0; level < _nLevels; ++level)
      {
         _alphaFbo1Pyramid[level]->deallocate();
         delete _alphaFbo1Pyramid[level];
         _alphaFbo2Pyramid[level]->deallocate();
         delete _alphaFbo2Pyramid[level];
      }

      deallocateRTTPyramid(_uBuffer1Pyramid);
      deallocateRTTPyramid(_uBuffer2Pyramid);

      deallocateTexturePyramid(_alphaTex1Pyramid);
      deallocateTexturePyramid(_alphaTex2Pyramid);

      deallocateTexturePyramid(_betaTexPyramid);

      deallocateRTTPyramid(_pBuffer1Pyramid);
      deallocateRTTPyramid(_pBuffer2Pyramid);

      deallocateRTTPyramid(_qhBuffer1Pyramid);
      deallocateRTTPyramid(_qhBuffer2Pyramid);
      deallocateRTTPyramid(_qvBuffer1Pyramid);
      deallocateRTTPyramid(_qvBuffer2Pyramid);
   } // end Structure_L1_Stereo_PD::deallocate()

   void
   Structure_L1_Stereo_PD::run(unsigned int I0_TexID, unsigned int I1_TexID)
   {
      for (int level = _nLevels-1; level >= _startLevel; --level)
      {
         RTT_Buffer * &ubuffer1 = _uBuffer1Pyramid[level];
         RTT_Buffer * &ubuffer2 = _uBuffer2Pyramid[level];

         FrameBufferObject * &alphaFbo1 = _alphaFbo1Pyramid[level];
         FrameBufferObject * &alphaFbo2 = _alphaFbo2Pyramid[level];

         RTT_Buffer * &pbuffer1 = _pBuffer1Pyramid[level];
         RTT_Buffer * &pbuffer2 = _pBuffer2Pyramid[level];

         RTT_Buffer * &qhbuffer1 = _qhBuffer1Pyramid[level];
         RTT_Buffer * &qhbuffer2 = _qhBuffer2Pyramid[level];
         RTT_Buffer * &qvbuffer1 = _qvBuffer1Pyramid[level];
         RTT_Buffer * &qvbuffer2 = _qvBuffer2Pyramid[level];

         if (level == _nLevels-1)
         {
            glClearColor(0, 0, 0, 0);

            ubuffer2->activate();
            glClear(GL_COLOR_BUFFER_BIT);

            alphaFbo2->activate();
            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawBuffer(GL_COLOR_ATTACHMENT1_EXT);
            glClear(GL_COLOR_BUFFER_BIT);

            glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
            
         }
         else
         {
            upsampleBuffer(_uBuffer2Pyramid[level+1]->textureID(), 2.0f, ubuffer2->getFBO());
            upsampleBuffer(_alphaFbo2Pyramid[level+1]->getColorTexture(0).textureID(), V3D::Vector4f(2.0f, 1.0f, 2.0f, 1.0f), *alphaFbo2);
         }

         {
            // Clear all dual variables
            glClearColor(0, 0, 0, 0);
            pbuffer2->activate();
            glClear(GL_COLOR_BUFFER_BIT);
            qhbuffer2->activate();
            glClear(GL_COLOR_BUFFER_BIT);
            qvbuffer2->activate();
            glClear(GL_COLOR_BUFFER_BIT);
         } // end scope

         int const w = _width / (1 << level);
         int const h = _height / (1 << level);

         RTT_Buffer& warpedBuffer = *_warpedBufferPyramid[level];

         //float const lambda = _lambda;
         float const lambda = _lambda / sqrtf(1 << level);
         //float const lambda = _lambda / (1 << level);

         float const ds = 1.0f / w;
         float const dt = 1.0f / h;

         float const PatchW = 4.0f;
         float const DictSize = 2.0f;
         float const Dmax = 1.0f;
         float const norm_1 = std::max(2.0f * PatchW, 4.0f + PatchW);
         float const norm_inf = std::max(2.0f, Dmax*DictSize + 1);
         float const tau_dual   = 1.0f;
         float const tau_primal = 0.95f / (tau_dual * norm_1 * norm_inf);

         // For MRT rendering
         GLenum const targetBuffers[2] = { GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT };

         for (int iter = 0; iter < _nOuterIterations; ++iter)
         {
            warpImageWithDisparities(ubuffer2->textureID(), I0_TexID, I1_TexID, level, warpedBuffer);
            //warpImageWithFlowField(ubuffer2->textureID(), I0_TexID, I1_TexID, level, warpedBuffer);

            setupNormalizedProjection();

            _shader_u->parameter("tau", tau_primal);
            _shader_u->parameter("lambda_tau", tau_primal*lambda);
            _shader_u->parameter("dxdy", ds, 0.0f, 0.0f, dt);

            _shader_alpha->parameter("tau", tau_primal);
            _shader_alpha->parameter("dict1h", _dictH[0], _dictH[1], _dictH[2], _dictH[3]);
            _shader_alpha->parameter("dict1v", _dictV[0], _dictV[1], _dictV[2], _dictV[3]);

            _shader_p->parameter("tau", tau_dual);
            _shader_p->parameter("rcpEta", 1.0f / _cfg._eta);

            _shader_q->parameter("tau", tau_dual);
            _shader_q->parameter("mu", _cfg._mu);
            _shader_q->parameter("dxdy", ds, 0.0f, 0.0f, dt);

            checkGLErrorsHere0();

            int const nInnerIterations = _nInnerIterations;
            //int const nInnerIterations = (level == _nLevels-1) ? _nInnerIterations : 0;
            for (int k = 0; k < nInnerIterations /* * sqrtf(resizeFactor) */; ++k)
            {
               // Update primal variables first

               // u (x and y component)
               ubuffer1->activate();
               ubuffer2->enableTexture(GL_TEXTURE0_ARB);
               warpedBuffer.enableTexture(GL_TEXTURE1_ARB);
               qhbuffer2->enableTexture(GL_TEXTURE2_ARB);
               qvbuffer2->enableTexture(GL_TEXTURE3_ARB);

               _shader_u->enable();
               renderNormalizedQuad(GPU_SAMPLE_REVERSE_NEIGHBORS, ds, dt);
               _shader_u->disable();

               ubuffer2->disableTexture(GL_TEXTURE0_ARB);
               warpedBuffer.disableTexture(GL_TEXTURE1_ARB);
               qhbuffer2->disableTexture(GL_TEXTURE2_ARB);
               qvbuffer2->disableTexture(GL_TEXTURE3_ARB);

               std::swap(ubuffer1, ubuffer2);
               checkGLErrorsHere0();

               // alpha
               alphaFbo1->activate();
               glDrawBuffersARB(2, targetBuffers);

               alphaFbo2->getColorTexture(0).enable(GL_TEXTURE0_ARB);
               qhbuffer2->enableTexture(GL_TEXTURE1_ARB);
               qvbuffer2->enableTexture(GL_TEXTURE2_ARB);
               pbuffer2->enableTexture(GL_TEXTURE3_ARB);

               _shader_alpha->enable();
               renderNormalizedQuad(GPU_SAMPLE_REVERSE_NEIGHBORS, ds, dt);
               _shader_alpha->disable();

               glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);

               std::swap(alphaFbo1, alphaFbo2);

               checkGLErrorsHere0();

               // Now the dual variables
               pbuffer1->activate();
               alphaFbo2->getColorTexture(1).enable(GL_TEXTURE0_ARB); // betas
               pbuffer2->enableTexture(GL_TEXTURE1_ARB);

               _shader_p->enable();
               renderNormalizedQuad(GPU_SAMPLE_REVERSE_NEIGHBORS, ds, dt);
               _shader_p->disable();

               std::swap(pbuffer1, pbuffer2);

               // q
               qhbuffer1->activate();
               qhbuffer2->enableTexture(GL_TEXTURE0_ARB);
               ubuffer2->enableTexture(GL_TEXTURE1_ARB);
               alphaFbo2->getColorTexture(1).enable(GL_TEXTURE2_ARB);

               glMultiTexCoord2f(GL_TEXTURE4, 1.0f, 0.0f);
               glMultiTexCoord4f(GL_TEXTURE5, _dictH[0], _dictH[1], _dictH[2], _dictH[3]);
               _shader_q->enable();
               renderNormalizedQuad(GPU_SAMPLE_REVERSE_NEIGHBORS, ds, dt);
               _shader_q->disable();

               std::swap(qhbuffer1, qhbuffer2);

               qvbuffer1->activate();
               qvbuffer2->enableTexture(GL_TEXTURE0_ARB);

               glMultiTexCoord2f(GL_TEXTURE4, 0.0f, 1.0f);
               glMultiTexCoord4f(GL_TEXTURE5, _dictV[0], _dictV[1], _dictV[2], _dictV[3]);
               _shader_q->enable();
               renderNormalizedQuad(GPU_SAMPLE_REVERSE_NEIGHBORS, ds, dt);
               _shader_q->disable();

               std::swap(qvbuffer1, qvbuffer2);
            } // end for (k)
         } // end for (iter)
      } // end for (level)
   } // end Structure_L1_Stereo_PD::run()

//----------------------------------------------------------------------

   void
   warpImageWithDisparities(unsigned int u_tex, unsigned int I0_tex,
                            unsigned int I1_tex, int level, RTT_Buffer& dest)
   {
      static Cg_FragmentProgram * shader = 0;
      if (shader == 0)
      {
         shader = new Cg_FragmentProgram("v3d_gpuvarstereo::warpImageWithDisparities::shader");
         shader->setProgramFromFile("VarStereo/disp_warp_image.cg");
         shader->compile();
         checkGLErrorsHere0();
      }

      int const w = dest.width();
      int const h = dest.height();

      dest.activate();

      setupNormalizedProjection();

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, u_tex);
      glEnable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, I0_tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, level);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
      glEnable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, I1_tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, level);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
      glEnable(GL_TEXTURE_2D);

      // Provide uniform paramter via texcoord to avoid recompilation of shaders
      glMultiTexCoord3f(GL_TEXTURE3_ARB, 1.0f/w, 1.0f/h, 1 << level);
      shader->enable();
      renderNormalizedQuad();
      shader->disable();

      glActiveTexture(GL_TEXTURE0);
      glDisable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE1);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
      //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glDisable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE2);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glDisable(GL_TEXTURE_2D);
   } // end warpImageWithFlowField()

} // end namespace V3D_GPU

#endif
