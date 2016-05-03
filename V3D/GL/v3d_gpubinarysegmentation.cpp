#include "v3d_gpubinarysegmentation.h"

#if defined(V3DLIB_GPGPU_ENABLE_CG)

#include <GL/glew.h>
#include <iostream>

using namespace std;

namespace V3D_GPU
{

   void
   BinarySegmentationUzawa::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBuffer[0]->allocate(w, h);
      _uBuffer[1]->allocate(w, h);
      _pBuffer[0]->allocate(w, h);
      _pBuffer[1]->allocate(w, h);
   }

   void
   BinarySegmentationUzawa::deallocate()
   {
      _uBuffer[0]->deallocate();
      _uBuffer[1]->deallocate();
      _pBuffer[0]->deallocate();
      _pBuffer[1]->deallocate();

      _width = _height = -1;
   }

   void
   BinarySegmentationUzawa::initializeIterations()
   {
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      _uBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      glClearColor(0, 0, 0, 0);
      _pBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
   }

   void
   BinarySegmentationUzawa::iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("BinarySegmentationUzawa::uShader");
         uShader->setProgramFromFile("binary_segmentation_uzawa_update_u.cg");
         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pShader == 0)
      {
         pShader = new Cg_FragmentProgram("BinarySegmentationUzawa::pShader");
         pShader->setProgramFromFile("binary_segmentation_uzawa_update_p.cg");
         pShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pShader->parameter("tau", _tau_dual);
      checkGLErrorsHere0();

      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, costTexId);
      glEnable(GL_TEXTURE_2D);

      glActiveTexture(GL_TEXTURE3);
      glBindTexture(GL_TEXTURE_2D, edgeWeightTexId);
      glEnable(GL_TEXTURE_2D);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Update p
         _pBuffer[0]->activate();
         _uBuffer[1]->enableTexture(GL_TEXTURE0);
         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         pShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pShader->disable();

         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         std::swap(_pBuffer[0], _pBuffer[1]);

         // Update u
         _uBuffer[0]->activate();

         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBuffer[1]->disableTexture(GL_TEXTURE0);
         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         std::swap(_uBuffer[0], _uBuffer[1]);
      } // end for (iter)

      glActiveTexture(GL_TEXTURE2);
      glDisable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE3);
      glDisable(GL_TEXTURE_2D);
   } // end BinarySegmentationUzawa::iterate()

//======================================================================

   void
   BinarySegmentationPD::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBuffer[0]->allocate(w, h);
      _uBuffer[1]->allocate(w, h);
      _pBuffer[0]->allocate(w, h);
      _pBuffer[1]->allocate(w, h);
   }

   void
   BinarySegmentationPD::deallocate()
   {
      _uBuffer[0]->deallocate();
      _uBuffer[1]->deallocate();
      _pBuffer[0]->deallocate();
      _pBuffer[1]->deallocate();

      delete _uBuffer[0];
      delete _uBuffer[1];
      delete _pBuffer[0];
      delete _pBuffer[1];

      _width = _height = -1;
   }

   void
   BinarySegmentationPD::initializeIterations()
   {
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      _uBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      glClearColor(0, 0, 0, 0);
      _pBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
   }

   void
   BinarySegmentationPD::iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("BinarySegmentationPD::uShader");
         uShader->setProgramFromFile("binary_segmentation_pd_update_u.cg");
         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pShader == 0)
      {
         pShader = new Cg_FragmentProgram("ConvexMRF_3Labels_Generic::pqShader");
         pShader->setProgramFromFile("binary_segmentation_pd_update_p.cg");
         pShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", _tau_primal);
      pShader->parameter("tau", _tau_dual);
      checkGLErrorsHere0();

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Update p
         _pBuffer[0]->activate();
         _uBuffer[1]->enableTexture(GL_TEXTURE0);
         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         glActiveTexture(GL_TEXTURE2);
         glBindTexture(GL_TEXTURE_2D, edgeWeightTexId);
         glEnable(GL_TEXTURE_2D);

         pShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pShader->disable();

         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         std::swap(_pBuffer[0], _pBuffer[1]);

         // Update u
         _uBuffer[0]->activate();

         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         glActiveTexture(GL_TEXTURE2);
         glBindTexture(GL_TEXTURE_2D, costTexId);
         glEnable(GL_TEXTURE_2D);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBuffer[1]->disableTexture(GL_TEXTURE0);
         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         glActiveTexture(GL_TEXTURE2);
         glDisable(GL_TEXTURE_2D);

         std::swap(_uBuffer[0], _uBuffer[1]);
      } // end for (iter)
   } // end BinarySegmentationPD::iterate()

//======================================================================

   void
   BinarySegmentationFwBw::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBuffer[0]->allocate(w, h);
      _uBuffer[1]->allocate(w, h);
   }

   void
   BinarySegmentationFwBw::deallocate()
   {
      _uBuffer[0]->deallocate();
      _uBuffer[1]->deallocate();

      delete _uBuffer[0];
      delete _uBuffer[1];

      _width = _height = -1;
   }

   void
   BinarySegmentationFwBw::initializeIterations()
   {
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      _uBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
   }

   void
   BinarySegmentationFwBw::iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("BinarySegmentationFwBw::uShader");
         uShader->setProgramFromFile("binary_segmentation_fwbw_update_u.cg");
         uShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("tau", this->tau * this->epsilon);
      uShader->parameter("epsilon", this->epsilon, 1.0f / this->epsilon);
      checkGLErrorsHere0();

      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, edgeWeightTexId);
      glEnable(GL_TEXTURE_2D);

      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, costTexId);
      glEnable(GL_TEXTURE_2D);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         _uBuffer[0]->activate();
         _uBuffer[1]->enableTexture(GL_TEXTURE0);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBuffer[1]->disableTexture(GL_TEXTURE0);

         std::swap(_uBuffer[0], _uBuffer[1]);
      } // end for (iter)

      glActiveTexture(GL_TEXTURE1);
      glDisable(GL_TEXTURE_2D);

      glActiveTexture(GL_TEXTURE2);
      glDisable(GL_TEXTURE_2D);
   } // end BinarySegmentationFwBw::iterate()

//======================================================================

   void
   BinarySegmentationRelaxed::allocate(int w, int h)
   {
      _width = w;
      _height = h;

      _uBuffer[0]->allocate(w, h);
      _uBuffer[1]->allocate(w, h);
      _pBuffer[0]->allocate(w, h);
      _pBuffer[1]->allocate(w, h);
   }

   void
   BinarySegmentationRelaxed::deallocate()
   {
      _uBuffer[0]->deallocate();
      _uBuffer[1]->deallocate();
      _pBuffer[0]->deallocate();
      _pBuffer[1]->deallocate();

      delete _uBuffer[0];
      delete _uBuffer[1];
      delete _pBuffer[0];
      delete _pBuffer[1];

      _width = _height = -1;
   }

   void
   BinarySegmentationRelaxed::initializeIterations()
   {
      glClearColor(0.5f, 0.5f, 0.5f, 0.5f);
      _uBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
      glClearColor(0, 0, 0, 0);
      _pBuffer[1]->activate();
      glClear(GL_COLOR_BUFFER_BIT);
   }

   void
   BinarySegmentationRelaxed::iterate(unsigned int costTexId, unsigned int edgeWeightTexId, int nIterations)
   {
      static Cg_FragmentProgram * uShader = 0;
      static Cg_FragmentProgram * pShader = 0;

      if (uShader == 0)
      {
         uShader = new Cg_FragmentProgram("BinarySegmentationRelaxed::uShader");
         uShader->setProgramFromFile("binary_segmentation_relaxed_update_u.cg");
         uShader->compile();
         checkGLErrorsHere0();
      }

      if (pShader == 0)
      {
         pShader = new Cg_FragmentProgram("BinarySegmentationRelaxed::pShader");
         pShader->setProgramFromFile("binary_segmentation_uzawa_update_p.cg");
         pShader->compile();
         checkGLErrorsHere0();
      }

      float const ds = 1.0f/_width;
      float const dt = 1.0f/_height;

      setupNormalizedProjection();

      uShader->parameter("theta", this->theta);
      pShader->parameter("tau", this->tau / this->theta);
      checkGLErrorsHere0();

      glActiveTexture(GL_TEXTURE2);
      glBindTexture(GL_TEXTURE_2D, costTexId);
      glEnable(GL_TEXTURE_2D);

      glActiveTexture(GL_TEXTURE3);
      glBindTexture(GL_TEXTURE_2D, edgeWeightTexId);
      glEnable(GL_TEXTURE_2D);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Update p
         _pBuffer[0]->activate();
         _uBuffer[1]->enableTexture(GL_TEXTURE0);
         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         pShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         pShader->disable();

         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         std::swap(_pBuffer[0], _pBuffer[1]);

         // Update u
         _uBuffer[0]->activate();

         _pBuffer[1]->enableTexture(GL_TEXTURE1);

         uShader->enable();
         renderNormalizedQuad(GPU_SAMPLE_NEIGHBORS, ds, dt);
         uShader->disable();

         _uBuffer[1]->disableTexture(GL_TEXTURE0);
         _pBuffer[1]->disableTexture(GL_TEXTURE1);

         std::swap(_uBuffer[0], _uBuffer[1]);
      } // end for (iter)

      glActiveTexture(GL_TEXTURE2);
      glDisable(GL_TEXTURE_2D);
      glActiveTexture(GL_TEXTURE3);
      glDisable(GL_TEXTURE_2D);
   } // end BinarySegmentationRelaxed::iterate()

} // end namespace V3D_GPU

#endif // defined(V3DLIB_GPGPU_ENABLE_CG)
