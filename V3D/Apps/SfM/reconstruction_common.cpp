#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"
#include "Base/v3d_vrmlio.h"
#include "Math/v3d_sparseeig.h"
#include "Math/v3d_optimization.h"
#include "Geometry/v3d_poseutilities.h"
#include "Geometry/v3d_mviewinitialization.h"

#include "reconstruction_common.h"

#include <list>

using namespace std;
using namespace V3D;

//----------------------------------------------------------------------

CalibrationDatabase::CalibrationDatabase(char const * calibDbName)
{
   ifstream is(calibDbName);
   if (!is) throwV3DErrorHere("Could not open calibration database file.");

   int nViews;
   is >> nViews;
   _imageDimensions.resize(nViews);
   _intrinsics.resize(nViews);
   _distortions.resize(nViews);

   for (int i = 0; i < nViews; ++i)
   {
      Matrix3x3d& K = _intrinsics[i];
      makeIdentityMatrix(K);
      is >> K[0][0] >> K[0][1] >> K[0][2];
      is >> K[1][1] >> K[1][2];

      is >> _distortions[i][0] >> _distortions[i][1] >> _distortions[i][2] >> _distortions[i][3];
      is >> _imageDimensions[i].first >> _imageDimensions[i].second;
   } // end for (i)
} // end CalibrationDatabase::CalibrationDatabase()

//----------------------------------------------------------------------

namespace
{

   struct HomographyOptimizer : public SimpleLevenbergOptimizer
   {
         HomographyOptimizer(vector<PointCorrespondence> const& corrs)
            : SimpleLevenbergOptimizer(2 * corrs.size(), 9),
              _corrs(corrs)
         {
            for (size_t i = 0; i < corrs.size(); ++i)
            {
               observation[2*i+0] = corrs[i].right.pos[0];
               observation[2*i+1] = corrs[i].right.pos[1];
            }
         }

         virtual void augmentDiagonal(double& v) const { v += lambda; }

         virtual void evalFunction(Vector<double>& res)
         {
            Matrix3x3d H;
            H(1, 1) = currentParameters[0]; H(1, 2) = currentParameters[1]; H(1, 3) = currentParameters[2];
            H(2, 1) = currentParameters[3]; H(2, 2) = currentParameters[4]; H(2, 3) = currentParameters[5];
            H(3, 1) = currentParameters[6]; H(3, 2) = currentParameters[7]; H(3, 3) = currentParameters[8];

            for (size_t i = 0; i < _corrs.size(); ++i)
            {
               Vector2f const& x = _corrs[i].left.pos;
               Vector2d y;
               multiply_A_v_projective(H, x, y);

               res[2*i+0] = y[0];
               res[2*i+1] = y[1];
            }
         } // end evalFunction()

         virtual void fillJacobian(Matrix<double>& J)
         {
            Vector<double> const& H = currentParameters;

            Matrix<double> C(2, 9);

            for (size_t i = 0; i < _corrs.size(); ++i)
            {
               double const x1 = _corrs[i].left.pos[0];
               double const x2 = _corrs[i].left.pos[1];

               double const x_ = H[0]*x1 + H[1]*x2 + H[2];
               double const y_ = H[3]*x1 + H[4]*x2 + H[5];
               double const z_ = H[6]*x1 + H[7]*x2 + H[8];
               double const z2 = z_ * z_;

               C(1, 1) = x1 / z_;       C(1, 2) = x2 / z_;       C(1, 3) = 1.0 / z_;
               C(1, 4) = 0;             C(1, 5) = 0;             C(1, 6) = 0;
               C(1, 7) = -x1 * x_ / z2; C(1, 8) = -x2 * x_ / z2; C(1, 9) = -x_ / z2;

               C(2, 1) = 0;             C(2, 2) = 0;             C(2, 3) = 0;
               C(2, 4) = x1 / z_;       C(2, 5) = x2 / z_;       C(2, 6) = 1.0 / z_;
               C(2, 7) = -x1 * y_ / z2; C(2, 8) = -x2 * y_ / z2; C(2, 9) = -y_ / z2;

               copyMatrixSlice(C, 0, 0, 2, 9, J, 2*i, 0);
            } // end for (i)
         } // end fillJacobian()

         vector<PointCorrespondence> const& _corrs;
   }; // end struct HomographyOptimizer

   inline int
   getShuffleCount(int N, int nRounds, int minSampleSize)
   {
      return std::max(1, (int)((float)nRounds*(float)minSampleSize/(float)N + 0.5));
   }

} // end namespace <>

namespace V3D
{

   void
   adjustHomography(Matrix3x3d& H, vector<PointCorrespondence> const& corrs, int nIterations)
   {
      HomographyOptimizer opt(corrs);

      opt.currentParameters[0] = H(1, 1); opt.currentParameters[1] = H(1, 2); opt.currentParameters[2] = H(1, 3);
      opt.currentParameters[3] = H(2, 1); opt.currentParameters[4] = H(2, 2); opt.currentParameters[5] = H(2, 3);
      opt.currentParameters[6] = H(3, 1); opt.currentParameters[7] = H(3, 2); opt.currentParameters[8] = H(3, 3);

      opt.maxIterations = nIterations;

      opt.minimize();

      H(1, 1) = opt.currentParameters[0]; H(1, 2) = opt.currentParameters[1]; H(1, 3) = opt.currentParameters[2];
      H(2, 1) = opt.currentParameters[3]; H(2, 2) = opt.currentParameters[4]; H(2, 3) = opt.currentParameters[5];
      H(3, 1) = opt.currentParameters[6]; H(3, 2) = opt.currentParameters[7]; H(3, 3) = opt.currentParameters[8];
   } // end adjustHomography()

   void
   computeRobustOrientationMLE(std::vector<PointCorrespondence> const& corrs,
                               Matrix3x3d const& K1, Matrix3x3d const& K2,
                               double inlierThreshold, int nSamples,
                               RobustOrientationResult& res,
                               bool reportInliers, RobustOrientationMode mode)
   {
      res.inliers.clear();

      int const N = corrs.size();
      int const minSize = 5;

      double const sqrInlierThreshold = inlierThreshold*inlierThreshold;

      if (N < minSize) throwV3DErrorHere("At least 5 point correspondences are required.");

      Matrix3x3d const invK1   = invertedMatrix(K1);
      Matrix3x3d const invK2   = invertedMatrix(K2);
      Matrix3x3d const invK2_t = invK2.transposed();

      vector<Matrix3x3d> Es;
      Matrix3x3d bestEssential;

      vector<PointCorrespondence> normalizedCorrs(corrs);

      vector<Vector2d> leftSamples, rightSamples;

      // normalize points by multiplication with inverse affine matrix
      for (int i = 0; i < N; ++i)
      {
         multiply_A_v_projective(invK1, corrs[i].left.pos, normalizedCorrs[i].left.pos);
         multiply_A_v_projective(invK2, corrs[i].right.pos, normalizedCorrs[i].right.pos);
      }

      unsigned int const nShuffles = getShuffleCount(N, nSamples, minSize);

      double minError = 1e30;

      double outlierFraction = 1.0;

      // vector to hold the indices of the sample points
      vector<unsigned int> indices(N);
      for (int i = 0; i < N; ++i) indices[i] = i;

      double x1[5], y1[5], x2[5], y2[5];

      vector<int> inliers;
      if (mode.iterativeRefinement) inliers.resize(N);

      unsigned int drawnSamples = 0;      
      for (unsigned int s = 0; s < nShuffles; ++s)
      {
         if (drawnSamples > nSamples) break;
         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-minSize; j += minSize)
         {
            if (drawnSamples > nSamples) break;
            Es.clear();

            x1[0] = normalizedCorrs[indices[j+0]].left.pos[0]; y1[0] = normalizedCorrs[indices[j+0]].left.pos[1];
            x1[1] = normalizedCorrs[indices[j+1]].left.pos[0]; y1[1] = normalizedCorrs[indices[j+1]].left.pos[1];
            x1[2] = normalizedCorrs[indices[j+2]].left.pos[0]; y1[2] = normalizedCorrs[indices[j+2]].left.pos[1];
            x1[3] = normalizedCorrs[indices[j+3]].left.pos[0]; y1[3] = normalizedCorrs[indices[j+3]].left.pos[1];
            x1[4] = normalizedCorrs[indices[j+4]].left.pos[0]; y1[4] = normalizedCorrs[indices[j+4]].left.pos[1];

            x2[0] = normalizedCorrs[indices[j+0]].right.pos[0]; y2[0] = normalizedCorrs[indices[j+0]].right.pos[1];
            x2[1] = normalizedCorrs[indices[j+1]].right.pos[0]; y2[1] = normalizedCorrs[indices[j+1]].right.pos[1];
            x2[2] = normalizedCorrs[indices[j+2]].right.pos[0]; y2[2] = normalizedCorrs[indices[j+2]].right.pos[1];
            x2[3] = normalizedCorrs[indices[j+3]].right.pos[0]; y2[3] = normalizedCorrs[indices[j+3]].right.pos[1];
            x2[4] = normalizedCorrs[indices[j+4]].right.pos[0]; y2[4] = normalizedCorrs[indices[j+4]].right.pos[1];

            Es.clear();
            try
            {
               computeEssentialsFromFiveCorrs(x1, y1, x2, y2, Es);
            }
            catch (std::exception exn)
            {
               Es.clear();
               cerr << "Exception caught from computeEssentialsFromFiveCorrs(): " << exn.what() << endl;
            }
            catch (std::string s)
            {
               Es.clear();
               cerr << "Exception caught from computeEssentialsFromFiveCorrs(): " << s << endl;
            }
            catch (...)
            {
               Es.clear();
               cerr << "Unknown exception from computeEssentialsFromFiveCorrs()." << endl;
            }

            ++drawnSamples;

            for (int r = 0; r < Es.size(); ++r)
            {
               Matrix3x3d fund = invK2_t * Es[r] * invK1;

               int nInliers = 0;
               double curError = 0;
               for (int i = 0; i < N; ++i)
               {
                  double const dist = sampsonEpipolarError(corrs[i], fund);
                  curError += std::min(dist, sqrInlierThreshold);
                  if (dist < sqrInlierThreshold) ++nInliers;
               }

               if (curError < minError)
               {
                  Matrix3x3d R;
                  Vector3d t;
                  bool const status = relativePoseFromEssential(Es[r], 5, x1, y1, x2, y2, R, t);
                  if (!status) continue;

                  if (mode.iterativeRefinement)
                  {
                     Matrix3x3d const R_orig = R;
                     Vector3d const t_orig = t;

                     /// get inlier index
                     inliers.clear();
                     for (int i = 0; i < N; ++i)
                     {
                        double const dist = sampsonEpipolarError(corrs[i], fund);
                        if (dist < sqrInlierThreshold) inliers.push_back(i);
                     }

                     vector<Vector2d> left(inliers.size());
                     vector<Vector2d> right(inliers.size());

                     for (int i = 0; i < inliers.size(); ++i)
                     {
                        int const ix = inliers[i];
                        left[i][0] = corrs[ix].left.pos[0];
                        left[i][1] = corrs[ix].left.pos[1];
                        right[i][0] = corrs[ix].right.pos[0];
                        right[i][1] = corrs[ix].right.pos[1];
                     }

                     Matrix3x3d const E_orig = computeEssentialFromRelativePose(R, t);

                     refineRelativePose(left, right, K1, K2, R, t, inlierThreshold);

                     Matrix3x3d E_refined = computeEssentialFromRelativePose(R, t);
                     Matrix3x3d F_refined = invK2_t * E_refined * invK1;

                     int const nInliers_orig = nInliers;

                     /// re-estimate inliers/error
                     double curError_refined = 0;
                     nInliers = 0;
                     for (int i = 0; i < N; ++i)
                     {
                        double const dist = sampsonEpipolarError(corrs[i], F_refined);
                        curError_refined += std::min(dist, sqrInlierThreshold);
                        if (dist < sqrInlierThreshold) ++nInliers;
                     }

                     if (curError_refined > curError)
                     {
                        cout << "computeRobustOrientationMLE(): curError_refined (" << curError_refined
                             << ") > curError (" << curError << ")." << endl;
                        cout << "inliers.size() = " << inliers.size() << endl;
                        cout << "nInliers_orig = " << nInliers_orig << ", nInliers = " << nInliers << endl;
                        cout << "R_orig = "; displayMatrix(R_orig);
                        cout << "R_refined = "; displayMatrix(R);
                        cout << "t_orig = "; displayVector(t_orig);
                        cout << "t_refined = "; displayVector(t);
                     }

                     fund  = F_refined;
                     Es[r] = E_refined;
                     curError = std::min(curError, curError_refined);
                  } // end if (mode.iterativeRefinement)

                  minError        = curError;
                  res.error       = minError / N;
                  res.essential   = Es[r];
                  res.fundamental = fund;
                  res.rotation    = R;
                  res.translation = t;

                  /// adaptive number of samples computation 
                  outlierFraction = 1.0 - float(nInliers - minSize) / float(corrs.size() - minSize);
                  nSamples = std::min(nSamples, ransacNSamples(minSize, outlierFraction, 1.0 - 10e-10));
               } // end if (curError < minError)
            } // end for (r)
         } // end for (j)
      } // end for (s)

      if (reportInliers)
      {
         res.inliers.reserve(N);
         for (int i = 0; i < N; ++i)
         {
            double const dist = sampsonEpipolarError(corrs[i], res.fundamental);
            if (dist < sqrInlierThreshold) res.inliers.push_back(i);
         }
      } // end if (reportInliers)
   } // end computeRobustOrientationMLE()

   void
   computeRobustAbsolutePoseMLE(std::vector<Vector2f> const& normPoints2d, std::vector<Vector3d> const& points3d,
                                double inlierThreshold, int nSamples,
                                RobustOrientationResult& res, RobustOrientationMode mode)throw()
   {
      res.inliers.clear();

      int const N = normPoints2d.size();
      int const minSize = 3;

      double const sqrInlierThreshold = inlierThreshold*inlierThreshold;

      if (N < minSize) throwV3DErrorHere("At least 3 point correspondences are required.");

      unsigned int const nShuffles = getShuffleCount(N, nSamples, minSize);

      double minError = 1e30;

      double outlierFraction = 1.0;

      // vector to hold the indices of the sample points
      vector<unsigned int> indices(N);
      for (int i = 0; i < N; ++i) indices[i] = i;

      Vector2f p1, p2, p3;
      Vector3d X1, X2, X3;

      CameraMatrix cam;

      vector<Matrix3x4d> RTs;
      vector<int>        curInliers;

      unsigned int drawnSamples = 0;      
      for (unsigned int s = 0; s < nShuffles; ++s)
      {
         if (drawnSamples > nSamples) break;
         // shuffle indices 
         random_shuffle(indices.begin(), indices.end());

         for (int j = 0; j < N-minSize; j += minSize)
         {
            if (drawnSamples > nSamples) break;

            p1 = normPoints2d[indices[j+0]]; X1 = points3d[indices[j+0]];
            p2 = normPoints2d[indices[j+1]]; X2 = points3d[indices[j+1]];
            p3 = normPoints2d[indices[j+2]]; X3 = points3d[indices[j+2]];

            RTs.clear();

            try
            {
               computeAbsolutePose3Point(p1, p2, p3, X1, X2, X3, RTs);
            }
            catch (std::exception exn)
            {
               RTs.clear();
               cerr << "Exception caught from computeAbsolutePose3Point(): " << exn.what() << endl;
            }
            catch (std::string s)
            {
               RTs.clear();
               cerr << "Exception caught from computeAbsolutePose3Point(): " << s << endl;
            }
            catch (...)
            {
               RTs.clear();
               cerr << "Unknown exception from computeAbsolutePose3Point()." << endl;
            }

            ++drawnSamples;

            for (int r = 0; r < RTs.size(); ++r)
            {
               cam.setOrientation(RTs[r]);

               curInliers.clear();

               double curError = 0;
               for (int i = 0; i < N; ++i)
               {
                  double const dist2 = sqrDistance_L2(normPoints2d[i], cam.projectPoint(points3d[i]));
                  curError += std::min(dist2, sqrInlierThreshold);
                  if (dist2 < sqrInlierThreshold) curInliers.push_back(i);
               }

               if (curError < minError)
               {
                  if (mode.iterativeRefinement)
                  {
                     Matrix3x3d R = cam.getRotation();
                     Vector3d   T = cam.getTranslation();

                     CameraMatrix cam_orig(cam);

                     vector<Vector2f> ps(curInliers.size());
                     vector<Vector3d> Xs(curInliers.size());

                     for (int i = 0; i < curInliers.size(); ++i)
                     {
                        int const ix = curInliers[i];
                        ps[i] = normPoints2d[ix];
                        Xs[i] = points3d[ix];
                     }
                     if(curInliers.size()<minSize)
                        continue;//throwV3DErrorHere("At least 3 point correspondences are required.");
                     refineAbsolutePose(ps, Xs, inlierThreshold, R, T);
                     cam.setRotation(R);
                     cam.setTranslation(T);

                     int const nInliers_orig = curInliers.size();

                     /// re-estimate inliers/error
                     double curError_refined = 0;
                     curInliers.clear();
                     for (int i = 0; i < N; ++i)
                     {
                        double const dist2 = sqrDistance_L2(normPoints2d[i], cam.projectPoint(points3d[i]));
                        curError_refined += std::min(dist2, sqrInlierThreshold);
                        if (dist2 < sqrInlierThreshold) curInliers.push_back(i);
                     }

                     if (curError_refined > curError)
                     {
                        cout << "computeRobustAbsolutePoseMLE(): curError_refined (" << curError_refined
                             << ") > curError (" << curError << ")." << endl;
                        cout << "inliers.size() = " << curInliers.size() << ", nInliers_orig = " << nInliers_orig << endl;
                        cout << "R_orig = "; displayMatrix(cam_orig.getRotation());
                        cout << "R_refined = "; displayMatrix(R);
                        cout << "t_orig = "; displayVector(cam_orig.getTranslation());
                        cout << "t_refined = "; displayVector(T);
                     }
                     curError = std::min(curError, curError_refined);
                  } // end if (mode.iterativeRefinement)

                  minError        = curError;
                  res.error       = minError / N;
                  res.rotation    = cam.getRotation();
                  res.translation = cam.getTranslation();
                  res.inliers     = curInliers;

                  /// adaptive number of samples computation 
                  outlierFraction = 1.0 - float(curInliers.size() - minSize) / float(N - minSize);
                  nSamples = std::min(nSamples, 3*ransacNSamples(minSize, outlierFraction, 1.0 - 10e-10));
                  //cout<<"nSamples needed with current outlierFraction ("<< outlierFraction<<"="<< curInliers.size() <<" inliers) : "<<nSamples<<" samples, current : "<<drawnSamples<<". Error = "<<minError<<endl;
                  /*
                  char buf[1024];vector<Vector3b> const wheel = makeColorWheel(20);
                  sprintf(buf, "cam_AbsPoseMLE_%i.wrl",curInliers.size());Matrix3x3d K;makeIdentityMatrix(K);
                  writeCameraFrustumToVRML(CameraMatrix(K,RTs[r]),1.0,Vector3b(0,255,0),buf,false);
                  for (int rr = 0; rr < RTs.size(); ++rr)
                  {
                     if(rr!=r)
                        writeCameraFrustumToVRML(CameraMatrix(K,RTs[rr]),1.0,wheel[rr],buf,true);
                  }
                  */
               } // end if (curError < minError)

            } // end for (r)
         } // end for (j)
      } // end for (s)
      //cout<<"Drawn "<<drawnSamples<<" samples"<<endl;
   } // end computeRobustAbsolutePoseMLE()

//**********************************************************************

   void
   computeScaleRatios(float cosAngleThreshold,
                      Matrix3x3d const& R01, Vector3d const& T01,
                      Matrix3x3d const& R12, Vector3d const& T12,
                      Matrix3x3d const& R20, Vector3d const& T20,
                      std::vector<PointCorrespondence> const& corrs01,
                      std::vector<PointCorrespondence> const& corrs12,
                      std::vector<PointCorrespondence> const& corrs02,
                      double& s012, double& s120, double& s201, double& weight)
   {
      Matrix3x3d I;  makeIdentityMatrix(I);
      Matrix3x4d P0; makeIdentityMatrix(P0);

      CameraMatrix cam01, cam12, cam20;

      int const view0 = corrs01[0].left.view;
      int const view1 = corrs01[0].right.view;
      int const view2 = corrs12[0].right.view;

      cam01.setIntrinsic(I);
      cam12.setIntrinsic(I);
      cam20.setIntrinsic(I);

      cam01.setRotation(R01);
      cam12.setRotation(R12);
      cam20.setRotation(R20);

      cam01.setTranslation(T01);
      cam12.setTranslation(T12);
      cam20.setTranslation(T20);

      Matrix3x4d const P01 = cam01.getProjection();
      Matrix3x4d const P12 = cam12.getProjection();
      Matrix3x4d const P20 = cam20.getProjection();

      std::vector<PointCorrespondence> allCorrs;
      allCorrs.reserve(corrs01.size() + corrs12.size() + corrs02.size());
      for (size_t k = 0; k < corrs01.size(); ++k) allCorrs.push_back(corrs01[k]);
      for (size_t k = 0; k < corrs12.size(); ++k) allCorrs.push_back(corrs12[k]);
      for (size_t k = 0; k < corrs02.size(); ++k) allCorrs.push_back(corrs02[k]);

      std::vector<TriangulatedPoint> model;
      TriangulatedPoint::connectTracks(allCorrs, model, 3);

//       for (int j = 0; j < model.size(); ++j)
//       {
//          cout << "X" << j << ": ";
//          TriangulatedPoint& X = model[j];
//          for (int l = 0; l < X.measurements.size(); ++l)
//          {
//             PointMeasurement const& m = X.measurements[l];
//             cout << "(" << m.view << ": " << m.id << ") ";
//          }
//          cout << endl;
//       }

      vector<double> ratios012, ratios120, ratios201;
      ratios012.reserve(model.size());
      ratios120.reserve(model.size());
      ratios201.reserve(model.size());

      for (size_t j = 0; j < model.size(); ++j)
      {
         vector<PointMeasurement> const& ms = model[j].measurements;
         if (ms.size() != 3) continue;

         bool foundView0 = false;
         bool foundView1 = false;
         bool foundView2 = false;

         PointMeasurement m0, m1, m2;

         for (int l = 0; l < ms.size(); ++l)
         {
            if (ms[l].view == view0)
            {
               foundView0 = true;
               m0 = ms[l];
            }
            else if (ms[l].view == view1)
            {
               foundView1 = true;
               m1 = ms[l];
            }
            else if (ms[l].view == view2)
            {
               foundView2 = true;
               m2 = ms[l];
            }
         }

         if (!foundView0) continue;
         if (!foundView1) continue;
         if (!foundView2) continue;

         // Found a point visible in all 3 views.

         // Check, if the pairwise triangulation angles are sufficient
         Vector3d ray0, ray1;
         ray0[0] = m0.pos[0]; ray0[1] = m0.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam01.getRay(m1.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         ray0[0] = m1.pos[0]; ray0[1] = m1.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam12.getRay(m2.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         ray0[0] = m2.pos[0]; ray0[1] = m2.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam20.getRay(m0.pos);
         if (innerProduct(ray0, ray1) > cosAngleThreshold) continue;

         // Compute scale ratios
         PointCorrespondence corr01, corr12, corr20;

         corr01.left = m0; corr01.right = m1;
         corr12.left = m1; corr12.right = m2;
         corr20.left = m2; corr20.right = m0;

         Vector3d X01 = triangulateLinear(P0, P01, corr01);
         if (X01[2] <= 0.0) continue;
         Vector3d X12 = triangulateLinear(P0, P12, corr12);
         if (X12[2] <= 0.0) continue;
         Vector3d X20 = triangulateLinear(P0, P20, corr20);
         if (X20[2] <= 0.0) continue;

         ratios012.push_back(distance_L2(X01, cam01.cameraCenter()) / norm_L2(X12));
         ratios120.push_back(distance_L2(X12, cam12.cameraCenter()) / norm_L2(X20));
         ratios201.push_back(distance_L2(X20, cam20.cameraCenter()) / norm_L2(X01));
      } // end for (j)

      //cout << "cam01.cameraCenter() = "; displayVector(cam01.cameraCenter());
      //cout << "cam12.cameraCenter() = "; displayVector(cam12.cameraCenter());
      //cout << "cam20.cameraCenter() = "; displayVector(cam20.cameraCenter());
      //displayVector(ratios012);
      //displayVector(ratios120);
      //displayVector(ratios201);

      //cout << "ratios012.size() = " << ratios012.size() << endl;
      if (ratios012.size() < 10) // There should be probably more tests
      {
         weight = -1.0f;
         return;
      }

      weight = ratios012.size();
      //weight = 1.0f;

      std::sort(ratios012.begin(), ratios012.end());
      std::sort(ratios120.begin(), ratios120.end());
      std::sort(ratios201.begin(), ratios201.end());

      s012 = medianQuantile(ratios012);
      s120 = medianQuantile(ratios120);
      s201 = medianQuantile(ratios201);
   } // end computeScaleRatios()

//**********************************************************************

   void
   computeScaleRatiosGeneralized(Matrix3x3d const& R01, Vector3d const& T01,
                                 Matrix3x3d const& R12, Vector3d const& T12,
                                 Matrix3x3d const& R20, Vector3d const& T20,
                                 std::vector<PointCorrespondence> const& corrs01,
                                 std::vector<PointCorrespondence> const& corrs12,
                                 std::vector<PointCorrespondence> const& corrs02,
                                 double& s012, double& s120, double& s201, double& weight, float const cosAngleThreshold)
   {
      Matrix3x3d I;  makeIdentityMatrix(I);
      Matrix3x4d P0; makeIdentityMatrix(P0);

      CameraMatrix cam01, cam12, cam20;

      int const view0 = corrs01[0].left.view;
      int const view1 = corrs01[0].right.view;
      int const view2 = corrs12[0].right.view;

      cam01.setIntrinsic(I);
      cam12.setIntrinsic(I);
      cam20.setIntrinsic(I);

      cam01.setRotation(R01);
      cam12.setRotation(R12);
      cam20.setRotation(R20);

      cam01.setTranslation(T01);
      cam12.setTranslation(T12);
      cam20.setTranslation(T20);

      Matrix3x4d const P01 = cam01.getProjection();
      Matrix3x4d const P12 = cam12.getProjection();
      Matrix3x4d const P20 = cam20.getProjection();

      std::vector<PointCorrespondence> allCorrs;
      allCorrs.reserve(corrs01.size() + corrs12.size() + corrs02.size());
      for (size_t k = 0; k < corrs01.size(); ++k) allCorrs.push_back(corrs01[k]);
      for (size_t k = 0; k < corrs12.size(); ++k) allCorrs.push_back(corrs12[k]);
      for (size_t k = 0; k < corrs02.size(); ++k) allCorrs.push_back(corrs02[k]);

      std::vector<TriangulatedPoint> model;
      TriangulatedPoint::connectTracks(allCorrs, model, 3);

      vector<double> ratios012, ratios120, ratios201;
      ratios012.reserve(model.size());
      ratios120.reserve(model.size());
      ratios201.reserve(model.size());

      for (size_t j = 0; j < model.size(); ++j)
      {
         vector<PointMeasurement> const& ms = model[j].measurements;
         if (ms.size() != 3) continue;

         bool foundView0 = false;
         bool foundView1 = false;
         bool foundView2 = false;

         PointMeasurement m0, m1, m2;

         for (int l = 0; l < ms.size(); ++l)
         {
            if (ms[l].view == view0)
            {
               foundView0 = true;
               m0 = ms[l];
            }
            else if (ms[l].view == view1)
            {
               foundView1 = true;
               m1 = ms[l];
            }
            else if (ms[l].view == view2)
            {
               foundView2 = true;
               m2 = ms[l];
            }
         }

         if (!foundView0) continue;
         if (!foundView1) continue;
         if (!foundView2) continue;

         // Found a point visible in all 3 views.

         // Check, if the pairwise triangulation angles are sufficient
         Vector3d ray0, ray1;
         ray0[0] = m0.pos[0]; ray0[1] = m0.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam01.getRay(m1.pos);
         bool const goodAngle01 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         ray0[0] = m1.pos[0]; ray0[1] = m1.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam12.getRay(m2.pos);
         bool const goodAngle12 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         ray0[0] = m2.pos[0]; ray0[1] = m2.pos[1]; ray0[2] = 1.0f;
         normalizeVector(ray0);
         ray1 = cam20.getRay(m0.pos);
         bool const goodAngle20 = (innerProduct(ray0, ray1) < cosAngleThreshold);

         int nGoodAngles = 0;
         if (goodAngle01) ++nGoodAngles;
         if (goodAngle12) ++nGoodAngles;
         if (goodAngle20) ++nGoodAngles;
         if (nGoodAngles < 2) continue;

         // Compute scale ratios
         PointCorrespondence corr01, corr12, corr20;

         corr01.left = m0; corr01.right = m1;
         corr12.left = m1; corr12.right = m2;
         corr20.left = m2; corr20.right = m0;

         // Initialize 3D points as very distance to handle close-to panoramic image pairs correctly
         Vector3d X01(0.0, 0.0, 1e6), X12(0.0, 0.0, 1e6), X20(0.0, 0.0, 1e6);

         if (goodAngle01) X01 = triangulateLinear(P0, P01, corr01);
         if (X01[2] <= 0.0) continue;
         if (goodAngle12) X12 = triangulateLinear(P0, P12, corr12);
         if (X12[2] <= 0.0) continue;
         if (goodAngle20) X20 = triangulateLinear(P0, P20, corr20);
         if (X20[2] <= 0.0) continue;

         ratios012.push_back(distance_L2(X01, cam01.cameraCenter()) / norm_L2(X12));
         ratios120.push_back(distance_L2(X12, cam12.cameraCenter()) / norm_L2(X20));
         ratios201.push_back(distance_L2(X20, cam20.cameraCenter()) / norm_L2(X01));
      } // end for (j)

      //cout << "cam01.cameraCenter() = "; displayVector(cam01.cameraCenter());
      //cout << "cam12.cameraCenter() = "; displayVector(cam12.cameraCenter());
      //cout << "cam20.cameraCenter() = "; displayVector(cam20.cameraCenter());
      //displayVector(ratios012);
      //displayVector(ratios120);
      //displayVector(ratios201);

      //cout << "ratios012.size() = " << ratios012.size() << endl;
      if (ratios012.size() < 10) // There should be probably more tests
      {
         weight = -1.0f;
         return;
      }

      weight = ratios012.size();
      //weight = 1.0f;

      std::sort(ratios012.begin(), ratios012.end());
      std::sort(ratios120.begin(), ratios120.end());
      std::sort(ratios201.begin(), ratios201.end());

      s012 = medianQuantile(ratios012);
      s120 = medianQuantile(ratios120);
      s201 = medianQuantile(ratios201);
   } // end computeScaleRatiosGeneralized()

} // end namespace V3D

//**********************************************************************

void
extractConnectedComponent(std::map<ViewPair, std::set<int> > const& pairThirdViewMap,
                          std::set<ViewTripletKey>& unhandledTriples,
                          std::set<ViewTripletKey>& connectedTriples,
                          std::set<ViewPair>& handledEdges)
{
   // Breadth-first search for connected components

   connectedTriples.clear();
   handledEdges.clear();

   list<ViewPair> edgeQueue;

   ViewTripletKey startTriple = *unhandledTriples.begin();
   unhandledTriples.erase(unhandledTriples.begin());
   connectedTriples.insert(startTriple);
   edgeQueue.push_back(ViewPair(startTriple.views[0], startTriple.views[1]));
   edgeQueue.push_back(ViewPair(startTriple.views[0], startTriple.views[2]));
   edgeQueue.push_back(ViewPair(startTriple.views[1], startTriple.views[2]));

   while (!edgeQueue.empty())
   {
      ViewPair curPair = edgeQueue.front();
      edgeQueue.pop_front();

      handledEdges.insert(curPair);

      map<ViewPair, set<int> >::const_iterator p = pairThirdViewMap.find(curPair);
      assert(p != pairThirdViewMap.end());
      set<int> const& thirdViews = (*p).second;

      for (set<int>::const_iterator q = thirdViews.begin(); q != thirdViews.end(); ++q)
      {
         int i0 = curPair.view0;
         int i1 = curPair.view1;
         int i2 = *q;
         sort3(i0, i1, i2);
         ViewTripletKey key(i0, i1, i2);

         if (connectedTriples.find(key) != connectedTriples.end()) continue;
         if (unhandledTriples.find(key) == unhandledTriples.end()) continue;

         connectedTriples.insert(key);
         unhandledTriples.erase(key);

         ViewPair pair01(i0, i1);
         ViewPair pair02(i0, i2);
         ViewPair pair12(i1, i2);

         if (handledEdges.find(pair01) == handledEdges.end()) edgeQueue.push_back(pair01);
         if (handledEdges.find(pair02) == handledEdges.end()) edgeQueue.push_back(pair02);
         if (handledEdges.find(pair12) == handledEdges.end()) edgeQueue.push_back(pair12);
      } // end for (q)
      //cout << "edgeQueue.size() = " << edgeQueue.size() << endl;
   } // end while
} // end computeConntectedComponent()

//**********************************************************************

#define USE_SYMMETRIC_SCALE_RATIOS 1

SubmodelReconstruction::SubmodelReconstruction(std::set<int> const& viewIds,
                                     std::set<ViewTripletKey> const& collectedTriples)
   : _nViews(viewIds.size()), _viewIdBackMap(viewIds.size())
{
   using namespace std;

   // Map view ids to the range [0, N-1]
   for (set<int>::const_iterator p = viewIds.begin(); p != viewIds.end(); ++p)
   {
      int newId = _viewIdMap.size();
      _viewIdMap.insert(make_pair(*p, newId));
      _viewIdBackMap[newId] = *p;
   } // end for (p)

   // Filter relevant triplets
   for (set<ViewTripletKey>::const_iterator p = collectedTriples.begin(); p != collectedTriples.end(); ++p)
   {
      int const v0 = (*p).views[0];
      int const v1 = (*p).views[1];
      int const v2 = (*p).views[2];

      if (viewIds.find(v0) != viewIds.end() &&
          viewIds.find(v1) != viewIds.end() &&
          viewIds.find(v2) != viewIds.end())
      {
         _triplets.push_back(*p);
         _viewPairs.insert(ViewPair(v0, v1));
         _viewPairs.insert(ViewPair(v0, v2));
         _viewPairs.insert(ViewPair(v1, v2));
      }
   } // end for (p)

   for (set<ViewPair>::const_iterator p = _viewPairs.begin(); p != _viewPairs.end(); ++p)
   {
      int const i0 = (*_viewIdMap.find((*p).view0)).second;
      int const i1 = (*_viewIdMap.find((*p).view1)).second;
      _viewPairVecPosMap.insert(make_pair(ViewPair(i0, i1), _viewPairVec.size()));
      _viewPairVec.push_back(ViewPair(i0, i1));
   } // end for (p)
} // end SubmodelReconstruction::SubmodelReconstruction()

void
SubmodelReconstruction::computeConsistentRotations(std::map<ViewPair, V3D::Matrix3x3d> const& relRotations)
{
   using namespace std;
   using namespace V3D;

   cout << "Computing consistent rotations..." << endl;
   {
      vector<Matrix3x3d> tmpRelRotations;
            
      for (set<ViewPair>::const_iterator p = _viewPairs.begin(); p != _viewPairs.end(); ++p)
         tmpRelRotations.push_back(relRotations.find(*p)->second);

      Timer t("computeConsistentRotations()");
      t.start();
      int const method = V3D_CONSISTENT_ROTATION_METHOD_SPARSE_EIG;
      vector<pair<int, int> > viewPairVec(_viewPairVec.size());
      for (size_t i = 0; i < _viewPairVec.size(); ++i)
         viewPairVec[i] = make_pair(_viewPairVec[i].view0, _viewPairVec[i].view1);

      V3D::computeConsistentRotations(_nViews, tmpRelRotations, viewPairVec, _rotations, method);
      t.stop();
      t.print();
   }
   cout << "done." << endl;
} // end SubmodelReconstruction::computeConsistentRotations()

void
SubmodelReconstruction::computeConsistentTranslationLengths(std::map<ViewPair, V3D::Vector3d> const& relTranslations,
                                                            V3D::CachedStorage<MatchDataTable>& matchDataCache,
                                                            std::map<ViewPair, int> const& viewPairOIDMap,
                                                            V3D::CachedStorage<TripletDataTable>& tripletDataCache,
                                                            std::map<ViewTripletKey, int> const& tripletOIDMap,
                                                            bool reestimateScaleRations)
{
   for (set<ViewPair>::const_iterator p = _viewPairs.begin(); p != _viewPairs.end(); ++p)
      _relTranslationVec.push_back(relTranslations.find(*p)->second);

   int const nViewPairs = _viewPairs.size();
   int const nTriplets  = _triplets.size();

   vector<pair<int, int> > nzB;
   vector<double>          valsB;
#if !defined(USE_SYMMETRIC_SCALE_RATIOS)
   nzB.reserve(6*nTriplets);
   valsB.reserve(6*nTriplets);
#else
   nzB.reserve(12*nTriplets);
   valsB.reserve(12*nTriplets);
#endif

   cout << "Gathering scale ratios from triplets..." << endl;
   Timer t("Gathering scale ratios");
   t.start();

   int n = 0;
   for (vector<ViewTripletKey>::const_iterator p = _triplets.begin(); p != _triplets.end(); ++p, ++n)
   {
      int const v0 = (*p).views[0];
      int const v1 = (*p).views[1];
      int const v2 = (*p).views[2];

      int const i0 = _viewIdMap[v0];
      int const i1 = _viewIdMap[v1];
      int const i2 = _viewIdMap[v2];

      assert(_viewPairVecPosMap.find(ViewPair(i0, i1)) != _viewPairVecPosMap.end());
      assert(_viewPairVecPosMap.find(ViewPair(i0, i2)) != _viewPairVecPosMap.end());
      assert(_viewPairVecPosMap.find(ViewPair(i1, i2)) != _viewPairVecPosMap.end());
      int const pos01 = (*_viewPairVecPosMap.find(ViewPair(i0, i1))).second;
      int const pos12 = (*_viewPairVecPosMap.find(ViewPair(i1, i2))).second;
      int const pos02 = (*_viewPairVecPosMap.find(ViewPair(i0, i2))).second;

      ViewTripletKey key(v0, v1, v2);
      assert(tripletOIDMap.find(key) != tripletOIDMap.end());
      int const oid = (*tripletOIDMap.find(key)).second;
      TripleReconstruction const * tripletModel = tripletDataCache[oid];

//                cout << "(v0, v1, v2) = (" << v0 << ", " << v1 << ", " << v2 << ")" << endl;
//                cout << "(V0, V1, V2) = (" << tripletModel->views[0] << ", " << tripletModel->views[1]
//                     << ", " << tripletModel->views[2] << ")" << endl;

      double s012, s120, s201, weight;

      if (!reestimateScaleRations)
      {
         // Here we use that the mapping between local and global view ids is monotone,
         // i.e. v0 < v1 < v2 means, that v0 has index 0 etc.
         CameraMatrix cam0(tripletModel->intrinsics[0], tripletModel->orientations[0]);
         CameraMatrix cam1(tripletModel->intrinsics[1], tripletModel->orientations[1]);
         CameraMatrix cam2(tripletModel->intrinsics[2], tripletModel->orientations[2]);

         double const d01 = distance_L2(cam0.cameraCenter(), cam1.cameraCenter());
         double const d20 = distance_L2(cam0.cameraCenter(), cam2.cameraCenter());
         double const d12 = distance_L2(cam1.cameraCenter(), cam2.cameraCenter());

         s012 = d12 / d01;
         s120 = d20 / d12;
         s201 = d01 / d20;
         weight = 1.0f;
      }
      else
      {
         Matrix3x3d const& R0 = _rotations[i0];
         Matrix3x3d const& R1 = _rotations[i1];
         Matrix3x3d const& R2 = _rotations[i2];

         Matrix3x3d const R01 = R1 * R0.transposed();
         Matrix3x3d const R12 = R2 * R1.transposed();
         Matrix3x3d const R20 = R0 * R2.transposed();

         Vector3d const& T01 = _relTranslationVec[pos01];
         Vector3d const& T12 = _relTranslationVec[pos12];
         Vector3d const& T02 = _relTranslationVec[pos02];
         Vector3d const  T20 = -(R20 * T02);

         int const oid01 = (*viewPairOIDMap.find(ViewPair(v0, v1))).second;
         int const oid02 = (*viewPairOIDMap.find(ViewPair(v0, v2))).second;
         int const oid12 = (*viewPairOIDMap.find(ViewPair(v1, v2))).second;

         vector<PointCorrespondence> corrs01 = matchDataCache[oid01]->corrs;
         vector<PointCorrespondence> corrs02 = matchDataCache[oid02]->corrs;
         vector<PointCorrespondence> corrs12 = matchDataCache[oid12]->corrs;

         float const cosAngleThreshold = 1.0f; // We should have already filtered degerate triplets.
         computeScaleRatios(cosAngleThreshold, R01, T01, R12, T12, R20, T20,
                            corrs01, corrs12, corrs02, s012, s120, s201, weight);
      } // end if

#if !defined(USE_SYMMETRIC_SCALE_RATIOS)
      nzB.push_back(make_pair(3*n+0, pos12)); valsB.push_back(weight);
      nzB.push_back(make_pair(3*n+0, pos01)); valsB.push_back(-weight*s012);

      nzB.push_back(make_pair(3*n+1, pos01)); valsB.push_back(weight);
      nzB.push_back(make_pair(3*n+1, pos02)); valsB.push_back(-weight*s201);

      nzB.push_back(make_pair(3*n+2, pos02)); valsB.push_back(weight);
      nzB.push_back(make_pair(3*n+2, pos12)); valsB.push_back(-weight*s120);
#else
      nzB.push_back(make_pair(6*n+0, pos12)); valsB.push_back(weight);
      nzB.push_back(make_pair(6*n+0, pos01)); valsB.push_back(-weight*s012);

      nzB.push_back(make_pair(6*n+1, pos01)); valsB.push_back(weight);
      nzB.push_back(make_pair(6*n+1, pos02)); valsB.push_back(-weight*s201);

      nzB.push_back(make_pair(6*n+2, pos02)); valsB.push_back(weight);
      nzB.push_back(make_pair(6*n+2, pos12)); valsB.push_back(-weight*s120);

      nzB.push_back(make_pair(6*n+3, pos12)); valsB.push_back(-weight/s012);
      nzB.push_back(make_pair(6*n+3, pos01)); valsB.push_back(weight);

      nzB.push_back(make_pair(6*n+4, pos01)); valsB.push_back(-weight/s201);
      nzB.push_back(make_pair(6*n+4, pos02)); valsB.push_back(weight);

      nzB.push_back(make_pair(6*n+5, pos02)); valsB.push_back(-weight/s120);
      nzB.push_back(make_pair(6*n+5, pos12)); valsB.push_back(weight);
#endif
   } // end for (n)

   t.stop();
   t.print();
   cout << "done." << endl;

   cout << "computing " << nViewPairs << " consistent translation lengths from "
        << 3*nTriplets << " scale ratios..." << endl;

   _baseLengths.newsize(nViewPairs);

#if !defined(USE_SYMMETRIC_SCALE_RATIOS)
   CCS_Matrix<double> B(3*nTriplets, nViewPairs, nzB, valsB);
#else
   CCS_Matrix<double> B(6*nTriplets, nViewPairs, nzB, valsB);
#endif
   {
      Timer t("translation lengths");
      t.start();
      Matrix<double> V;
      Vector<double> sigma;
      SparseSymmetricEigConfig cfg;
      cfg.tolerance = 1e-8;
      cfg.maxArnoldiIterations = 100000;
      //cfg.nColumnsV = B.num_cols() / 2;
      computeSparseSVD(B, V3D_ARPACK_SMALLEST_MAGNITUDE_EIGENVALUES, 1, sigma, V, cfg);
      //computeSparseSVD(B, V3D_ARPACK_SMALLEST_EIGENVALUES, 1, sigma, V, cfg);
      V.getColumnSlice(0, V.num_rows(), 0, _baseLengths);
      t.stop();
      t.print();
   }

   cout << "done." << endl;
   if (_baseLengths[0] < 0.0) _baseLengths *= -1.0f;
   //cout << "l = "; displayVector(_baseLengths);
   _baseLengths *= sqrt(nViewPairs);

   for (int k = 0; k < nViewPairs; ++k)
   {
      normalizeVector(_relTranslationVec[k]);
      scaleVectorIP(_baseLengths[k], _relTranslationVec[k]);
   }
} // end SubmodelReconstruction::computeConsistentTranslationLengths()

void
SubmodelReconstruction::computeConsistentTranslations()
{
   _translations.resize(_nViews);

   int const nViewPairs = _viewPairs.size();

   vector<pair<int, int> > nzA;
   vector<double>          valsA;
   nzA.reserve((3+9)*nViewPairs + 3*_nViews);
   valsA.reserve((3+9)*nViewPairs + 3*_nViews);

   Vector<double> rhs(3*nViewPairs + 3);

   for (size_t n = 0; n < nViewPairs; ++n)
   {
      int const i0 = _viewPairVec[n].view0;
      int const i1 = _viewPairVec[n].view1;
      int const v0 = _viewIdBackMap[i0];
      int const v1 = _viewIdBackMap[i1];

      assert(_viewPairVecPosMap.find(ViewPair(i0, i1)) != _viewPairVecPosMap.end());
      int const pos01 = (*_viewPairVecPosMap.find(ViewPair(i0, i1))).second;

      Matrix3x3d const& R0  = _rotations[i0];
      Matrix3x3d const& R1  = _rotations[i1];
      Matrix3x3d const  R01 = R1 * R0.transposed();
      Vector3d   const& T01 = _relTranslationVec[pos01];

      //copyMatrixSlice(I, 0, 0, 3, 3, A, 3*n, 3*i1);
      nzA.push_back(make_pair(3*n+0, 3*i1+0)); valsA.push_back(1.0);
      nzA.push_back(make_pair(3*n+1, 3*i1+1)); valsA.push_back(1.0);
      nzA.push_back(make_pair(3*n+2, 3*i1+2)); valsA.push_back(1.0);

      //copyMatrixSlice((-1.0) * R01, 0, 0, 3, 3, A, 3*n, 3*i0);
      nzA.push_back(make_pair(3*n+0, 3*i0+0)); valsA.push_back(-R01[0][0]);
      nzA.push_back(make_pair(3*n+0, 3*i0+1)); valsA.push_back(-R01[0][1]);
      nzA.push_back(make_pair(3*n+0, 3*i0+2)); valsA.push_back(-R01[0][2]);
      nzA.push_back(make_pair(3*n+1, 3*i0+0)); valsA.push_back(-R01[1][0]);
      nzA.push_back(make_pair(3*n+1, 3*i0+1)); valsA.push_back(-R01[1][1]);
      nzA.push_back(make_pair(3*n+1, 3*i0+2)); valsA.push_back(-R01[1][2]);
      nzA.push_back(make_pair(3*n+2, 3*i0+0)); valsA.push_back(-R01[2][0]);
      nzA.push_back(make_pair(3*n+2, 3*i0+1)); valsA.push_back(-R01[2][1]);
      nzA.push_back(make_pair(3*n+2, 3*i0+2)); valsA.push_back(-R01[2][2]);

      rhs[3*n+0] = T01[0];
      rhs[3*n+1] = T01[1];
      rhs[3*n+2] = T01[2];
   } // end for (n)

   // Add equations such that the center of gravity is 0 to remove the global
   // translation ambiguity (and make AtA of full rank).
   for (int k = 0; k < _nViews; ++k)
   {
      int const row = 3*nViewPairs;
      nzA.push_back(make_pair(row+0, 3*k+0)); valsA.push_back(1.0);
      nzA.push_back(make_pair(row+1, 3*k+1)); valsA.push_back(1.0);
      nzA.push_back(make_pair(row+2, 3*k+2)); valsA.push_back(1.0);
   }
   rhs[3*nViewPairs+0] = 0.0;
   rhs[3*nViewPairs+1] = 0.0;
   rhs[3*nViewPairs+2] = 0.0;

   CCS_Matrix<double> A(3*nViewPairs + 3, 3*_nViews, nzA, valsA);

   cout << "computing consistent translations..." << endl;
   Timer t("translations");
   t.start();

   Matrix<double> AtA(3*_nViews, 3*_nViews);
   Vector<double> At_rhs(3*_nViews);
   multiply_At_A_SparseDense(A, AtA);
   multiply_At_v_Sparse(A, rhs, At_rhs);

   Cholesky<double> chol(AtA);
   Vector<double> X = chol.solve(At_rhs);
   cout << "done." << endl;

   t.stop();
   t.print();

   for (int i = 0; i < _nViews; ++i)
   {
      _translations[i][0] = X[3*i+0];
      _translations[i][1] = X[3*i+1];
      _translations[i][2] = X[3*i+2];
   } // end for (i)
} // end SubmodelReconstruction::computeConsistentTranslations()

void
SubmodelReconstruction::computeConsistentTranslations_L1(V3D::CachedStorage<TripletDataTable>& tripletDataCache,
                                                         std::map<ViewTripletKey, int> const& tripletOIDMap)
{
   _translations.resize(_nViews);

   int const nViewPairs = _viewPairs.size();

   vector<Vector3d> c_ji, c_jk;
   vector<Vector3i> ijks;

   for (vector<ViewTripletKey>::const_iterator p = _triplets.begin(); p != _triplets.end(); ++p)
   {
      int const v0 = (*p).views[0];
      int const v1 = (*p).views[1];
      int const v2 = (*p).views[2];

      int const i0 = _viewIdMap[v0];
      int const i1 = _viewIdMap[v1];
      int const i2 = _viewIdMap[v2];

      int const oid = tripletOIDMap.find(*p)->second;

      TripleReconstruction const * tripletData = tripletDataCache[oid];

      assert(tripletData->views[0] == v0);
      assert(tripletData->views[1] == v1);
      assert(tripletData->views[2] == v2);

      Matrix3x4d RT01 = getRelativeOrientation(tripletData->orientations[0], tripletData->orientations[1]);
      Matrix3x4d RT02 = getRelativeOrientation(tripletData->orientations[0], tripletData->orientations[2]);
      Matrix3x4d RT12 = getRelativeOrientation(tripletData->orientations[1], tripletData->orientations[2]);

      Matrix3x4d RT10 = getRelativeOrientation(tripletData->orientations[1], tripletData->orientations[0]);
      Matrix3x4d RT20 = getRelativeOrientation(tripletData->orientations[2], tripletData->orientations[0]);
      Matrix3x4d RT21 = getRelativeOrientation(tripletData->orientations[2], tripletData->orientations[1]);

      Vector3d T01 = RT01.col(3), T02 = RT02.col(3), T12 = RT12.col(3);
      Vector3d T10 = RT10.col(3), T20 = RT20.col(3), T21 = RT21.col(3);

      // Recall: c_j - c_i = -R_j^T T_j + R_i^T T_i = -R_j^T (T_j - R_ij T_i) = -R_j^T T_ij
      Vector3d c01 = _rotations[i1].transposed() * (-T01);
      Vector3d c02 = _rotations[i2].transposed() * (-T02);
      Vector3d c12 = _rotations[i2].transposed() * (-T12);

      Vector3d c10 = _rotations[i0].transposed() * (-T10);
      Vector3d c20 = _rotations[i0].transposed() * (-T20);
      Vector3d c21 = _rotations[i1].transposed() * (-T21);

      ijks.push_back(makeVector3(i0, i1, i2)); c_ji.push_back(c10); c_jk.push_back(c12);
      ijks.push_back(makeVector3(i2, i0, i1)); c_ji.push_back(c02); c_jk.push_back(c01);
      ijks.push_back(makeVector3(i1, i2, i0)); c_ji.push_back(c21); c_jk.push_back(c20);
   } // end for (p)

   vector<Vector3d> centers(_nViews);

   {
      MultiViewInitializationParams_BOS params;
      params.verbose = true;
      params.nIterations = 10000;

      Timer t("computeConsistentCameraCenters()");
      t.start();
      //computeConsistentCameraCenters_L1(c_ji, c_jk, ijks, centers, true);
      computeConsistentCameraCenters_L2_BOS(c_ji, c_jk, ijks, centers, params);
      t.stop();
      t.print();
   }

   for (int i = 0; i < _nViews; ++i) _translations[i] = _rotations[i] * (-centers[i]);
} // end SubmodelReconstruction::computeConsistentTranslations_L1()

void
SubmodelReconstruction::generateSparseReconstruction(std::vector<V3D::PointCorrespondence> const& allCorrs)
{
   using namespace std;
   using namespace V3D;

   vector<PointCorrespondence> corrs;
   for (size_t k = 0; k < allCorrs.size(); ++k)
   {
      PointCorrespondence c = allCorrs[k];

      map<int, int>::const_iterator p1 = _viewIdMap.find(c.left.view);
      map<int, int>::const_iterator p2 = _viewIdMap.find(c.right.view);
               

      if (p1 != _viewIdMap.end() && p2 != _viewIdMap.end())
      {
         // Already bring view ids to the [0..N-1] range
         c.left.view = p1->second;
         c.right.view = p2->second;
         corrs.push_back(c);
      }
   } // end for (k)

   _sparseReconstruction.clear();
   int const nRequiredViews = 3;
   TriangulatedPoint::connectTracks(corrs, _sparseReconstruction, nRequiredViews);
   cout << "sparse reconstruction (before logical filtering) has " << _sparseReconstruction.size() << " 3D points." << endl;
   filterConsistentSparsePoints(_sparseReconstruction);
   cout << "sparse reconstruction (after logical filtering) has " << _sparseReconstruction.size() << " 3D points." << endl;

   _cameras.resize(_nViews);
   Matrix3x3d K; makeIdentityMatrix(K);

   for (int i = 0; i < _nViews; ++i)
   {
      _cameras[i].setIntrinsic(K);
      _cameras[i].setRotation(_rotations[i]);
      _cameras[i].setTranslation(_translations[i]);
   }

   for (int i = 0; i < _sparseReconstruction.size(); ++i)
      _sparseReconstruction[i].pos = triangulateLinear(_cameras, _sparseReconstruction[i].measurements);
} // end SubmodelReconstruction::generateSparseReconstruction()
