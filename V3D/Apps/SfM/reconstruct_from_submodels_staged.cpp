#include "reconstruction_common.h"

#include "Base/v3d_cfgfile.h"
#include "Base/v3d_timer.h"
#include "Base/v3d_utilities.h"
#include "Math/v3d_sparseeig.h"
#include "Geometry/v3d_mviewutilities.h"
#include "Geometry/v3d_metricbundle.h"

#include <queue>
#include <iostream>
#include <sstream>
#include <list>

using namespace std;
using namespace V3D;

namespace
{

   void
   extractConnectedComponent(std::map<int, std::set<int> > const& edgeMap,
                             std::set<std::pair<int, int> >& unhandledEdges,
                             std::set<std::pair<int, int> >& connectedEdges,
                             std::set<int>& handledNodes)
   {
      // Breadth-first search for connected components
      using namespace std;

      connectedEdges.clear();
      handledNodes.clear();

      list<int> nodeQueue;

      pair<int, int> startEdge = *unhandledEdges.begin();
      unhandledEdges.erase(unhandledEdges.begin());
      connectedEdges.insert(startEdge);
      nodeQueue.push_back(startEdge.first);
      nodeQueue.push_back(startEdge.second);

      while (!nodeQueue.empty())
      {
         int curNode = nodeQueue.front();
         nodeQueue.pop_front();

         handledNodes.insert(curNode);

         map<int, set<int> >::const_iterator p = edgeMap.find(curNode);
         assert(p != edgeMap.end());
         set<int> const& otherNodes = (*p).second;

         for (set<int>::const_iterator q = otherNodes.begin(); q != otherNodes.end(); ++q)
         {
            int i0 = curNode;
            int i1 = *q;
            sort2(i0, i1);
            pair<int, int> key(i0, i1);

            if (connectedEdges.find(key) != connectedEdges.end()) continue;
            if (unhandledEdges.find(key) == unhandledEdges.end()) continue;

            connectedEdges.insert(key);
            unhandledEdges.erase(key);

            if (handledNodes.find(i0) == handledNodes.end()) nodeQueue.push_back(i0);
            if (handledNodes.find(i1) == handledNodes.end()) nodeQueue.push_back(i1);
         } // end for (q)
      } // end while
   } // end computeConntectedComponent()

} // end namespace <>

int
main(int argc, char * argv[])
{
   if (argc != 3)
   {
      cerr << "Usage: " << argv[0] << " <config file> <int stage>" << endl;
      return -1;
   }

   try
   {
      cout << "Reading config file..." << endl;
      ConfigurationFile cf(argv[1]);

      bool   const applyBA     = cf.get("APPLY_BUNDLE", true);
      int    const bundleMode  = cf.get("BUNDLE_MODE", FULL_BUNDLE_METRIC);

      double maxReproError = cf.get("MAX_REPROJECTION_ERROR_RECONSTRUCT", -1.0);
      if (maxReproError < 0) maxReproError = cf.get("MAX_REPROJECTION_ERROR_TRIPLET", -1.0);

      int nRequiredAlignmentPoints = cf.get("REQUIRED_ALIGNMENT_POINTS_RECONSTRUCT", -1);
      if (nRequiredAlignmentPoints < 0) nRequiredAlignmentPoints = cf.get("REQUIRED_ALIGNMENT_POINTS", 50);
      cout << "done." << endl;

      int const stage = atoi(argv[2]);

      CalibrationDatabase calibDb("calibration_db.txt");

      cout << "Connecting to DBs..." << endl;

      char dbName[200];
      if (stage == 0)
         strcpy(dbName, "submodels.db");
      else
         sprintf(dbName, "submodels-stage_%i.db", stage);

      SQLite3_Database submodelsDB(dbName);

      SubmodelsTable submodelsTable = submodelsDB.getTable<SubmodelReconstruction>("submodels_data");
      CachedStorage<SubmodelsTable> submodelsCache(submodelsTable, 100);
      cout << "done." << endl;

      int const nAllSubModels = submodelsTable.size();
      cout << "nAllSubModels = " << nAllSubModels << endl;

      vector<Matrix3x3d> allRelRotations; // between submodels
      vector<Vector3d>   allRelTranslations;
      vector<double>     allRelScales;
      vector<double>     allRelWeights;
      vector<pair<int, int> >  allSubModelPairs;
      map<pair<int, int>, int> subModelPairPosMap;

      vector<vector<int> >          allSubModelViews;
      vector<vector<CameraMatrix> > allSubModelCameras;

      SerializableVector<PointCorrespondence> allCorrs;
      serializeDataFromFile("allcorrs.bin", allCorrs);
      cout << "allCorrs.size() = " << allCorrs.size() << endl;

      for (int j1 = 0; j1 < nAllSubModels; ++j1)
      {
         SubmodelReconstruction const& subModel1 = *submodelsCache[j1];
         allSubModelViews.push_back(subModel1._viewIdBackMap);
         allSubModelCameras.push_back(subModel1._cameras);
      } // end for (j1)

      {
         double const minEdgeWeight = sqrt(double(nRequiredAlignmentPoints));

         char name[200];
         if (stage == 0)
            strcpy(name, "subcomp_alignment.bin");
         else
            sprintf(name, "subcomp_alignment-stage_%i.bin", stage);

         ifstream is(name, ios::binary);
         BinaryIStreamArchive ar(is);

         vector<int> allSubModelPairsTmp;

         vector<Matrix3x3d> allRelRotationsTmp;
         vector<Vector3d>   allRelTranslationsTmp;
         vector<double>     allRelScalesTmp;
         vector<double>     allRelWeightsTmp;

         serializeVector(allRelRotationsTmp, ar);
         serializeVector(allRelTranslationsTmp, ar);
         serializeVector(allRelScalesTmp, ar);
         serializeVector(allRelWeightsTmp, ar);
         serializeVector(allSubModelPairsTmp, ar);

         cout << "Read in " << allSubModelPairsTmp.size() << " pairwise transformations between submodels." << endl;

         for (size_t k = 0; k < allSubModelPairsTmp.size(); ++k)
         {
            if (allRelWeightsTmp[k] < minEdgeWeight) continue;

            int const pair = allSubModelPairsTmp[k];
            int const j1 = pair & 0xffff;
            int const j2 = (pair >> 16);

            if (edgesBlackList.find(make_pair(j1, j2)) != edgesBlackList.end() ||
                edgesBlackList.find(make_pair(j2, j1)) != edgesBlackList.end()) continue;

            subModelPairPosMap.insert(make_pair(make_pair(j1, j2), allSubModelPairs.size()));
            allSubModelPairs.push_back(make_pair(j1, j2));
            allRelRotations.push_back(allRelRotationsTmp[k]);
            allRelTranslations.push_back(allRelTranslationsTmp[k]);
            allRelScales.push_back(allRelScalesTmp[k]);
            allRelWeights.push_back(allRelWeightsTmp[k]);
         } // end for (k)
         cout << allSubModelPairs.size() << " are remaining." << endl;
      } // end scope

      vector<set<int> > connComponents;
      map<int, set<int> > mstAdjacencyMap;
      if (!useRandomSubmodelGrowing)
      {
         // Compute the MSTs and number of components for the epipolar/camera graph

         vector<pair<int, int> > edges;
         vector<double> weights;

         for (set<ViewPair>::const_iterator p = allSubModelPairs.begin(); p != allSubModelPairs.end(); ++p)
         {
            edges.push_back(make_pair(p->view0, p->view1));
            weights.push_back(viewPairWeightMap.find(*p)->second);
         }

         vector<pair<int, int> > mstEdges;

         getMinimumSpanningForest(edges, weights, mstEdges, connComponents);
         cout << "Camera graph has " << connComponents.size() << " connected component(s)." << endl;

         for (size_t i = 0; i < mstEdges.size(); ++i)
         {
            mstAdjacencyMap[mstEdges[i].first].insert(mstEdges[i].second);
            mstAdjacencyMap[mstEdges[i].second].insert(mstEdges[i].first);
         }
      } // end scope


      vector<pair<int, int> > filteredEdges;
      vector<set<int> > connSubModels;

      if (useMST_Upgrade)
      {
         getMinimumSpanningForest(allSubModelPairs, allRelWeights, filteredEdges, connSubModels);
      }
      else
      {
         // Use global alignment of submodels
         filteredEdges = allSubModelPairs;

         map<int, set<int> > subcompEdgeMap;
         set<pair<int, int> > subcompEdges;
         for (int k = 0; k < allSubModelPairs.size(); ++k)
         {
            int const i0 = allSubModelPairs[k].first;
            int const i1 = allSubModelPairs[k].second;
            subcompEdges.insert(allSubModelPairs[k]);
            subcompEdgeMap[i0].insert(i1);
            subcompEdgeMap[i1].insert(i0);
         } // end for (k)

         while (!subcompEdges.empty())
         {
            set<int>             connectedNodes;
            set<pair<int, int> > connectedEdges;
            extractConnectedComponent(subcompEdgeMap, subcompEdges, connectedEdges, connectedNodes);
            connSubModels.push_back(connectedNodes);
         }
      } // end if (useMST_Upgrade)

      cout << "Total number of components/models: " << connSubModels.size() << endl;

      for (int componentId = 0; componentId < connSubModels.size(); ++componentId)
      {
         set<int> const& connectedNodes = connSubModels[componentId];
         set<pair<int, int> > connectedEdges;
         // Filter MST edges that are in this component
         for (size_t k = 0; k < filteredEdges.size(); ++k)
         {
            int const m0 = filteredEdges[k].first;
            int const m1 = filteredEdges[k].second;
            if (connectedNodes.find(m0) != connectedNodes.end() &&
                connectedNodes.find(m1) != connectedNodes.end())
               connectedEdges.insert(filteredEdges[k]);
         }

         int const nSubModels = connectedNodes.size();

         cout << "This component has " << connectedNodes.size() << " sub-components: ";
         for (set<int>::const_iterator p = connectedNodes.begin(); p != connectedNodes.end(); ++p)
            cout << (*p) << " ";
         cout << endl;

         vector<vector<int> >          subModelViews;
         vector<vector<CameraMatrix> > subModelCameras;

         // Map submodel ids to the range [0, N-1]
         map<int, int> subcompIdMap;
         vector<int> subcompIdBackMap(nSubModels);
         for (set<int>::const_iterator p = connectedNodes.begin(); p != connectedNodes.end(); ++p)
         {
            int newId = subcompIdMap.size();
            subcompIdMap.insert(make_pair(*p, newId));
            subcompIdBackMap[newId] = *p;

            subModelViews.push_back(allSubModelViews[*p]);
            subModelCameras.push_back(allSubModelCameras[*p]);
         }

         vector<Matrix3x3d> relRotations; // between submodels
         vector<Vector3d>   relTranslations;
         vector<double>     relScales;
         vector<double>     relWeights;
         vector<pair<int, int> > subModelPairs;

         for (set<pair<int, int> >::const_iterator p = connectedEdges.begin(); p != connectedEdges.end(); ++p)
         {
            pair<int, int> const& pair = *p;
            int const pos = subModelPairPosMap.find(pair)->second;

            int const i0 = subcompIdMap.find(pair.first)->second;
            int const i1 = subcompIdMap.find(pair.second)->second;

            subModelPairs.push_back(make_pair(i0, i1));
            relRotations.push_back(allRelRotations[pos]);
            relTranslations.push_back(allRelTranslations[pos]);
            relScales.push_back(allRelScales[pos]);
            relWeights.push_back(allRelWeights[pos]);
         }
         int const nPairs = subModelPairs.size();

         vector<Matrix3x3d> rotations;
         computeConsistentRotations(nSubModels, relRotations, subModelPairs, rotations);

         Vector<double> scales(nSubModels);
         //cout << "relScales = "; displayVector(relScales);

         {
            vector<pair<int, int> > nzB;
            vector<double>          valsB;
            nzB.reserve(nPairs);
            valsB.reserve(nPairs);

            for (int k = 0; k < nPairs; ++k)
            {
               int const i0 = subModelPairs[k].first;
               int const i1 = subModelPairs[k].second;

               double const weight = 1.0; //relWeights[k];

               nzB.push_back(make_pair(k, i1)); valsB.push_back(weight);
               nzB.push_back(make_pair(k, i0)); valsB.push_back(-weight*relScales[k]);
            } // end for (k)

            CCS_Matrix<double> B(nPairs, nSubModels, nzB, valsB);
            Matrix<double> V;
            Vector<double> sigma;
            SparseSymmetricEigConfig cfg;
            cfg.tolerance = 1e-8;
            cfg.maxArnoldiIterations = 100000;
            //cfg.nColumnsV = B.num_cols() / 2;
            computeSparseSVD(B, V3D_ARPACK_SMALLEST_MAGNITUDE_EIGENVALUES, 1, sigma, V, cfg);
            //computeSparseSVD(B, V3D_ARPACK_SMALLEST_EIGENVALUES, 1, sigma, V, cfg);
            V.getColumnSlice(0, V.num_rows(), 0, scales);

            if (scales[0] < 0.0) scales *= -1.0f;
            cout << "scales = "; displayVector(scales);
            scales *= sqrt(nSubModels);
         } // end scope consistent scales

         vector<Vector3d> translations(nSubModels);

         {
            vector<pair<int, int> > nzA;
            vector<double>          valsA;
            nzA.reserve((3+9)*nPairs + 3*nSubModels);
            valsA.reserve((3+9)*nPairs + 3*nSubModels);

            Vector<double> rhs(3*nPairs + 3);

            for (size_t k = 0; k < nPairs; ++k)
            {
               int const i0 = subModelPairs[k].first;
               int const i1 = subModelPairs[k].second;

               Matrix3x3d const& R0  = rotations[i0];
               Matrix3x3d const& R1  = rotations[i1];
               Matrix3x3d        R01 = R1 * R0.transposed();
               Vector3d          T01 = relTranslations[k];

               double const s01 = scales[i1]/scales[i0];
               scaleMatrixIP(s01, R01);
               scaleVectorIP(s01, T01);

               //copyMatrixSlice(I, 0, 0, 3, 3, A, 3*n, 3*i1);
               nzA.push_back(make_pair(3*k+0, 3*i1+0)); valsA.push_back(1.0);
               nzA.push_back(make_pair(3*k+1, 3*i1+1)); valsA.push_back(1.0);
               nzA.push_back(make_pair(3*k+2, 3*i1+2)); valsA.push_back(1.0);

               //copyMatrixSlice((-1.0) * R01, 0, 0, 3, 3, A, 3*n, 3*i0);
               nzA.push_back(make_pair(3*k+0, 3*i0+0)); valsA.push_back(-R01[0][0]);
               nzA.push_back(make_pair(3*k+0, 3*i0+1)); valsA.push_back(-R01[0][1]);
               nzA.push_back(make_pair(3*k+0, 3*i0+2)); valsA.push_back(-R01[0][2]);
               nzA.push_back(make_pair(3*k+1, 3*i0+0)); valsA.push_back(-R01[1][0]);
               nzA.push_back(make_pair(3*k+1, 3*i0+1)); valsA.push_back(-R01[1][1]);
               nzA.push_back(make_pair(3*k+1, 3*i0+2)); valsA.push_back(-R01[1][2]);
               nzA.push_back(make_pair(3*k+2, 3*i0+0)); valsA.push_back(-R01[2][0]);
               nzA.push_back(make_pair(3*k+2, 3*i0+1)); valsA.push_back(-R01[2][1]);
               nzA.push_back(make_pair(3*k+2, 3*i0+2)); valsA.push_back(-R01[2][2]);

               rhs[3*k+0] = T01[0];
               rhs[3*k+1] = T01[1];
               rhs[3*k+2] = T01[2];
            } // end for (k)

            // Add equations such that the center of gravity is 0 to remove the global
            // translation ambiguity (and make AtA of full rank).
            for (int k = 0; k < nSubModels; ++k)
            {
               int const row = 3*nPairs;
               nzA.push_back(make_pair(row+0, 3*k+0)); valsA.push_back(1.0);
               nzA.push_back(make_pair(row+1, 3*k+1)); valsA.push_back(1.0);
               nzA.push_back(make_pair(row+2, 3*k+2)); valsA.push_back(1.0);
            }
            rhs[3*nPairs+0] = 0.0;
            rhs[3*nPairs+1] = 0.0;
            rhs[3*nPairs+2] = 0.0;

            CCS_Matrix<double> A(3*nPairs + 3, 3*nSubModels, nzA, valsA);

            Matrix<double> AtA(3*nSubModels, 3*nSubModels);
            Vector<double> At_rhs(3*nSubModels);
            multiply_At_A_SparseDense(A, AtA);
            multiply_At_v_Sparse(A, rhs, At_rhs);

            Cholesky<double> chol(AtA);
            Vector<double> X = chol.solve(At_rhs);

            for (int i = 0; i < nSubModels; ++i)
            {
               translations[i][0] = X[3*i+0];
               translations[i][1] = X[3*i+1];
               translations[i][2] = X[3*i+2];
            } // end for (i)
         } // end scope consistent translations

         set<int> compViewsIds;
         for (int k = 0; k < nSubModels; ++k)
         {
            for (size_t l = 0; l < subModelViews[k].size(); ++l)
               compViewsIds.insert(subModelViews[k][l]);
         }
         cout << "Views in this component: ";
         for (set<int>::const_iterator p = compViewsIds.begin(); p != compViewsIds.end(); ++p)
            cout << *p << " ";
         cout << endl;

         int const nCompViews = compViewsIds.size();

         // Map view ids to the range [0, N-1]
         map<int, int> compViewIdMap;
         vector<int> compViewIdBackMap(nCompViews);
         for (set<int>::const_iterator p = compViewsIds.begin(); p != compViewsIds.end(); ++p)
         {
            int newId = compViewIdMap.size();
            compViewIdMap.insert(make_pair(*p, newId));
            compViewIdBackMap[newId] = *p;
         }

         vector<CameraMatrix> compCameras(nCompViews);
         vector<int>          cameraAssignmentCount(nCompViews, 0);

         for (int k = 0; k < nSubModels; ++k)
         {
            Matrix3x3d const& R_xform = rotations[k];
            Vector3d   const& T_xform = translations[k];
            double            s_xform = scales[k];

            vector<CameraMatrix>& cameras = subModelCameras[k];

            for (int i = 0; i < cameras.size(); ++i)
            {
               Matrix3x3d const R_cam = cameras[i].getRotation();
               Vector3d   const T_cam = cameras[i].getTranslation();

               Matrix3x3d R_new = R_cam * R_xform;
               //Vector3d   T_new = s_xform * (R_cam * T_xform + T_cam);
               Vector3d   T_new = (1.0/s_xform) * (R_cam * T_xform + T_cam);
               //Vector3d   T_new = R_cam * T_xform + (1.0/s_xform) * T_cam;
               cameras[i].setRotation(R_new);
               cameras[i].setTranslation(T_new);

               int const subModelId = subcompIdBackMap[k]; // This is the global submodel ID
               int const viewId = subModelViews[k][i]; // This is the global view ID

               if (membershipBlackList.find(make_pair(subModelId, viewId)) != membershipBlackList.end())
                  continue;

               int const compViewId = compViewIdMap[viewId];

               CameraMatrix const& srcCam = cameras[i];
               CameraMatrix&       dstCam = compCameras[compViewId];

               if (cameraAssignmentCount[compViewId] == 0)
               {
                  dstCam = srcCam;
                  cameraAssignmentCount[compViewId] = 1;
               }
               else
               {
                  double const N = cameraAssignmentCount[compViewId];
                  // Average the translation
                  dstCam.setTranslation(1.0/(N+1.0) * (N*dstCam.getTranslation() + srcCam.getTranslation()));

                  // Average the rotation
                  Vector4d Q_src, Q_dst;
                  createQuaternionFromRotationMatrix(srcCam.getRotation(), Q_src);
                  createQuaternionFromRotationMatrix(dstCam.getRotation(), Q_dst);
                  Q_dst = N*Q_dst + Q_src;
                  Matrix3x3d R_dst;
                  createRotationMatrixFromQuaternion(Q_dst, R_dst);
                  dstCam.setRotation(R_dst);

                  ++cameraAssignmentCount[compViewId];
               }
            } // end for (i)
         } // end for (k)

         SerializableVector<PointCorrespondence> compCorrs;

         for (size_t k = 0; k < allCorrs.size(); ++k)
         {
            PointCorrespondence c = allCorrs[k];

            int const v0 = c.left.view;
            int const v1 = c.right.view;

            map<int, int>::const_iterator p0 = compViewIdMap.find(c.left.view);
            map<int, int>::const_iterator p1 = compViewIdMap.find(c.right.view);

            if (p0 == compViewIdMap.end() || p1 == compViewIdMap.end()) continue;

            // Bring view ids to the [0..N-1] range
            c.left.view = p0->second;
            c.right.view = p1->second;
            compCorrs.push_back(c);
         } // end for (k)
         cout << compCorrs.size() << " correspondences in this component." << endl;

         vector<TriangulatedPoint> sparseReconstruction;
         int const nRequiredViews = 3;
         TriangulatedPoint::connectTracks(compCorrs, sparseReconstruction, nRequiredViews);
         cout << "sparse reconstruction (before logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;
         filterConsistentSparsePoints(sparseReconstruction);
         cout << "sparse reconstruction (after logical filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         for (int i = 0; i < sparseReconstruction.size(); ++i)
            sparseReconstruction[i].pos = triangulateLinear(compCameras, sparseReconstruction[i].measurements);

         filterInlierSparsePoints(calibDb, compViewIdBackMap, maxReproError, nRequiredViews, compCameras, sparseReconstruction);
         cout << "sparse reconstruction (after geometric filtering) has " << sparseReconstruction.size() << " 3D points." << endl;

         {
            vector<int> camCounts(compCameras.size(), 0);
            for (int j = 0; j < sparseReconstruction.size(); ++j)
            {
               TriangulatedPoint const& X = sparseReconstruction[j];
               for (int k = 0; k < X.measurements.size(); ++k)
                  ++camCounts[X.measurements[k].view];
            }
            cout << "camCounts = "; displayVector(camCounts);
         } // end scope

         showAccuracyInformation(calibDb, compViewIdBackMap, compCameras, sparseReconstruction);

         BundlePointStructure bundleStruct(sparseReconstruction);

         if (bundleStruct.points3d.size() == 0) continue;

         char wrlName[200];
         sprintf(wrlName, "points3d-%i.wrl", componentId);
         writePointsToVRML(bundleStruct.points3d, wrlName);

         if (applyBA)
         {
            cout << "compCameras.size() = " << compCameras.size() << endl;
            cout << "bundleStruct.points3d.size() = " << bundleStruct.points3d.size() << endl;
            cout << "bundleStruct.measurements.size() = " << bundleStruct.measurements.size() << endl;

            {
#if 1
               vector<CameraMatrix> savedCameras(compCameras);
               vector<Vector3d> savedPoints(bundleStruct.points3d);
               ScopedBundleExtrinsicNormalizer extNormalizer(savedCameras, savedPoints);

               StdDistortionFunction distortion;
               Matrix3x3d K; makeIdentityMatrix(K);
               V3D::CommonInternalsMetricBundleOptimizer opt(bundleMode, 1.0, K, distortion,
                                                             savedCameras, savedPoints, bundleStruct.measurements,
                                                             bundleStruct.correspondingView, bundleStruct.correspondingPoint);
               V3D::optimizerVerbosenessLevel = 1;
               opt.tau = 1e-3;
               opt.maxIterations = 100;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
               if (bundleMode > 0) cout << "New intrinsic: "; displayMatrix(K);

               // Keep the normalized positions, but scale everything
               double const scale = compCameras.size() + savedPoints.size();
               for (int i = 0; i < savedCameras.size(); ++i)
               {
                  Vector3d c = savedCameras[i].cameraCenter();
                  savedCameras[i].setCameraCenter(scale * c);
               }
               for (int j = 0; j < savedPoints.size(); ++j)
                  scaleVectorIP(scale, savedPoints[j]);

               compCameras = savedCameras;
               bundleStruct.points3d = savedPoints;
#else
               StdDistortionFunction distortion;
               Matrix3x3d K; makeIdentityMatrix(K);
               V3D::CommonInternalsMetricBundleOptimizer opt(bundleMode, 1.0, K, distortion,
                                                             compCameras, bundleStruct.points3d, bundleStruct.measurements,
                                                             bundleStruct.correspondingView, bundleStruct.correspondingPoint);
               V3D::optimizerVerbosenessLevel = 1;
               opt.tau = 1e-3;
               opt.maxIterations = 100;
               opt.minimize();
               cout << "optimizer status = " << opt.status << endl;
               if (bundleMode > 0) cout << "New intrinsic: "; displayMatrix(K);
#endif
            } // end scope

            bundleStruct.createPointStructure(sparseReconstruction);

            sprintf(wrlName, "ba-points3d-%i.wrl", componentId);
            cout << "Writing " << wrlName << endl;
            writeGoodPointsToVRML(calibDb, compViewIdBackMap, compCameras, sparseReconstruction, wrlName, 2.0, 3);
         } // end if (applyBA)

         SerializableVector<TriangulatedPoint> finalReconstruction;
         bundleStruct.createPointStructure(finalReconstruction, true);

         showAccuracyInformation(calibDb, compViewIdBackMap, compCameras, finalReconstruction);

//          {
//             char name[200];
//             sprintf(name, "model-%i-cams.txt", componentId);
//             ofstream os(name);
//             os << nCompViews << endl;
//             for (int i = 0; i < nCompViews; ++i)
//             {
//                os << compViewIdBackMap[i] << " ";
//                Matrix3x4d RT = compCameras[i].getOrientation();
//                os << RT[0][0] << " " << RT[0][1] << " " << RT[0][2] << " " << RT[0][3] << endl;
//                os << RT[1][0] << " " << RT[1][1] << " " << RT[1][2] << " " << RT[1][3] << endl;
//                os << RT[2][0] << " " << RT[2][1] << " " << RT[2][2] << " " << RT[2][3] << endl;
//             }
//          } // end scope

//          {
//             char name[200];
//             sprintf(name, "model-%i-points.txt", componentId);
//             ofstream os(name);

//             os << finalReconstruction.size() << endl;
//             os.precision(10);

//             Vector3f p;

//             for (size_t i = 0; i < finalReconstruction.size(); ++i)
//             {
//                TriangulatedPoint const& X = finalReconstruction[i];
//                os << X.pos[0] << " " << X.pos[1] << " " << X.pos[2] << " " << X.measurements.size() << " ";
//                for (int k = 0; k < X.measurements.size(); ++k)
//                {
//                   PointMeasurement m = X.measurements[k];

//                   Matrix3x3d const K = calibDb.getIntrinsic(compViewIdBackMap[m.view]);
//                   multiply_A_v_affine(K, m.pos, p);

//                   m.pos[0] = p[0]; m.pos[1] = p[1];
//                   os << m.view << " " << m.id << " " << m.pos[0] << " " << m.pos[1] << " ";
//                }
//                os << endl;
//             }
//          } // end scope
      } // end for (componentId)
   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (std::exception exn)
   {
      cerr << "Exception caught: " << exn.what() << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   return 0;
} // end main()
