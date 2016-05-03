#include "Geometry/v3d_mviewutilities.h"
#include "Math/v3d_linear_tnt.h"

using namespace std;
using namespace V3D;

namespace
{
   template <typename T, int N>
   inline void
   projectOnSimplex(T * u)
   {
      T const negEps = 0.0;

      bool isActive[N];
      for (int i = 0; i < N; ++i) isActive[i] = false;

      for (int iter = 0; iter < N; ++iter)
      {
         T sum = 0;
         int K = 0;
         for (int i = 0; i < N; ++i)
            if (!isActive[i])
            {
               sum += u[i];
               ++K;
            }
         T const mean = (sum - T(1)) / K;
         for (int i = 0; i < N; ++i)
            if (!isActive[i]) u[i] -= mean;

         bool done = true;

         for (int i = 0; i < N; ++i)
         {
            if (u[i] < negEps)
            {
               u[i] = T(0);
               isActive[i] = true;
               done = false;
            }
         }
         if (done) break;
      } // end for
   } // end projectOnSimplex()

   // Project Z into the space of symm. PSD 4x4 matrices with trace(Z)=1.
   // This is the convex hull of SO(3).
   inline void
   projectConvHull_SO3(Matrix4x4d& Z)
   {
      // Symmetrize input matrix first
      Matrix<double> ZZ(4, 4);
      copyMatrix(0.5 * (Z + Z.transposed()), ZZ);

      Eigenvalue<double> eig(ZZ);
      Matrix4x4d V;
      copyMatrix(eig.getV(), V);
      Vector<double> lambda(4);
      eig.getRealEigenvalues(lambda);
      projectOnSimplex<double, 4>(&lambda[0]);

      Matrix4x4d D;
      makeZeroMatrix(D);
      for (int i = 0; i < 4; ++i) D[i][i] = lambda[i];
      Z = V*D*V.transposed();
   } // end projectConvHull_SO3()

   inline void
   proxDataResidual_Frobenius(double const sigma, double& t, Matrix3x3d& A)
   {
      // Matrix3x3d I;
      // makeIdentityMatrix(I);
      // double const R = matrixNormFrobenius(A - I);
      double const R = matrixNormFrobenius(A);

      if (R <= sigma + t) return; // nothing to do

      double const lambda = 0.5*(sigma + t + R)/R;
      A = lambda * A;// + (1-lambda)*I;
      t = lambda*R - sigma;
   } // end proxDataResidual_Frobenius()

   /* To convert from quaterions to rotation matrices
    *         0    1    2    3     4    5    6     7    8    9
    *       Z11  Z12  Z13  Z14   Z22  Z23  Z24   Z33  Z34  Z44
    * Q = [   1                    1              -1        -1 // R11, R12 etc. to ...
    *                       -2          2
    *                   2                    2
    *                        2          2
    *         1                   -1               1        -1
    *             -2                                    2
    *                  -2                    2
    *              2                                    2
    *         1                   -1              -1         1 ] // ... R33
    *
    * or column major version, R11, R21, ... R33:
    *         0    1    2    3     4    5    6     7    8    9
    *       Z11  Z12  Z13  Z14   Z22  Z23  Z24   Z33  Z34  Z44
    * Q = [   1                    1              -1        -1
    *                        2          2
    *                  -2                    2
    *                       -2          2
    *         1                   -1               1        -1
    *              2                                    2
    *                   2                    2
    *             -2                                    2
    *         1                   -1              -1         1 ] // ... R33
    */

   // vectorize the upper part of a 4x4 symmetric matrix
   inline void
   vecSymm3x3d(Matrix4x4d const& A, double * y)
   {
      y[0] = A[0][0]; y[1] = A[0][1]; y[2] = A[0][2]; y[3] = A[0][3];
      y[4] = A[1][1]; y[5] = A[1][2]; y[6] = A[1][3];
      y[7] = A[2][2]; y[8] = A[2][3];
      y[9] = A[3][3];
   }

   inline void
   unvecSymm3x3d(double const * y, Matrix4x4d& A)
   {
      A[0][0] = y[0]; A[0][1] = y[1]; A[0][2] = y[2]; A[0][3] = y[3];
      A[1][1] = y[4]; A[1][2] = y[5]; A[1][3] = y[6];
      A[2][2] = y[7]; A[2][3] = y[8];
      A[3][3] = y[9];

      for (int i = 0; i < 4; ++i)
         for (int j = i+1; j < 4; ++j) A[j][i] = A[i][j];
   }

   // Enforce R2 + R12*R1 + A = 0, where Ri = Q*Zi
   inline void
   proxConsistency(Matrix3x3d const& R12, Matrix4x4d& Z1, Matrix4x4d& Z2, Matrix3x3d& A)
   {
      InlineMatrix<double, 3, 10> Q1, Q2, Q3, RQ1, RQ2, RQ3; // columns of Q, Qi: R^10 -> R^3
      makeZeroMatrix(Q1);
      makeZeroMatrix(Q2);
      makeZeroMatrix(Q3);

      /*
       *     0    1    2    3     4    5    6     7    8    9
       *   Z11  Z12  Z13  Z14   Z22  Z23  Z24   Z33  Z34  Z44
       */
      Q1[0][0] = 1;  Q1[0][4] = 1; Q1[0][7] = -1; Q1[0][9] = -1;
      Q1[1][3] = 2;  Q1[1][5] = 2;
      Q1[2][2] = -2; Q1[2][6] = 2;

      Q2[0][3] = -2; Q2[0][5] = 2;
      Q2[1][0] = 1;  Q2[1][4] = -1; Q2[1][7] = 1; Q2[1][9] = -1;
      Q2[2][1] = 2;  Q2[2][8] = 2;

      Q3[0][2] = 2;  Q3[0][6] = 2;
      Q3[1][1] = -2; Q3[1][8] = 2;
      Q3[2][0] = 1;  Q3[2][4] = -1; Q3[2][7] = -1; Q3[2][9] = 1;

      multiply_A_B(R12, Q1, RQ1);
      multiply_A_B(R12, Q2, RQ2);
      multiply_A_B(R12, Q3, RQ3);

      scaleMatrixIP(-1, RQ1);
      scaleMatrixIP(-1, RQ2);
      scaleMatrixIP(-1, RQ3);

      InlineMatrix<double, 9, 2*10+9> B;
      makeZeroMatrix(B);

      copyMatrixSlice(RQ1, 0, 0, 3, 10, B, 0, 0);
      copyMatrixSlice(RQ2, 0, 0, 3, 10, B, 3, 0);
      copyMatrixSlice(RQ3, 0, 0, 3, 10, B, 6, 0);

      copyMatrixSlice(Q1, 0, 0, 3, 10, B, 0, 10);
      copyMatrixSlice(Q2, 0, 0, 3, 10, B, 3, 10);
      copyMatrixSlice(Q3, 0, 0, 3, 10, B, 6, 10);

      for (int i = 0; i < 9; ++i) B[i][20+i] = 1;

      // Recall: min_x ||x-y||^2 s.t. Bx = 0 is given by x = y - B'*inv(B*B')*B*y
      Matrix<double> BBt(9, 9);
      multiply_A_Bt(B, B, BBt);
      Cholesky<double> chol(BBt);
      Vector<double> y(29);
      vecSymm3x3d(Z1, &y[0]);
      vecSymm3x3d(Z2, &y[10]);
      y[20] = A[0][0]; y[21] = A[0][1]; y[22] = A[0][2];
      y[23] = A[1][0]; y[24] = A[1][1]; y[25] = A[1][2];
      y[26] = A[2][0]; y[27] = A[2][1]; y[28] = A[2][2];

      Vector<double> By(9);
      multiply_A_v(B, y, By);
      Vector<double> z = chol.solve(By);
      InlineVector<double, 29> Bt_z;
      multiply_At_v(B, z, Bt_z);
      for (int i = 0; i < y.size(); ++i) y[i] -= Bt_z[i];

      unvecSymm3x3d(&y[0], Z1);
      unvecSymm3x3d(&y[10], Z2);
      A[0][0] = y[20]; A[0][1] = y[21]; A[0][2] = y[22];
      A[1][0] = y[23]; A[1][1] = y[24]; A[1][2] = y[25];
      A[2][0] = y[26]; A[2][1] = y[27]; A[2][2] = y[28];
   } // end proxConsistency()

   inline Matrix3x3d
   getRotationFromQuat(Matrix4x4d const& q)
   {
      Matrix3x3d R;
      R[0][0] = q[0][0] + q[1][1] - q[2][2] - q[3][3];
      R[0][1] = 2*(q[1][2] - q[0][3]);
      R[0][2] = 2*(q[1][3] + q[0][2]);
      R[1][0] = 2*(q[1][2] + q[0][3]);
      R[1][1] = q[0][0] - q[1][1] + q[2][2] - q[3][3];
      R[1][2] = 2*(q[2][3] - q[0][1]);
      R[2][0] = 2*(q[1][3] - q[0][2]);
      R[2][1] = 2*(q[2][3] + q[0][1]);
      R[2][2] = q[0][0] - q[1][1] - q[2][2] + q[3][3];
      return R;
   }

} // end namespace <>

namespace V3D
{

   void
   computeConsistentRotations_Linf(double const sigma, int const nIterations, int const nViews,
                                   std::vector<Matrix3x3d> const& relativeRotations,
                                   std::vector<std::pair<int, int> > const& viewPairs,
                                   std::vector<Matrix3x3d>& rotations, std::vector<double>& zs)
   {
      double const gamma = 1.0;

      int const nRelPoses = relativeRotations.size();

      rotations.resize(nViews);

      Matrix3x3d zero3x3d;
      makeZeroMatrix(zero3x3d);

      Matrix4x4d zeroQuat;
      makeZeroMatrix(zeroQuat); zeroQuat[0][0] = 1;

      double const denomT = 1.0 / (1.0 + nRelPoses);

      vector<double> denomQ(nViews, 1.0); // from the psd constraint
      for (int k = 0; k < nRelPoses; ++k)
      {
         int const i = viewPairs[k].first;
         int const j = viewPairs[k].second;
         denomQ[i] += 1;
         denomQ[j] += 1;
      }
      for (int i = 0; i < nViews; ++i) denomQ[i] = 1.0 / denomQ[i];

      double T = 0.0;
      vector<double> T1(nRelPoses);
      vector<double> ZT1(nRelPoses, 0.0);
      double T2;
      double ZT2 = 0;

      vector<Matrix4x4d> Q(nViews, zeroQuat);
      vector<Matrix4x4d> Q1(nViews, zeroQuat);
      vector<Matrix4x4d> ZQ1(nViews, zeroQuat);
      vector<Matrix4x4d> Q2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> Q2j(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2j(nRelPoses, zeroQuat);

      vector<Matrix3x3d> A(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A2(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA2(nRelPoses, zero3x3d);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Convex hull of rotation matrices
         for (int i = 0; i < nViews; ++i)
         {
            Matrix4x4d q = Q[i] + ZQ1[i];
            if (i > 0)
               projectConvHull_SO3(q);
            else
            {
               makeZeroMatrix(q);
               q[0][0] = 1;
            }
            Q1[i] = q;
            addMatricesIP(Q[i] - q, ZQ1[i]);
         } // end for (i)

         // Shrinkage of T (we want to minimize T)
         T2 = std::max(0.0, T + ZT2 - gamma);
         ZT2 += T - T2;

         // Cone constraint
         for (int k = 0; k < nRelPoses; ++k)
         {
            double t = T + ZT1[k];
            Matrix3x3d a = A[k] + ZA1[k];

            proxDataResidual_Frobenius(sigma, t, a);

            T1[k] = t;
            ZT1[k] += T - t;
            A1[k] = a;
            addMatricesIP(A[k] - a, ZA1[k]);
         } // end for (k)

         // Enforce linear consistency
         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            Matrix4x4d qi = Q[i] + ZQ2i[k];
            Matrix4x4d qj = Q[j] + ZQ2j[k];
            Matrix3x3d a = A[k] + ZA2[k];

            proxConsistency(relativeRotations[k], qi, qj, a);

            Q2i[k] = qi;
            Q2j[k] = qj;
            A2[k] = a;
            addMatricesIP(Q[i] - qi, ZQ2i[k]);
            addMatricesIP(Q[j] - qj, ZQ2j[k]);
            addMatricesIP(A[k] - a, ZA2[k]);
         } // end for (k)

         // Averaging of the solutions
         for (int i = 0; i < nViews; ++i) Q[i] = Q1[i] - ZQ1[i];

         T = T2 - ZT2;
         for (int k = 0; k < nRelPoses; ++k) T += T1[k] - ZT1[k];
         T *= denomT;
         T = std::max(0.0, T);

         for (int k = 0; k < nRelPoses; ++k) A[k] = A1[k] - ZA1[k];

         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            addMatricesIP(Q2i[k] - ZQ2i[k], Q[i]);
            addMatricesIP(Q2j[k] - ZQ2j[k], Q[j]);
            addMatricesIP(A2[k] - ZA2[k], A[k]);
         } // end for (k)

         for (int i = 0; i < nViews; ++i) scaleMatrixIP(denomQ[i], Q[i]);
         for (int k = 0; k < nRelPoses; ++k) scaleMatrixIP(0.5, A[k]);

         if ((iter % 500) == 0)
         {
            cout << "iter: " << iter << " t = " << T << endl;
            cout << "T1 = "; displayVector(T1);
            cout << "ZT1 = "; displayVector(ZT1);
            cout << "T2 = " << T2 << " ZT2 = " << ZT2 << endl;

            Matrix<double> ZZ(4, 4);
            for (int i = 0; i < nViews; ++i)
            {
               copyMatrix(Q[i], ZZ);
               SVD<double> svd(ZZ);
               cout << "Q = "; displayMatrix(ZZ);
               cout << "SV = "; displayVector(svd.getSingularValues());
               //Matrix3x3d R = getRotationFromQuat(Q[i]);
               //cout << "R = "; displayMatrix(R);
            } // end for (i)
         }
      } // end for (iter)

      rotations.resize(nViews);
      for (int i = 0; i < nViews; ++i)
         rotations[i] = getRotationFromQuat(Q[i]);

      zs = ZT1;
   } // end computeConsistentRotations_Linf()

   void
   computeConsistentRotations_L1(double const sigma, int const nIterations, int const nViews,
                                 std::vector<Matrix3x3d> const& relativeRotations,
                                 std::vector<std::pair<int, int> > const& viewPairs,
                                 std::vector<Matrix3x3d>& rotations)
   {
      double const gamma = 1.0;

      int const nRelPoses = relativeRotations.size();

      rotations.resize(nViews);

      Matrix3x3d zero3x3d;
      makeZeroMatrix(zero3x3d);

      Matrix4x4d zeroQuat;
      makeZeroMatrix(zeroQuat); zeroQuat[0][0] = 1;

      vector<double> denomQ(nViews, 1.0); // from the psd constraint
      for (int k = 0; k < nRelPoses; ++k)
      {
         int const i = viewPairs[k].first;
         int const j = viewPairs[k].second;
         denomQ[i] += 1;
         denomQ[j] += 1;
      }
      for (int i = 0; i < nViews; ++i) denomQ[i] = 1.0 / denomQ[i];

      vector<double> T(nRelPoses, 0.0);
      vector<double> T1(nRelPoses);
      vector<double> ZT1(nRelPoses, 0.0);
      vector<double> T2(nRelPoses);
      vector<double> ZT2(nRelPoses, 0.0);

      vector<Matrix4x4d> Q(nViews, zeroQuat);
      vector<Matrix4x4d> Q1(nViews, zeroQuat);
      vector<Matrix4x4d> ZQ1(nViews, zeroQuat);
      vector<Matrix4x4d> Q2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> Q2j(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2j(nRelPoses, zeroQuat);

      vector<Matrix3x3d> A(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A2(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA2(nRelPoses, zero3x3d);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Convex hull of rotation matrices
         for (int i = 0; i < nViews; ++i)
         {
            Matrix4x4d q = Q[i] + ZQ1[i];
            projectConvHull_SO3(q);
            Q1[i] = q;
            addMatricesIP(Q[i] - q, ZQ1[i]);
         } // end for (i)

         // Shrinkage of T (we want to minimize T)
         for (int k = 0; k < nRelPoses; ++k)
         {
            T2[k] = std::max(0.0, T[k] + ZT2[k] - gamma);
            ZT2[k] += T[k] - T2[k];
         } // end for (k)

         // Cone constraint
         for (int k = 0; k < nRelPoses; ++k)
         {
            double t = T1[k] + ZT1[k];
            Matrix3x3d a = A[k] + ZA1[k];

            proxDataResidual_Frobenius(sigma, t, a);

            T1[k] = t;
            ZT1[k] += T[k] - t;
            A1[k] = a;
            addMatricesIP(A[k] - a, ZA1[k]);
         } // end for (k)

         // Enforce linear consistency
         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            Matrix4x4d qi = Q[i] + ZQ2i[k];
            Matrix4x4d qj = Q[j] + ZQ2j[k];
            Matrix3x3d a = A[k] + ZA2[k];

            proxConsistency(relativeRotations[k], qi, qj, a);

            Q2i[k] = qi;
            Q2j[k] = qj;
            A2[k] = a;
            addMatricesIP(Q[i] - qi, ZQ2i[k]);
            addMatricesIP(Q[j] - qj, ZQ2j[k]);
            addMatricesIP(A[k] - a, ZA2[k]);
         } // end for (k)

         // Averaging of the solutions
         for (int i = 0; i < nViews; ++i) Q[i] = Q1[i] - ZQ1[i];

         for (int k = 0; k < nRelPoses; ++k)
            T[k] = std::max(0.0, 0.5 * (T1[k] - ZT1[k] + T2[k] - ZT2[k]));

         for (int k = 0; k < nRelPoses; ++k) A[k] = A1[k] - ZA1[k];

         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            addMatricesIP(Q2i[k] - ZQ2i[k], Q[i]);
            addMatricesIP(Q2j[k] - ZQ2j[k], Q[j]);
            addMatricesIP(A2[k] - ZA2[k], A[k]);
         } // end for (k)

         for (int i = 0; i < nViews; ++i) scaleMatrixIP(denomQ[i], Q[i]);
         for (int k = 0; k < nRelPoses; ++k) scaleMatrixIP(0.5, A[k]);

         if ((iter % 500) == 0)
         {
            cout << "iter: " << iter << endl;
            cout << " T = "; displayVector(T);
         }
      } // end for (iter)

      rotations.resize(nViews);
      for (int i = 0; i < nViews; ++i)
         rotations[i] = getRotationFromQuat(Q[i]);
   } // end computeConsistentRotations_L1()

   void
   computeConsistentRotations_LSQ(int const nIterations, int const nViews,
                                  std::vector<Matrix3x3d> const& relativeRotations,
                                  std::vector<std::pair<int, int> > const& viewPairs,
                                  std::vector<Matrix3x3d>& rotations)
   {
      double const gamma = 1.0;

      int const nRelPoses = relativeRotations.size();

      rotations.resize(nViews);

      Matrix3x3d zero3x3d;
      makeZeroMatrix(zero3x3d);

      Matrix4x4d zeroQuat;
      makeZeroMatrix(zeroQuat); zeroQuat[0][0] = 1;

      vector<double> denomQ(nViews, 1.0); // from the psd constraint
      for (int k = 0; k < nRelPoses; ++k)
      {
         int const i = viewPairs[k].first;
         int const j = viewPairs[k].second;
         denomQ[i] += 1;
         denomQ[j] += 1;
      }
      for (int i = 0; i < nViews; ++i) denomQ[i] = 1.0 / denomQ[i];

      vector<Matrix4x4d> Q(nViews, zeroQuat);
      vector<Matrix4x4d> Q1(nViews, zeroQuat);
      vector<Matrix4x4d> ZQ1(nViews, zeroQuat);
      vector<Matrix4x4d> Q2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> Q2j(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2i(nRelPoses, zeroQuat);
      vector<Matrix4x4d> ZQ2j(nRelPoses, zeroQuat);

      vector<Matrix3x3d> A(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> A2(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA1(nRelPoses, zero3x3d);
      vector<Matrix3x3d> ZA2(nRelPoses, zero3x3d);

      for (int iter = 0; iter < nIterations; ++iter)
      {
         // Convex hull of rotation matrices
         for (int i = 0; i < nViews; ++i)
         {
            Matrix4x4d q = Q[i] + ZQ1[i];
            projectConvHull_SO3(q);
            Q1[i] = q;
            addMatricesIP(Q[i] - q, ZQ1[i]);
         } // end for (i)

         // Squared Frobenius term 0.5*sum_k |A_k|_F^2
         for (int k = 0; k < nRelPoses; ++k)
         {
            Matrix3x3d a = A[k] + ZA1[k];
            scaleMatrixIP(1.0 / (1.0 + gamma), a);
            A1[k] = a;
            addMatricesIP(A[k] - a, ZA1[k]);
         } // end for (k)

         // Enforce linear consistency
         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            Matrix4x4d qi = Q[i] + ZQ2i[k];
            Matrix4x4d qj = Q[j] + ZQ2j[k];
            Matrix3x3d a = A[k] + ZA2[k];

            proxConsistency(relativeRotations[k], qi, qj, a);

            Q2i[k] = qi;
            Q2j[k] = qj;
            A2[k] = a;
            addMatricesIP(Q[i] - qi, ZQ2i[k]);
            addMatricesIP(Q[j] - qj, ZQ2j[k]);
            addMatricesIP(A[k] - a, ZA2[k]);
         } // end for (k)

         // Averaging of the solutions
         for (int i = 0; i < nViews; ++i) Q[i] = Q1[i] - ZQ1[i];

         for (int k = 0; k < nRelPoses; ++k) A[k] = A1[k] - ZA1[k];

         for (int k = 0; k < nRelPoses; ++k)
         {
            int const i = viewPairs[k].first;
            int const j = viewPairs[k].second;

            addMatricesIP(Q2i[k] - ZQ2i[k], Q[i]);
            addMatricesIP(Q2j[k] - ZQ2j[k], Q[j]);
            addMatricesIP(A2[k] - ZA2[k], A[k]);
         } // end for (k)

         for (int i = 0; i < nViews; ++i) scaleMatrixIP(denomQ[i], Q[i]);
         for (int k = 0; k < nRelPoses; ++k) scaleMatrixIP(0.5, A[k]);

         if ((iter % 500) == 0)
         {
            double E = 0;
            for (int k = 0; k < nRelPoses; ++k) E += sqrMatrixNormFrobenius(A[k]);
            cout << "iter: " << iter << " E = " << E << endl;
         }
      } // end for (iter)

      rotations.resize(nViews);
      for (int i = 0; i < nViews; ++i)
         rotations[i] = getRotationFromQuat(Q[i]);
   } // end computeConsistentRotations_LSQ()

} // end namespace V3D
