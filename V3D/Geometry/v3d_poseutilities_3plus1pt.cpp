#include "Math/v3d_linear.h"
#include "Math/v3d_mathutilities.h"
#include "Geometry/v3d_poseutilities.h"

#define X2_POS_LEX 0
#define Y2_POS_LEX 8
#define Z2_POS_LEX 30
#define W2_POS_LEX 53
#define XY_POS_LEX 1
#define XZ_POS_LEX 2
#define XW_POS_LEX 3
#define YZ_POS_LEX 11
#define YW_POS_LEX 15
#define ZW_POS_LEX 37

namespace
{

   inline void
   generateConstraint3P1P_lex(double p1, double p2, double q1, double q2, double Y3, double * dst)
   {
      double const two = 2.0;

      dst[X2_POS_LEX] = p1*q2*Y3+p2*q1*Y3+p1*q2+p2*q1; // x^2
      dst[Y2_POS_LEX] = -p1*q2*Y3-p2*q1*Y3-p1*q2-p2*q1; // y^2
      dst[Z2_POS_LEX] = -p1*q2*Y3+p2*q1*Y3+p1*q2-p2*q1; // z^2
      dst[W2_POS_LEX] = p1*q2*Y3-p2*q1*Y3-p1*q2+p2*q1; // w^2
      dst[XY_POS_LEX] = two*p2*q2*Y3-two*p1*q1*Y3+two*p2*q2-two*p1*q1; // xy
      dst[XZ_POS_LEX] = two*q2*Y3+two*p2;
      dst[XW_POS_LEX] = two*q1*Y3-two*p1;
      dst[YZ_POS_LEX] = -two*q1*Y3-two*p1;
      dst[YW_POS_LEX] = two*q2*Y3-two*p2;
      dst[ZW_POS_LEX] = -two*p2*q2*Y3-two*p1*q1*Y3+two*p2*q2+two*p1*q1;
   } // end generateConstraintEG()

   inline void
   getRotationMatrix(double x, double y, double z, double w, V3D::Matrix3x3d& m)
   {
      double const len = sqrt(x*x + y*y + z*z + w*w);
      double const s = (len > 0.0) ? (2.0 / len) : 0.0;

      double const xs = x*s;  double const ys = y*s;  double const zs = z*s;
      double const wx = w*xs; double const wy = w*ys; double const wz = w*zs;
      double const xx = x*xs; double const xy = x*ys; double const xz = x*zs;
      double const yy = y*ys; double const yz = y*zs; double const zz = z*zs;
      m[0][0] = 1.0 - (yy+zz); m[0][1] = xy-wz;         m[0][2] = xz+wy;
      m[1][0] = xy+wz;         m[1][1] = 1.0 - (xx+zz); m[1][2] = yz-wx;
      m[2][0] = xz-wy;         m[2][1] = yz+wx;         m[2][2] = 1.0 - (xx+yy);
   } // end getRotationMatrix()

} // end namespace

namespace V3D
{

   template <typename Num>
   bool
   computeRelativePose_3Plus1Point(double const p1[3], double const p2[3],
                                   double const q1[3], double const q2[3],
                                   double Y3,
                                   std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts,
                                   int method)
   {
      typedef Num Field;

      Rs.clear();
      Ts.clear();

      if (method == V3D_3P1P_METHOD_LEX)
      {
         Matrix<double> F(4, 56);
         makeZeroMatrix(F);

         // x^2 + y^2 + z^2 + w^2 - 1 = 0
         F[0][X2_POS_LEX] = 1.0; F[0][Y2_POS_LEX] = 1.0;
         F[0][Z2_POS_LEX] = 1.0; F[0][W2_POS_LEX] = 1.0; F[0][55] = -1.0;
         generateConstraint3P1P_lex(p1[0], p2[0], q1[0], q2[0], Y3, F[1]);
         generateConstraint3P1P_lex(p1[1], p2[1], q1[1], q2[1], Y3, F[2]);
         generateConstraint3P1P_lex(p1[2], p2[2], q1[2], q2[2], Y3, F[3]);

         convertToRowEchelonMatrix(F);
         convertToReducedRowEchelonMatrix(F);

         Num G[34][56];

         for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 56; ++j)
               G[i][j] = F[i][j];

#include "v3d_3plus1pt_lex_generated.h"

         std::vector<Num> rootsW;
         Num coeffsW[9];
         coeffsW[8] = G[33][39];
         coeffsW[7] = G[33][41];
         coeffsW[6] = G[33][43];
         coeffsW[5] = G[33][45];
         coeffsW[4] = G[33][47];
         coeffsW[3] = G[33][49];
         coeffsW[2] = G[33][51];
         coeffsW[1] = G[33][53];
         coeffsW[0] = G[33][55];

         RootFindingParameters<Num> rootParams;
         rootParams.maxBisectionIterations = 300;
         computeRealRootsOfPolynomial(8, coeffsW, rootsW, rootParams);

         for (int i = 0; i < rootsW.size(); ++i)
         {
            Num const w2 = rootsW[i];
            if (w2 < 0) continue;

            Num const w  = sqrt(double(w2));
            Num const w3 = w2*w;
            Num const w4 = w2*w2;
            Num const w5 = w3*w2;
            Num const w7 = w2*w5;
            Num const w9 = w2*w7;
            Num const w11 = w2*w9;
            Num const w13 = w2*w11;
            Num const w15 = w2*w13;

            Num const z = -(G[32][40]*w15 + G[32][42]*w13 + G[32][44]*w11 + G[32][46]*w9 + G[32][48]*w7 +
                            G[32][50]*w5 + G[32][52]*w3 + G[32][54]*w);

            Num const z2 = z*z;
            Num const z3 = z2*z;
            Num const z4 = z2*z2;
            Num const z5 = z3*z2;

            Num const y = -(G[13][17]*z5 + G[13][19]*z4*w + G[13][22]*z3*w2 + G[13][24]*z3 + G[13][27]*z2*w3 +
                            G[13][29]*z2*w + G[13][34]*z*w4 + G[13][36]*z*w2 + G[13][38]*z +
                            G[13][50]*w5 + G[13][52]*w3 + G[13][54]*w);

            Num const y2 = y*y;
            Num const y3 = y2*y;

            Num const x = -(G[4][5]*y3 + G[4][6]*y2*z + G[4][7]*y2*w + G[4][9]*y*z2 +
                            G[4][10]*y*z*w + G[4][14]*y*w2 + G[4][16]*y +
                            G[4][24]*z3 + G[4][29]*z2*w + G[4][36]*z*w2 + G[4][38]*z + G[4][52]*w3 + G[4][54]*w);

            Matrix3x3d R;
            getRotationMatrix(x, y, z, w, R);
            Vector3d X(makeVector3(0.0, 0.0, 1.0));
            Vector3d Y(makeVector3(0.0, 0.0, Y3));
            Vector3d T = Y - R*X;
            Rs.push_back(R);
            Ts.push_back(T);
         } // end for (i)
         return true;
      }
      else
      {
         return false;
      }
   } // end computeRelativePose_3Plus1Point()

   template bool
   computeRelativePose_3Plus1Point<double>(double const p1[3], double const p2[3],
                                           double const q1[3], double const q2[3],
                                           double Y3,
                                           std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts,
                                           int method);

   template bool
   computeRelativePose_3Plus1Point<long double>(double const p1[3], double const p2[3],
                                                double const q1[3], double const q2[3],
                                                double Y3,
                                                std::vector<Matrix3x3d>& Rs, std::vector<Vector3d>& Ts,
                                                int method);

} // end namespace V3D
