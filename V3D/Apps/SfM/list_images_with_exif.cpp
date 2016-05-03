#include <iostream>
#include <sstream>

#include "Base/v3d_image.h"
#include "Base/v3d_exifreader.h"
#include "Base/v3d_cfgfile.h"

using namespace std;
using namespace V3D;

int main(int argc, char * argv[])
{
   if (argc != 2)
   {
      cerr << "Usage: " << argv[0] << " <image list file>" << endl;
      return -1;
   }

   std::vector<std::string> entries;

   {
      ifstream is(argv[1]);
      string name;
      while (is >> name)
      {
         entries.push_back(name);
      }
   }
   cout << "Checking calibration data for " << entries.size() << " images." << endl;

   char name[1024];

   int nFoundCalibFiles = 0, nFoundEXIFs = 0;

   for (size_t i = 0; i < entries.size(); ++i)
   {
      bool readEXIF = false;

      int w, h;
      double fx, fy;
      readEXIF = getCalibrationFromEXIF(entries[i].c_str(), w, h, fx, fy);
      if (readEXIF) cout << entries[i] << endl;
   } // end for (i)

   return 0;
}
