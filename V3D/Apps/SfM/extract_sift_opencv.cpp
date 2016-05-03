#include <iostream>
#include <sstream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "Base/v3d_image.h"
#include "Base/v3d_feature.h"

using namespace std;
using namespace V3D;
using namespace cv;


int main(int argc, char * argv[])
{
   if (argc != 4 && argc != 5)
   {
      cerr << "Usage: " << argv[0] << " <image list file> <min. octave> <DoG threshold> [<silhouette images list file>]" << endl;
      return -1;
   }

   try
   {
      std::vector<std::string> entries;
      std::vector<std::string> silEntries;

      {
         ifstream is(argv[1]);
         string name;
         while (is >> name)
         {
            entries.push_back(name);
         }
      }

      if (argc == 5)
      {
         ifstream is(argv[4]);
         string name;
         while (is >> name)
         {
            silEntries.push_back(name);
         }
         if (entries.size() != silEntries.size())
         {
            cerr << "The number of images and the number of silhouettes do not match." << endl;
            return -2;
         }
      }

      int firstOctave = atoi(argv[2]);
      float peakThreshold = atof(argv[3]);

      SIFT::DetectorParams detectorParams(peakThreshold, 10.0);
      SIFT::DescriptorParams descriptorParams();


#pragma omp parallel for
      //for each image
      for (size_t i = 0; i < entries.size(); ++i)
      {
         char imgName[1024];
         strncpy(imgName, entries[i].c_str(), 1024);


         cout<<imgName<<endl;
         Mat im = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);

         Size sizeIm;
         sizeIm = im.size();
         int  noctaves = std::max ((int)floor (log2 (std::min(sizeIm.width, sizeIm.height))) - firstOctave - 3, 1) ;
         cout<<noctaves<<endl;
         SIFT::CommonParams commonParams(noctaves, 3, firstOctave, 0);
         SiftFeatureDetector detector(detectorParams,commonParams);
         SiftDescriptorExtractor extractor(descriptorParams, commonParams);

         vector<KeyPoint> keypoints;
         detector.detect(im, keypoints);

         Mat descriptors;
         extractor.compute(im, keypoints, descriptors);

         Size descSize = descriptors.size();
         SerializableVector<SIFT_Feature> extractedFeatures;

         cout<< descSize.height <<endl;
         cout<< "Number of extracted keypoints: "<<keypoints.size() <<endl;

         for(int k = 0; k<descSize.height; ++k)
         {
             SIFT_Feature feature;
             float const x           = keypoints[k].pt.x;
             float const y           = keypoints[k].pt.y;
             float const scale       = keypoints[k].size;
             float const orientation = keypoints[k].angle;

             feature.id        = k;
             feature.position  = makeVector2<float>(x, y);
             feature.scale     = scale;
             feature.direction = makeVector2<float>(cos(orientation), sin(orientation));
             for (int l = 0; l < 128; ++l) feature.descriptor[l] = descriptors.at<float>(k,l); //times 512??

             feature.normalizeDescriptor();

             //save in the vector of features
             extractedFeatures.push_back(feature);
         }

         //save all the features to a file
         ostringstream oss;
         oss << entries[i] << ".features";
         cout << "writing " << extractedFeatures.size() << " to " << oss.str() << endl;
         serializeDataToFile(oss.str().c_str(), extractedFeatures, true);


      } //end for each image

   }
   catch (std::string s)
   {
      cerr << "Exception caught: " << s << endl;
   }
   catch (...)
   {
      cerr << "Unhandled exception." << endl;
   }

   return 0;
}
