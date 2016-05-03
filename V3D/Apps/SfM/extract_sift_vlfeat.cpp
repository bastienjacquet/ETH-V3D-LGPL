extern "C" {
  #include <vl/generic.h>
#include <vl/sift.h>
}

#include <iostream>
#include <sstream>

#include "Base/v3d_image.h"
#include "Base/v3d_feature.h"

using namespace std;
using namespace V3D;

/*int main (int argc, const char * argv[]) {
  VL_PRINT ("Hello world!") ;
    cout<<"whazzuuup";
  return 0;
}*/

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

      cout << "Extracting feature, multithreaded if possible." << endl;
      cout << "firstOctave="<<firstOctave << " peakThreshold="<<peakThreshold << endl;
      //for each image
#pragma omp parallel for
      for (size_t i = 0; i < entries.size(); ++i)
      {
         char imgName[1024];
         strncpy(imgName, entries[i].c_str(), 1024);

         //cout<<imgName<<endl;

         Image<unsigned char>  im;
         loadImageFile(imgName,im);

         int h=im.height();
         int w=im.width();

         VlSiftFilt *siftFilter = vl_sift_new(w,h,-1,3,firstOctave);
         vl_sift_set_peak_thresh(siftFilter,peakThreshold);

         float* imFloat=new float[h*w];

         if (im.numChannels() == 3)
         {
            for(int y=0;y<h;++y) //image conversion to float vector
               for(int x=0;x<w;++x)
               {
                  imFloat[y*w+x]=0.2989*im(x,y,0)+0.587*im(x,y,1)+0.114*im(x,y,2);
               }
         }
         else
         {
            for(int y=0;y<h;++y) //image conversion to float vector
               for(int x=0;x<w;++x)
                  imFloat[y*w+x] = im(x,y);
         }

         SerializableVector<SIFT_Feature> extractedFeatures;

         int count=0; //counts the total number of features of the image

         int err=0;
         bool first=true;


         while(true)
         {
             VlSiftKeypoint const *keys;
             int nkeys;

             if(first) //first octave
             {
                 first=false;
                 err=vl_sift_process_first_octave(siftFilter,imFloat);
             }
             else //following octaves
                 err=vl_sift_process_next_octave(siftFilter);

             if(err==VL_ERR_EOF) //until no more octaves are left
             {
                 err=VL_ERR_OK;
                 break;
             }

             //run detector
             vl_sift_detect(siftFilter);

             keys=vl_sift_get_keypoints(siftFilter);
             nkeys=vl_sift_get_nkeypoints(siftFilter);

             //for each keypoint
             for(int k=0;k<nkeys;k++)
             {
                 double angles[4];
                 int nangles;

                 //obtain orientations
                 nangles=vl_sift_calc_keypoint_orientations(siftFilter,angles,&keys[k]);
                 //for each orientation
                 for(int j=0;j<nangles;j++)
                 {
                     //calculate descriptor
                     float descriptor[128];
                     vl_sift_calc_keypoint_descriptor(siftFilter,descriptor,&keys[k],angles[j]);
                     //create feature
                     SIFT_Feature feature;

                     float const x           = keys[k].x;
                     float const y           = keys[k].y;
                     float const scale       = keys[k].sigma;
                     float const orientation = angles[j];

                     feature.id        = count;
                     feature.position  = makeVector2<float>(x, y);
                     feature.scale     = scale;
                     feature.direction = makeVector2<float>(cos(orientation), sin(orientation));
                     for (int l = 0; l < 128; ++l) feature.descriptor[l] = descriptor[l]; //times 512??

                     count++;
                     feature.normalizeDescriptor();

                     //save in the vector of features
                     extractedFeatures.push_back(feature);
                 }//end for each orientation

             }//end for each keypoint
         }//end while

         if (!silEntries.empty())
         {
            Image<unsigned char> silImage;
            loadImageFile(silEntries[i].c_str(), silImage);

            SerializableVector<SIFT_Feature> filteredFeatures;
            for (int k = 0; k < extractedFeatures.size(); ++k)
            {
               SIFT_Feature const& feature = extractedFeatures[k];
               int const X = int(feature.position[0] + 0.5f);
               int const Y = int(feature.position[1] + 0.5f);

               if (silImage(X, Y) > 128)
                  filteredFeatures.push_back(feature);
            }
            extractedFeatures = filteredFeatures;
         } // end if

         //save all the features to a file
         ostringstream oss;
         oss << entries[i] << ".features";
         cout << "writing " << extractedFeatures.size() << " features from "<<entries[i]<<" ["<< w<< "x"<< h<<"] to " << oss.str() << endl;
         serializeDataToFile(oss.str().c_str(), extractedFeatures, true);

         //free memory
         vl_sift_delete(siftFilter);

         delete[] imFloat;
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
