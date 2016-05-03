// -*- C++ -*-
#ifndef V3D_VIDEO_IO_H
#define V3D_VIDEO_IO_H

#if defined(V3DLIB_ENABLE_FFMPEG)

#define USE_SWS_SCALING 0

extern "C"
{
#if !defined (V3DLIB_ENABLE_FFMPEG_VERSION_2009)
# include <ffmpeg/avcodec.h>
# include <ffmpeg/avformat.h>
# include <ffmpeg/avutil.h>
# if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
#  include <ffmpeg/swscale.h>
# endif
#else
# include <libavcodec/avcodec.h>
# include <libavformat/avformat.h>
# include <libavutil/avutil.h>
# if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
#  include <libswscale/swscale.h>
# endif
#endif // !defined(V3DLIB_ENABLE_FFMPEG_VERSION_2009)
}

namespace V3D
{

   struct VideoDecoderFFMPEG
   {
         VideoDecoderFFMPEG(char const * videoFileName);
         ~VideoDecoderFFMPEG();

         bool canDecodeFrames() const { return _canDecode; }

         void skipFrame();
         void seekToFrame(unsigned long timestamp);
         bool decodeRGBFrame(unsigned char * dest);
         bool decodeMonochromeFrame(unsigned char * dest);

         int frameWidth() const
         {
            if (!_pCodecCtx) return -1;
            return _pCodecCtx->width;
         }

         int frameHeight() const
         {
            if (!_pCodecCtx) return -1;
            return _pCodecCtx->height;
         }

      protected:
         bool              _canDecode;
         AVFormatContext * _pFormatCtx;
         AVCodecContext  * _pCodecCtx;
         AVCodec         * _pCodec;
         int               _videoStream;
         AVFrame         * _pFrame; 
         AVFrame         * _pFrameRGB;
         AVFrame         * _pFrameMono;
         uint8_t         * _bufferRGB;
         uint8_t         * _bufferMono;
#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
         SwsContext      * _imgConvertCtxMono;
         SwsContext      * _imgConvertCtxRGB;
#endif
   }; // end struct VideoDecoderFFMPEG

   struct VideoEncoderFFMPEG
   {
         VideoEncoderFFMPEG(char const * videoFileName, int width, int height, long bitrate);
         ~VideoEncoderFFMPEG();

         void encodeRGBFrame(unsigned char const * frame);

         void encodeMonochromeFrame(unsigned char const * frame)
         {
            int const w = _pCodecCtx->width;
            int const h = _pCodecCtx->height;
            int const sz = w*h;

            unsigned char * rgbFrame = new unsigned char[3*sz];
            for (int i = 0; i < sz; ++i)
               rgbFrame[3*i+0] = rgbFrame[3*i+1] = rgbFrame[3*i+2] = frame[i];

            this->encodeRGBFrame(rgbFrame);

            delete [] rgbFrame;
         }

      protected:
         AVFormatContext * _pFormatCtx;
         AVCodecContext  * _pCodecCtx;
         AVCodec         * _pCodec;
         AVFrame         * _pFrameRGB;
         AVFrame         * _pFrameYUV;
         AVStream        * _stream;
         AVOutputFormat  * _format;

         int       _outbuf_size;
         uint8_t * _outbuf;
         uint8_t * _bufferRGB;
         uint8_t * _bufferYUV;

#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
         SwsContext      * _imgConvertCtxMono;
         SwsContext      * _imgConvertCtxRGB;
         SwsContext      * _imgConvertCtxYUV;
#endif
   }; // end struct VideoEncoderFFMPEG

} // end namespace V3D

#endif // defined(V3DLIB_ENABLE_FFMPEG)

#endif
