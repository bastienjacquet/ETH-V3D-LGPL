#include "Base/v3d_videoio.h"
#include "Base/v3d_exception.h"

using namespace std;

#if defined(V3DLIB_ENABLE_FFMPEG)

namespace
{

   void
   initializeFFMPEG()
   {
      static bool initialized = false;
      if (!initialized)
      {
         av_register_all();
         initialized = true;
      }
   }

} // end namespace

namespace V3D
{

   // Reading videos using the ffmpeg library is largely based on Martin Boehme's tutorial.

   VideoDecoderFFMPEG::VideoDecoderFFMPEG(char const * videoFileName)
      : _canDecode(false),
        _pFormatCtx(0), _pCodecCtx(0), _pCodec(0)
#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      , _imgConvertCtxMono(0), _imgConvertCtxRGB(0)
#endif
   {
      initializeFFMPEG();

      // Open video file
      if (av_open_input_file(&_pFormatCtx, videoFileName, NULL, 0, NULL) != 0)
         throw V3D::Exception(string("Cannot open file ") + string(videoFileName));

      // Retrieve stream information
      if (av_find_stream_info(_pFormatCtx) < 0)
      {
         // Couldn't find stream information
         throw V3D::Exception(string("Could not find stream in ") + string(videoFileName));
      }

      // Dump information about file onto standard error
      dump_format(_pFormatCtx, 0, videoFileName, false);

      // Find the first video stream
      _videoStream = -1;
      for (int i = 0; i < _pFormatCtx->nb_streams; ++i)
      {
         if (_pFormatCtx->streams[i]->codec->codec_type==CODEC_TYPE_VIDEO)
         {
            _videoStream = i;
            break;
         }
      } // end for (i)
      if (_videoStream == -1)
      {
         // Didn't find a video stream
         throw V3D::Exception(string("Could not find stream in ") + string(videoFileName));
      }

      // Get a pointer to the codec context for the video stream
      _pCodecCtx = _pFormatCtx->streams[_videoStream]->codec;

      // Find the decoder for the video stream
      _pCodec = avcodec_find_decoder(_pCodecCtx->codec_id);
      if (_pCodec == NULL)
         throw V3D::Exception(string("No codec found in ") + string(videoFileName));

      // Open codec
      if (avcodec_open(_pCodecCtx, _pCodec) < 0)
         throw V3D::Exception(string("Could not open codec in ") + string(videoFileName));

      // Allocate video frame
      _pFrame = avcodec_alloc_frame();

      // Allocate AVFrame structures
      _pFrameRGB = avcodec_alloc_frame();
      if (_pFrameRGB == NULL)
         throw V3D::Exception(string("Could not allocate RGB frame for ") + string(videoFileName));

      _pFrameMono = avcodec_alloc_frame();
      if (_pFrameMono == NULL)
         throw V3D::Exception(string("Could not allocate monochrome frame for ") + string(videoFileName));

      int numBytes;

      // Determine required buffer size and allocate buffer
      numBytes = avpicture_get_size(PIX_FMT_RGB24,
                                    _pCodecCtx->width,
                                    _pCodecCtx->height);
      _bufferRGB = new uint8_t[numBytes];

      // Assign appropriate parts of buffer to image planes in pFrameRGB
      avpicture_fill((AVPicture *)_pFrameRGB, _bufferRGB, PIX_FMT_RGB24,
                     _pCodecCtx->width, _pCodecCtx->height);

      // The same for the monochrome version
      numBytes = avpicture_get_size(PIX_FMT_GRAY8,
                                    _pCodecCtx->width,
                                    _pCodecCtx->height);
      _bufferMono = new uint8_t[numBytes];

      // Assign appropriate parts of buffer to image planes in pFrameMono
      avpicture_fill((AVPicture *)_pFrameMono, _bufferMono, PIX_FMT_GRAY8,
                     _pCodecCtx->width, _pCodecCtx->height);

#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      int const sws_flags = SWS_BICUBIC;

      _imgConvertCtxMono = sws_getContext(_pCodecCtx->width, _pCodecCtx->height,
                                          _pCodecCtx->pix_fmt,
                                          _pCodecCtx->width, _pCodecCtx->height,
                                          PIX_FMT_GRAY8,
                                          sws_flags, NULL, NULL, NULL);

      _imgConvertCtxRGB = sws_getContext(_pCodecCtx->width, _pCodecCtx->height,
                                         _pCodecCtx->pix_fmt,
                                         _pCodecCtx->width, _pCodecCtx->height,
                                         PIX_FMT_RGB24,
                                         sws_flags, NULL, NULL, NULL);
#endif
      _canDecode = true;
   } // end VideoDecoderFFMPEG::VideoDecoderFFMPEG()

   VideoDecoderFFMPEG::~VideoDecoderFFMPEG()
   {
      if (!_canDecode) return;

      delete [] _bufferRGB;
      delete [] _bufferMono;
      av_free(_pFrameRGB);
      av_free(_pFrameMono);

      // Free the YUV frame
      av_free(_pFrame);

      // Close the codec
      avcodec_close(_pCodecCtx);

      // Close the video file
      av_close_input_file(_pFormatCtx);
   }

   void
   VideoDecoderFFMPEG::skipFrame()
   {
      if (!_canDecode) return;

      int frameFinished;
      AVPacket packet;
      bool frameRead = false;

      // Read frames and save first five frames to disk
      while (!frameRead && av_read_frame(_pFormatCtx, &packet) >= 0)
      {
         // Is this a packet from the video stream?
         if (packet.stream_index == _videoStream)
         {
            // Decode video frame
            avcodec_decode_video(_pCodecCtx, _pFrame, &frameFinished, 
                                 packet.data, packet.size);

            // Did we get a video frame?
            if (frameFinished)
               frameRead = true;
         } // end if

         // Free the packet that was allocated by av_read_frame
         av_free_packet(&packet);
      } // end while
   } // end VideoDecoderFFMPEG::skipFrame()

   void
   VideoDecoderFFMPEG::seekToFrame(unsigned long timestamp)
   {
      av_seek_frame(_pFormatCtx, 0, timestamp, 0);
   }

   bool
   VideoDecoderFFMPEG::decodeRGBFrame(unsigned char * dest)
   {
      if (!_canDecode) return false;

      int frameFinished;
      AVPacket packet;
      bool frameRead = false;

      // Read frames and save first five frames to disk
      while (!frameRead && av_read_frame(_pFormatCtx, &packet) >= 0)
      {
         // Is this a packet from the video stream?
         if (packet.stream_index == _videoStream)
         {
            // Decode video frame
            avcodec_decode_video(_pCodecCtx, _pFrame, &frameFinished,
                                 packet.data, packet.size);

            // Did we get a video frame?
            if (frameFinished)
            {
               // Convert the image from its native format to RGB
#if !defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
               img_convert((AVPicture *)_pFrameRGB, PIX_FMT_RGB24,
                           (AVPicture *)_pFrame, _pCodecCtx->pix_fmt, _pCodecCtx->width,
                           _pCodecCtx->height);
#else
               int dstStride[1] = { 3*_pCodecCtx->width };
               uint8_t * dst[1] = { (uint8_t *)dest };
               sws_scale(_imgConvertCtxRGB, _pFrame->data, _pFrame->linesize,
                         0, _pCodecCtx->height, dst, dstStride);
#endif
               frameRead = true;
            } // end if
         } // end if

         // Free the packet that was allocated by av_read_frame
         av_free_packet(&packet);
      } // end while

#if !defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      if (frameRead)
      {
         int const w = _pCodecCtx->width;
         int const h = _pCodecCtx->height;

         int const linesize = _pFrameRGB->linesize[0];
         uint8_t * dataStart = _pFrameRGB->data[0];

         for (int y = 0; y < h; ++y)
            std::copy(dataStart + y*linesize, dataStart + y*linesize + 3*w, dest + 3*y*w);
      } // end if
#endif
      return frameRead;
   } // end VideoDecoderFFMPEG::decodeRGBFrame()

   bool
   VideoDecoderFFMPEG::decodeMonochromeFrame(unsigned char * dest)
   {
      if (!_canDecode) return false;

      int frameFinished;
      AVPacket packet;
      bool frameRead = false;

      // Read frames and save first five frames to disk
      while (!frameRead && av_read_frame(_pFormatCtx, &packet) >= 0)
      {
         // Is this a packet from the video stream?
         if (packet.stream_index == _videoStream)
         {
            // Decode video frame
            avcodec_decode_video(_pCodecCtx, _pFrame, &frameFinished,
                                 packet.data, packet.size);

            // Did we get a video frame?
            if (frameFinished)
            {
               // Convert the image from its native format to monochrome
#if !defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
               img_convert((AVPicture *)_pFrameMono, PIX_FMT_GRAY8,
                           (AVPicture *)_pFrame, _pCodecCtx->pix_fmt, _pCodecCtx->width,
                           _pCodecCtx->height);
#else
               int dstStride[1] = { _pCodecCtx->width };
               uint8_t * dst[1] = { (uint8_t *)dest };
               sws_scale(_imgConvertCtxMono, _pFrame->data, _pFrame->linesize,
                         0, _pCodecCtx->height, dst, dstStride);
#endif
               frameRead = true;
            } // end if
         } // end if

         // Free the packet that was allocated by av_read_frame
         av_free_packet(&packet);
      } // end while

#if !defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      if (frameRead)
      {
         int const w = _pCodecCtx->width;
         int const h = _pCodecCtx->height;

         int const linesize = _pFrameMono->linesize[0];
         uint8_t * dataStart = _pFrameMono->data[0];

         for (int y = 0; y < h; ++y)
            std::copy(dataStart + y*linesize, dataStart + y*linesize + w, dest + y*w);
      } // end if
#endif
      return frameRead;
   } // end VideoDecoderFFMPEG::decodeMonochromeFrame()

//----------------------------------------------------------------------

   VideoEncoderFFMPEG::VideoEncoderFFMPEG(char const * videoFileName, int width, int height, long bitrate)
#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      :_imgConvertCtxMono(0), _imgConvertCtxRGB(0)
#endif
   {
      initializeFFMPEG();

      // auto detect the output format from the name. default is mpeg.
      _format = guess_format(NULL, videoFileName, NULL);
      if (!_format)
         throw V3D::Exception(string("Could not deduce output format from file extension for ") + string(videoFileName));

      _pFormatCtx = av_alloc_format_context();
      _pFormatCtx->oformat = _format;

      strncpy(_pFormatCtx->filename, videoFileName, 1024);

      _stream = av_new_stream(_pFormatCtx, 0);
      if (!_stream)
         throw V3D::Exception(string("Could not allocate stream for ") + string(videoFileName));

      _pCodecCtx = _stream->codec;

      int codec_id = av_guess_codec(_pFormatCtx->oformat, NULL, _pFormatCtx->filename, NULL, CODEC_TYPE_VIDEO);

      _pCodecCtx->codec_id = (CodecID) codec_id;
      _pCodec = avcodec_find_encoder(_pCodecCtx->codec_id);

      _pCodecCtx->codec_type = CODEC_TYPE_VIDEO;

      /* put sample parameters */
      _pCodecCtx->bit_rate = bitrate;

      /* resolution must be a multiple of two */
      _pCodecCtx->width = width;
      _pCodecCtx->height = height;
      _pCodecCtx->time_base = (AVRational){1,25};

      _pCodecCtx->gop_size = 12; /* emit one intra frame every twelve frames at most */
      _pCodecCtx->pix_fmt = PIX_FMT_YUV420P;

      if (_pCodecCtx->codec_id == CODEC_ID_MPEG1VIDEO)
      {
         /* needed to avoid using macroblocks in which some coeffs overflow
            this doesnt happen with normal video, it just happens here as the
            motion of the chroma plane doesnt match the luma plane */
         _pCodecCtx->mb_decision = 2;
      }
      // some formats want stream headers to be seperate
      if (!strcmp(_pFormatCtx->oformat->name, "mp4") ||
          !strcmp(_pFormatCtx->oformat->name, "mov") ||
          !strcmp(_pFormatCtx->oformat->name,  "3gp"))
         _pCodecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;

      if (av_set_parameters(_pFormatCtx, 0) < 0)
         throw V3D::Exception(string("Invalid output format parameters for ") + string(videoFileName));

      dump_format(_pFormatCtx, 0, videoFileName, 1);

      // now that all the parameters are set, we can open the audio and
      // video codecs and allocate the necessary encode buffers
      if (!_stream)
         throw V3D::Exception(string("Couldn't open video stream for ") + string(videoFileName));

      // find the video encoder
      _pCodec = avcodec_find_encoder(_pCodecCtx->codec_id);
      if (!_pCodec)
         throw V3D::Exception(string("codec not found for ") + string(videoFileName));

      // open the codec
      if (avcodec_open(_pCodecCtx, _pCodec) < 0)
         throw V3D::Exception(string("Could not open codec ") + string(videoFileName));

      _outbuf = NULL;

      if (!(_pFormatCtx->oformat->flags & AVFMT_RAWPICTURE))
      {
         /* allocate output buffer */
         /* XXX: API change will be done */
         _outbuf_size = 200000;
         _outbuf = new uint8_t[_outbuf_size];
      }

      // open the output file, if needed 
      if (!(_format->flags & AVFMT_NOFILE))
      {
         if (url_fopen(&_pFormatCtx->pb, videoFileName, URL_WRONLY) < 0)
            throw V3D::Exception(string("Couldn't open output file for writing ") + string(videoFileName));
      }

      // write the stream header, if any
      av_write_header(_pFormatCtx);

      // Allocate AVFrame structures
      _pFrameRGB = avcodec_alloc_frame();
      if (_pFrameRGB == NULL)
         throw V3D::Exception(string("Could not allocate RGB frame for ") + string(videoFileName));
      _pFrameYUV = avcodec_alloc_frame();
      if (_pFrameYUV == NULL)
         throw V3D::Exception(string("Could not allocate YUV frame for ") + string(videoFileName));

      int numBytes;

      // Determine required buffer size and allocate buffer
      numBytes = avpicture_get_size(PIX_FMT_RGB24, width, height);
      _bufferRGB = new uint8_t[numBytes];

      // Assign appropriate parts of buffer to image planes in pFrameRGB
      avpicture_fill((AVPicture *)_pFrameRGB, _bufferRGB, PIX_FMT_RGB24, width, height);

      // The same for the monochrome version
      numBytes = avpicture_get_size(PIX_FMT_YUV420P, width, height);
      _bufferYUV = new uint8_t[numBytes];

      // Assign appropriate parts of buffer to image planes in pFrameRGB
      avpicture_fill((AVPicture *)_pFrameYUV, _bufferYUV, PIX_FMT_YUV420P, width, height);

#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      int const sws_flags = SWS_BICUBIC;

      _imgConvertCtxMono = sws_getContext(_pCodecCtx->width, _pCodecCtx->height,
                                          PIX_FMT_GRAY8,
                                          _pCodecCtx->width, _pCodecCtx->height,
                                          _pCodecCtx->pix_fmt,
                                          sws_flags, NULL, NULL, NULL);

      _imgConvertCtxRGB = sws_getContext(_pCodecCtx->width, _pCodecCtx->height,
                                         PIX_FMT_RGB24,
                                         _pCodecCtx->width, _pCodecCtx->height,
                                         _pCodecCtx->pix_fmt,
                                         sws_flags, NULL, NULL, NULL);

      _imgConvertCtxYUV = sws_getContext(_pCodecCtx->width, _pCodecCtx->height,
                                         PIX_FMT_YUV420P,
                                         _pCodecCtx->width, _pCodecCtx->height,
                                         _pCodecCtx->pix_fmt,
                                         sws_flags, NULL, NULL, NULL);
#endif
   } // end VideoEncoderFFMPEG::VideoEncoderFFMPEG()

   VideoEncoderFFMPEG::~VideoEncoderFFMPEG()
   {
      /* write the delayed frames */
      if (!(_pFormatCtx->oformat->flags & AVFMT_RAWPICTURE))
      {
         for (;;)
         {
            int out_size = avcodec_encode_video(_pCodecCtx, _outbuf, _outbuf_size, NULL);
            if (out_size == 0) break;

            AVPacket pkt;
            av_init_packet(&pkt);
            pkt.pts = av_rescale_q(_pCodecCtx->coded_frame->pts, _pCodecCtx->time_base, _stream->time_base);
            if(_pCodecCtx->coded_frame->key_frame)
               pkt.flags |= PKT_FLAG_KEY;
            pkt.stream_index = _stream->index;
            pkt.data = _outbuf;
            pkt.size = out_size;

            // write the compressed frame in the media file
            av_write_frame(_pFormatCtx, &pkt);
         } // end for
      } // end if

      av_write_trailer(_pFormatCtx);

      avcodec_close(_pCodecCtx);

      // free the streams
      for (int i = 0; i < _pFormatCtx->nb_streams; ++i)
      {
         av_freep(&_pFormatCtx->streams[i]->codec);
         av_freep(&_pFormatCtx->streams[i]);
      }

      if (!(_format->flags & AVFMT_NOFILE))
      {
#if defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
         url_fclose(_pFormatCtx->pb);
#else
         url_fclose(&_pFormatCtx->pb);
#endif
      }

      delete [] _bufferRGB;
      delete [] _bufferYUV;
      av_free(_pFrameRGB);
      av_free(_pFrameYUV);

      if (_outbuf) delete [] _outbuf;

      av_free(_pFormatCtx);
   }

   void
   VideoEncoderFFMPEG::encodeRGBFrame(unsigned char const * frame)
   {
      int const w = _pCodecCtx->width;
      int const h = _pCodecCtx->height;

      int const linesize = _pFrameRGB->linesize[0];
      uint8_t * dataStart = _pFrameRGB->data[0];

      for (int y = 0; y < h; ++y)
         std::copy(frame + 3*w*y, frame + 3*w*(y+1), dataStart + y*linesize);

      // Convert the image from RGB to YUV420P
#if !defined(V3DLIB_ENABLE_FFMPEG_SWS_SCALING)
      img_convert((AVPicture *)_pFrameYUV, PIX_FMT_YUV420P,
                  (AVPicture *)_pFrameRGB, PIX_FMT_RGB24, w, h);
#else
      // Not sure if this works...
      int dstStride[1] = { 3*_pCodecCtx->width };
      int linesizes[1] = { linesize };
      uint8_t * src[1] = { (uint8_t *)dataStart };
# if 0
      sws_scale(_imgConvertCtxRGB, src, linesizes,
                0, _pCodecCtx->height, _pFrameYUV->data, dstStride);
# else
      // This seems to work...
      sws_scale(_imgConvertCtxRGB, _pFrameRGB->data, _pFrameRGB->linesize,
                0, _pCodecCtx->height, _pFrameYUV->data, _pFrameYUV->linesize);
# endif
#endif

      int ret = 0;

      if (_pFormatCtx->oformat->flags & AVFMT_RAWPICTURE)
      {
         // raw video case. The API will change slightly in the near futur for that
         AVPacket pkt;
         av_init_packet(&pkt);

         pkt.flags |= PKT_FLAG_KEY;
         pkt.stream_index = _stream->index;
         pkt.data = (uint8_t *)_pFrameYUV;
         pkt.size = sizeof(AVPicture); // ??? Is this correct?

         ret = av_write_frame(_pFormatCtx, &pkt);
      }
      else
      {
         // encode the image
         int out_size = avcodec_encode_video(_pCodecCtx, _outbuf, _outbuf_size, _pFrameYUV);
         // if zero size, it means the image was buffered
         if (out_size > 0)
         {
            AVPacket pkt;
            av_init_packet(&pkt);
            pkt.pts = av_rescale_q(_pCodecCtx->coded_frame->pts, _pCodecCtx->time_base, _stream->time_base);

            if (_pCodecCtx->coded_frame->key_frame)
               pkt.flags |= PKT_FLAG_KEY;
            pkt.stream_index = _stream->index;
            pkt.data = _outbuf;
            pkt.size = out_size;

            /* write the compressed frame in the media file */
            ret = av_write_frame(_pFormatCtx, &pkt);
         }
      } // end if
      if (ret != 0)
         throw V3D::Exception("Error while writing video frame.");
   } // end VideoEncoderFFMPEG::encodeRGBFrame()

} // end namespace V3D

#endif // defined(V3DLIB_ENABLE_FFMPEG)
