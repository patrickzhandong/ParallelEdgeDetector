#ifndef __CUDA_PROCESS_H__
#define __CUDA_PROCESS_H__
#include <png.h>

unsigned char *cudaMain(int width, int height, png_bytep *row_pointers, int *output_width, int *output_height, int type, int dilation);


#endif