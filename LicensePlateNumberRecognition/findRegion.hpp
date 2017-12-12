#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

unsigned char *find_region(int width, int height, unsigned char *image, int *output_width, int *output_height,
	                       unsigned char *originalImage, unsigned char *original2Image);