#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_BRIGHTNESS 255

unsigned char *sobel(int width, int height, unsigned char *input_image);
int save_output_image(int width, int height, unsigned char *output_image, char *output);