#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>


void dilation(int xradius, int yradius,int type, int width, int height, unsigned char *input_image);
void erosion(int radius,int width, int height, unsigned char *input_image);
void location(int width, int height, unsigned char *input_image);