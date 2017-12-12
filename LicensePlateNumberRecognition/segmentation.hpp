#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#define TOLERANCE 5
#define THRESHOLD 20

void segmentation(int width, int height, unsigned char *input_image, int *x_segs, int *y_segs, int num_seg);