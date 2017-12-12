#include <png.h>
#define PNG_DEBUG 3

unsigned char *process_file(png_bytep *row_pointers, int type);
png_bytep *preprocess(char *input, char *output, int *image_width, int *image_height, int *type);
