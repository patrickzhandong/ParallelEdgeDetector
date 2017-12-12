#include <unistd.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include "preprocess.hpp"
#include "sobel.hpp"
#include "segmentation.hpp"
#include "connected_component.hpp"
#include "binarize.hpp"
#include "location.hpp"
#include "findRegion.hpp"
#include "cudaprocess.h"
#include <time.h>

#define MAX_FILENAME 1024

void parallel_detect(char *input, char *output, int dilation) {
	int width, height, output_width, output_height, type;
	png_bytep *row_pointers = preprocess(input, NULL, &width, &height, &type);
	unsigned char *output_image = cudaMain(width, height, row_pointers, &output_width, &output_height, type, dilation);
	if (dilation) {
		printf("saving dilation image only...\n");
	}
	save_output_image(output_width, output_height, output_image, output);
}

void sequential_detect(char *input, char *output, int dilation) {
	int width, height, output_width, output_height, type, msec;
	clock_t start = clock();

	png_bytep *row_pointers = preprocess(input, NULL, &width, &height, &type);
	unsigned char *preprocess_image = process_file(row_pointers, type);
	clock_t time1 = clock();
	msec = (time1 - start) * 1000 / CLOCKS_PER_SEC;
	printf("preprocess png file takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	unsigned char *original_image1 = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	memcpy(original_image1, preprocess_image, width * height * sizeof(unsigned char));
	unsigned char *original_image2 = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	memcpy(original_image2, preprocess_image, width * height * sizeof(unsigned char));
	clock_t time2 = clock();
	msec = (time2 - time1) * 1000 / CLOCKS_PER_SEC;
	printf("saving original image takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	unsigned char *edge_image = sobel(width, height, preprocess_image);
	clock_t time3 = clock();
	msec = (time3 - time2) * 1000 / CLOCKS_PER_SEC;
	printf("sobel edge detection takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	binarize(width, height, edge_image, 0);
	clock_t time4 = clock();
	msec = (time4 - time3) * 1000 / CLOCKS_PER_SEC;
	printf("binarizing takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	location(width,height,edge_image);
	clock_t time5 = clock();
	msec = (time5 - time4) * 1000 / CLOCKS_PER_SEC;
	printf("dilation takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	if (dilation) {
		printf("saving dilation image only...\n");
		save_output_image(width, height, edge_image, output);
		return;
	}
	unsigned char *output_image = find_region(width, height, edge_image, &output_width, &output_height, original_image1, original_image2);
	clock_t time6 = clock();
	msec = (time6 - time5) * 1000 / CLOCKS_PER_SEC;
	printf("determining the location of plate and extracting number segments takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	msec = (time6 - start) * 1000 / CLOCKS_PER_SEC;
	printf("The whole process takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	if (output_image == NULL) {
		printf("failed to find license plate\n");
		return;
	}
	save_output_image(output_width, output_height, output_image, output);
	free(output_image);
}

int main(int argc, char **argv) {
	char c;
	char input[MAX_FILENAME];
	char output[MAX_FILENAME];
	int parallel = 0;
	int dilation = 0;

	while ((c = getopt(argc, argv, "i:o:p:d:")) != -1) {
		switch (c) {
			case 'i':
			std::strcpy(input, optarg);
			break;

			case 'o':
			std::strcpy(output, optarg);
			break;

			case 'p':
			parallel = 1;
			break;

			case 'd':
			dilation = 1;
			break;
		}
	}

	printf("input filename: %s\n", input);
	printf("output filename: %s\n", output);
	if (parallel) {
		printf("applying parallel solution...\n");
		parallel_detect(input, output, dilation);
	} else {
		printf("applying sequential solution...\n");
		sequential_detect(input, output, dilation);
	}
    return 0;
}
