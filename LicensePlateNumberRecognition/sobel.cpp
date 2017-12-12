#include <unistd.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <float.h>
#include <math.h>
#include "cudaprocess.h"
#include "sobel.hpp"

double mask[9] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0}; // 3x3 mask

double convolution(unsigned char *input_image, int row, int col, int width, int height) {
	// do convolution in both x-direction and y-direction
	double horizontal_total = 0.0;
	double vertical_total = 0.0;

	int mask_row, mask_col;
	for (int i = row - 1; i < row + 2; i ++) {
		for (int j = col - 1; j < col + 2; j ++) {
			if (j >= 0 && j < width && i >= 0 && i < height) {
				mask_row = i - (row - 1);
				mask_col = j - (col - 1);
				horizontal_total += input_image[i * width + j] * mask[mask_row * 3 + mask_col];
				vertical_total += input_image[i * width + j] * mask[mask_col * 3 + mask_row];
			}
		}
	}

	return sqrt(horizontal_total * horizontal_total + vertical_total * vertical_total);
}

unsigned char *sobel(int width, int height, unsigned char *input_image) {
	double conv_result;
	double min_val = DBL_MAX;
	double max_val = DBL_MIN;
	
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			//printf("got %d\n", row * width + col);
			conv_result = convolution(input_image, row, col, width, height);
			if (conv_result < min_val) min_val = conv_result;
			if (conv_result > max_val) max_val = conv_result;
		}
	}
	//printf("to here\n");
	unsigned char *output_image = (unsigned char*)malloc(sizeof(unsigned char) * (width * height));
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			conv_result = convolution(input_image, row, col, width, height);
			output_image[row * width + col] = MAX_BRIGHTNESS * (conv_result - min_val) / (max_val - min_val);
			//printf("%d\n", output_image[row * width + col]);

		}
	}
	return output_image;
}

/*
// parsing cited from http://cis.k.hosei.ac.jp/~wakahara/myppm.h
int process_initial_image(char *input) {
	char buffer[MAX_BUF_SIZE];
	width = -1;
	height = -1;
	int max_gray = -1;

	FILE *fp = fopen(input, "rb");

	fgets(buffer, MAX_BUF_SIZE, fp);

	if (buffer[0] != 'P' || buffer[1] != '5') {
		printf("Incorrect file format\n");
		return -1;
	}

	while (width == -1 || height == -1) {
		fgets(buffer, MAX_BUF_SIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d %d", &width, &height);
		}
	}

	while (max_gray == -1) {
		fgets(buffer, MAX_BUF_SIZE, fp);
		if (buffer[0] != '#') {
			sscanf(buffer, "%d", &max_gray);
		}
	}

	if (max_gray != MAX_BRIGHTNESS) {
		printf("Invalid maximum gray level\n");
		return -1;
	}

	input_image = (unsigned char*)malloc(sizeof(unsigned char) * (width * height));
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			input_image[row * width + col] = (unsigned char)fgetc(fp);
		}
	}

	printf("Successfully loaded input image!\n");
	fclose(fp);
	return 0;
}*/

int save_output_image(int width, int height, unsigned char *output_image, char *output) {
	FILE *fp = fopen(output, "wb");
	fputs("P5\n", fp);
	fprintf(fp, "%d %d\n", width, height);
	fprintf(fp, "%d\n", MAX_BRIGHTNESS);

	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			fputc(output_image[row * width + col], fp);
		}
	}
	printf("Successfully write to output image!\n");
	fclose(fp);
	return 0;
}

