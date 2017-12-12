#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include "binarize.hpp"
#include "cudaprocess.h"

void binarize(int width, int height, unsigned char *image, int reverse) {
	unsigned char curr_pixel;
	unsigned char max_pixel = 0;
	unsigned char min_pixel = 255;

	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			curr_pixel = image[row * width + col];
			if (curr_pixel < min_pixel) {
				min_pixel = curr_pixel;
			}

			if (curr_pixel > max_pixel) {
				max_pixel = curr_pixel;
			}
		}
	}

	unsigned char prev_threshold = 255;
	unsigned char curr_threshold = (min_pixel + max_pixel) / 2;
	unsigned long below_threshold_sum = 0;
	unsigned long above_threshold_sum = 0;
	unsigned long below_threshold_count = 0;
	unsigned long above_threshold_count = 0;
	unsigned long below_threshold_avg;
	unsigned long above_threshold_avg;

	while (curr_threshold != prev_threshold) {
		below_threshold_sum = 0;
		above_threshold_sum = 0;
		below_threshold_count = 0;
		above_threshold_count = 0;

		for (int row = 0; row < height; row ++) {
			for (int col = 0; col < width; col ++) {
				curr_pixel = image[row * width + col];
				if (curr_pixel <= curr_threshold) {
					below_threshold_sum += curr_pixel;
					below_threshold_count += 1;
				} else {
					above_threshold_sum += curr_pixel;
					above_threshold_count += 1;
				}
			}
		}

		prev_threshold = curr_threshold;
		below_threshold_avg = below_threshold_count == 0 ? max_pixel : below_threshold_sum / below_threshold_count;
		above_threshold_avg = above_threshold_count == 0 ? min_pixel : above_threshold_sum / above_threshold_count;
		curr_threshold = (below_threshold_avg + above_threshold_avg) / 2;
	}

	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			curr_pixel = image[row * width + col];
			if (reverse) {
				if (curr_pixel <= curr_threshold - 10) {
					image[row * width + col] = 255;
				} else {
					image[row * width + col] = 0;
				}
			} else {
				if (curr_pixel <= curr_threshold) {
					image[row * width + col] = 0;
				} else {
					image[row * width + col] = 255;
				}
			}
		}
	}
}