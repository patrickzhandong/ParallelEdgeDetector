#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "segmentation.hpp"

bool is_background_color(unsigned char background_color, unsigned char color) {
	if (color > background_color) {
		return color - background_color <= TOLERANCE;
	}
	return background_color - color <= TOLERANCE;
}

void segmentation(int width, int height, unsigned char *input_image, int *x_segs, int *y_segs, int num_seg) {

	unsigned char background_color = input_image[0]; // assume corner has background color
	char in_segment = 0;
	int seg_count = 0;
	unsigned char pixel1, pixel2;

	for (int col = 0; col < width - 1; col ++) {
		char has_non_background_color = 0;
		for (int row = 0; row < height; row ++) {
			pixel1 = input_image[row * width + col];
			pixel2 = input_image[row * width + col + 1];

			if (!is_background_color(background_color, pixel1) || !is_background_color(background_color, pixel2)) {
				has_non_background_color = 1;
			}

			if (pixel2 > pixel1) {
				if (pixel2 - pixel1 >= THRESHOLD) {
					if (!in_segment) {
						in_segment = 1;
						x_segs[seg_count * 2] = col;
					}
				}
			} else {
				if (pixel1 - pixel2 >= THRESHOLD) {
					if (!in_segment) {
						in_segment = 1;
						x_segs[seg_count * 2] = col;
					}
				}
			}
		}

		if (!has_non_background_color) {
			if (in_segment) {
				in_segment = 0;
				x_segs[seg_count * 2 + 1] = col;
				seg_count += 1;
			}
		}
	}

	in_segment = 0;
	for (int i = 0; i < seg_count; i ++) {
		int left_bound = x_segs[2 * i];
		int right_bound = x_segs[2 * i + 1];
		in_segment = 0;
		for (int row = 0; row < height - 1; row ++) {
			char has_non_background_color = 0;
			for (int col = left_bound; col <= right_bound; col ++) {
				pixel1 = input_image[row * width + col];
				pixel2 = input_image[(row + 1) * width + col];

				if (!is_background_color(background_color, pixel1) || !is_background_color(background_color, pixel2)) {
					has_non_background_color = 1;
				}

				if (pixel2 > pixel1) {
					if (pixel2 - pixel1 >= THRESHOLD) {
						if (!in_segment) {
							in_segment = 1;
							y_segs[i * 2] = row;
						}
					}
				} else {
					if (pixel1 - pixel2 >= THRESHOLD) {
						if (!in_segment) {
							in_segment = 1;
							y_segs[i * 2] = row;
						}
					}
				}

			}

			if (!has_non_background_color) {
				if (in_segment) {
					y_segs[i * 2 + 1] = row;
					break;
				}
			}
		}
	}

	printf("Expect to find %d segments, actually found %d segments\n", num_seg, seg_count);
	for (int i = 0; i < seg_count; i ++) {
		int x_lo = std::max(x_segs[2 * i] - 5, 0);
		int x_hi = std::min(x_segs[2 * i + 1] + 5, width - 1);
		int y_lo = std::max(y_segs[2 * i] - 5, 0);
		int y_hi = std::min(y_segs[2 * i + 1] + 5, height - 1);

		printf("%d th segment bound is: x_lo: %d, x_hi: %d, y_lo: %d, y_hi: %d\n", i, x_lo, x_hi, y_lo, y_hi);
		for (int x = x_lo; x <= x_hi; x ++) { // first row, last row
			input_image[y_lo * width + x] = 255 - background_color;
			input_image[y_hi * width + x] = 255 - background_color;
		}

		for (int y = y_lo; y <= y_hi; y ++) { // first col, last col
			input_image[y * width + x_lo] = 255 - background_color;
			input_image[y * width + x_hi] = 255 - background_color;
		}
	}
}