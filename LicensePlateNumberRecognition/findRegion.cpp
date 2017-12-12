#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <bits/stdc++.h>
#include <limits.h>
#include <unordered_map>
#include "findRegion.hpp"
#include "connected_component.hpp"
#include "binarize.hpp"

int find_root2(std::unordered_map<int, int> union_find, int label) {
	while (union_find.find(label) != union_find.end()) {
		label = union_find.at(label);
	}
	return label;
}

void draw(int *segments, int num_seg, int width, int height, unsigned char *input_image) {

	for (int i = 0; i < num_seg; i ++) {
		int x1 = segments[4 * i];
		int x2 = segments[4 * i + 1];
		int y1 = segments[4 * i + 2];
		int y2 = segments[4 * i + 3];

		for (int x = x1; x < x2; x ++) {
			input_image[y1 * width + x] = 127;
			input_image[(y2 - 1) * width + x] = 127;
		}

		for (int y = y1; y < y2; y ++) {
			input_image[y * width + x1] = 127;
			input_image[y * width + (x2 - 1)] = 127;
		}
	}
}

unsigned char *extractRegion(int x1, int x2, int y1, int y2, int width, int height, unsigned char *original_image) {

	int cwidth = x2 - x1;
	int cheight = y2 - y1;
	unsigned char *output_image = (unsigned char*)malloc(sizeof(unsigned char) * cwidth * cheight);
    for (int y = y1; y < y2; y ++) {
		for (int x = x1; x < x2; x ++) {
			output_image[(y - y1) * cwidth + (x - x1)] = original_image[y * width + x];
		}
	}
	return output_image;

}

unsigned char *find_region(int width, int height, unsigned char *image, int *output_width, int *output_height,
	             unsigned char *originalImage, unsigned char *original2Image) {
	int *labels = (int*)calloc(sizeof(int), width * height);
	int count = 1;
	int curr, left_upper, upper, right_upper, left;
	int left_upper_label, upper_label, right_upper_label, left_label, min_label;
	std::unordered_map<int, int> union_find;

	// first pass, assign initial labels
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {

			curr = row * width + col;
			if (image[curr] == 0) continue;

			left_upper_label = INT_MAX;
			upper_label = INT_MAX;
			right_upper_label = INT_MAX;
			left_label = INT_MAX;

			// top left
			if (row != 0 && col != 0) {
				left_upper = (row - 1) * width + col - 1;
				if (image[left_upper] == image[curr]) {
					left_upper_label = find_root2(union_find, labels[left_upper]);
				}
			}

			// top
			if (row != 0) {
				upper = (row - 1) * width + col;
				if (image[upper] == image[curr]) {
					upper_label = find_root2(union_find, labels[upper]);
				}
			}

			// top right
			if (row != 0 && col != width - 1) {
				right_upper = (row - 1) * width + col + 1;
				if (image[right_upper] == image[curr]) {
					right_upper_label = find_root2(union_find, labels[right_upper]);
				}
			}

			// left
			if (col != 0) {
				left = row * width + col - 1;
				if (image[left] == image[curr]) {
					left_label = find_root2(union_find, labels[left]);
				}
			}

			min_label = std::min(std::min(std::min(left_upper_label, upper_label), right_upper_label), left_label);

			if (min_label == INT_MAX) {
				labels[curr] = count;
				count ++;
			} else {
				
				labels[curr] = min_label;
				if (left_upper_label != INT_MAX && left_upper_label > min_label) {
					union_find[left_upper_label] = min_label;
				}

				if (upper_label != INT_MAX && upper_label > min_label) {
					union_find[upper_label] = min_label;
				}

				if (right_upper_label != INT_MAX && right_upper_label > min_label) {
					union_find[right_upper_label] = min_label;
				}

				if (left_label != INT_MAX && left_label > min_label) {
					union_find[left_label] = min_label;
				}
			}

		}
	}

	std::unordered_map<int, int*> segments_coordinates;

	// second pass
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			if (!image[row * width + col]) continue;

			curr = labels[row * width + col];
			while (union_find.find(curr) != union_find.end()) {
				curr = union_find.at(curr);
			}
			labels[row * width + col] = curr;

			if (segments_coordinates.find(curr) == segments_coordinates.end()) {
				int *coordinates = (int*)malloc(sizeof(int) * 4);
				coordinates[0] = col;
				coordinates[1] = col;
				coordinates[2] = row;
				coordinates[3] = row;
				segments_coordinates[curr] = coordinates;
			} else {
				int *coordinates = segments_coordinates.at(curr);
				coordinates[0] = std::min(coordinates[0], col);
				coordinates[1] = std::max(coordinates[1], col);
				coordinates[2] = std::min(coordinates[2], row);
				coordinates[3] = std::max(coordinates[3], row);
			}
		}
	}

	int suitable = 0;
	int max_num_seg = -1;
	int best_x1, best_x2, best_y1, best_y2;
	int num_seg1, num_seg2;
	for (auto key: segments_coordinates) {
		int *coordinates = key.second;
		int x1 = coordinates[0];//std::max(0, coordinates[0] - 2);
		int x2 = coordinates[1];//std::min(width, coordinates[1] + 2);
		int y1 = coordinates[2];//std::max(0, coordinates[2] - 2);
		int y2 = coordinates[3];//std::min(height, coordinates[3] + 2);
		
		int cwidth = x2 - x1;
		int cheight = y2 - y1;
		if (cwidth < 10 * cheight && 2 * cwidth > 3 * cheight &&
			cwidth * cheight * 50> width * height &&
			cwidth * cheight * 10 < width * height) {
			suitable ++;

		    printf("testing suitable region %d: x1: %d, x2: %d, y1: %d, y2: %d ...\n", suitable, x1, x2, y1, y2);
		    unsigned char *output_image1 = extractRegion(x1, x2, y1, y2, width, height, originalImage);
		    unsigned char *output_image2 = extractRegion(x1, x2, y1, y2, width, height, original2Image);
		    binarize(cwidth, cheight, output_image1, 0);
		    binarize(cwidth, cheight, output_image2, 1);
		    num_seg1 = 0;
		    num_seg2 = 0;
		    int *best_segments1 = find_connected_component(cwidth, cheight, output_image1, &num_seg1);
		    int *best_segments2 = find_connected_component(cwidth, cheight, output_image2, &num_seg2);
		    int curr_num_seg = std::max(num_seg1, num_seg2);
		    printf("suitable region %d has maximum %d segments\n", suitable, curr_num_seg);
		    if (max_num_seg == -1 || curr_num_seg > max_num_seg) {
		    	best_x1 = x1;
		    	best_x2 = x2;
		    	best_y1 = y1;
		    	best_y2 = y2;
		    	max_num_seg = curr_num_seg;
		    }
		    free(output_image1);
		    free(output_image2);
		    free(best_segments2);
		    free(best_segments1);
		    
		}
		free(coordinates);
	}

	printf("total suitable region: %d, outputting the best one...\n", suitable);

	if (suitable == 0) {
		return NULL;
	}

	*output_width = best_x2 - best_x1;
	*output_height = best_y2 - best_y1;
	unsigned char *output_image1 = extractRegion(best_x1, best_x2, best_y1, best_y2, width, height, originalImage);
	unsigned char *output_image2 = extractRegion(best_x1, best_x2, best_y1, best_y2, width, height, original2Image);
	binarize(*output_width, *output_height, output_image1, 1);
	binarize(*output_width, *output_height, output_image2, 0);
	num_seg1 = 0;
	num_seg2 = 0;
	int *best_segments1 = find_connected_component(*output_width, *output_height, output_image1, &num_seg1);
	int *best_segments2 = find_connected_component(*output_width, *output_height, output_image2, &num_seg2);
	if (num_seg1 > num_seg2) {
		draw(best_segments1, num_seg1, *output_width, *output_height, output_image1);
		printf("number of number segments is %d\n", num_seg1);
		free(output_image2);
		free(best_segments1);
		free(best_segments2);
		return output_image1;
	}
	draw(best_segments2, num_seg2, *output_width, *output_height, output_image2);
	printf("number of number segments is %d\n", num_seg2);
	free(output_image1);
	free(best_segments2);
	free(best_segments1);
	return output_image2;
}

