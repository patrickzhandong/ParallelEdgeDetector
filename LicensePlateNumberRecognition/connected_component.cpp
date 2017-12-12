#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <bits/stdc++.h>
#include <limits.h>
#include <unordered_map>
#include "connected_component.hpp"
#include "cudaprocess.h"

int find_root(std::unordered_map<int, int> union_find, int label) {
	while (union_find.find(label) != union_find.end()) {
		label = union_find.at(label);
	}
	return label;
}

int *find_connected_component(int width, int height, unsigned char *image, int *num_seg) {
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
					left_upper_label = find_root(union_find, labels[left_upper]);
				}
			}

			// top
			if (row != 0) {
				upper = (row - 1) * width + col;
				if (image[upper] == image[curr]) {
					upper_label = find_root(union_find, labels[upper]);
				}
			}

			// top right
			if (row != 0 && col != width - 1) {
				right_upper = (row - 1) * width + col + 1;
				if (image[right_upper] == image[curr]) {
					right_upper_label = find_root(union_find, labels[right_upper]);
				}
			}

			// left
			if (col != 0) {
				left = row * width + col - 1;
				if (image[left] == image[curr]) {
					left_label = find_root(union_find, labels[left]);
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
			curr = labels[row * width + col];
			if (curr == 0) continue;

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

	*num_seg = 0;
	int *segments1;
	for (auto key: segments_coordinates) {
		int *coordinates = key.second;
		int x1 = std::max(0, coordinates[0] - 2);
		int x2 = std::min(width, coordinates[1] + 2);
		int y1 = std::max(0, coordinates[2] - 2);
		int y2 = std::min(height, coordinates[3] + 2);

		int rwidth = x2 - x1;
		int rheight = y2 - y1;
		if (rwidth != 0 && rheight != 0 && rheight / rwidth >= 1 && rheight / rwidth <= 5) {
			if (rwidth * rheight >= width * height / 35) {
				*num_seg  = *num_seg + 1;
				/*
				for (int x = x1; x < x2; x ++) {
					image[y1 * width + x] = 127;
					image[(y2 - 1) * width + x] = 127;
				}

				for (int y = y1; y < y2; y ++) {
					image[y * width + x1] = 127;
					image[y * width + (x2 - 1)] = 127;
				}*/
			}
		}
	}
	
	segments1 = (int*)malloc(sizeof(int) * 4 * (*num_seg));
	int count1 = 0;
	for (auto key:segments_coordinates) {
		int *coordinates = key.second;
		int x1 = std::max(0, coordinates[0] - 2);
		int x2 = std::min(width, coordinates[1] + 2);
		int y1 = std::max(0, coordinates[2] - 2);
		int y2 = std::min(height, coordinates[3] + 2);
	
		int rwidth = x2 - x1;
		int rheight = y2 - y1;
		if (rwidth != 0 && rheight != 0 && rheight / rwidth >= 1 && rheight / rwidth <= 5) {
			if (rwidth * rheight >= width * height / 35) {
				segments1[count1 * 4] = x1;
				segments1[count1 * 4 + 1] = x2;
				segments1[count1 * 4 + 2] = y1;
				segments1[count1 * 4 + 3] = y2;
				count1 ++;
			}
		}
	}
	return segments1;
}
