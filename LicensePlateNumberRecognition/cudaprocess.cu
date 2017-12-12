#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <png.h>
#include <limits.h>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cudaprocess.h"
#include "preprocess.hpp"

// struct GlobalConstants {
// 	int width;
// 	int height;
// 	unsigned char *cimage;
// 	png_bytep *rows;
// };

//  GlobalConstants cuConstParams;
#define MAX_BRIGHTNESS 255
#define BLOCK_HEIGHT 32
#define BLOCK_WIDTH 32

unsigned char *original2Image;
unsigned char *originalImage;
unsigned char *tempImage;

__constant__ int mask[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

__global__ void kernelDilation(int width, int height, int xr, int yr,unsigned char *ori_image, unsigned char *aft_image) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int r = index / width; 
	int c = index % width;
	aft_image[r * width + c] = 0;
	if (index >= width * height) {
		return;
	}
	for (int i = r - xr; i <= r + xr; i++) {
		for (int j = c - yr; j <= c + yr; j++) {
			if (i >= 0 && i < height &&
				j >= 0 && j < width &&
				ori_image[i * width + j] == 255) {
				aft_image[r * width + c] = 255;
				break;
			}
		}
	} 
}

__global__ void kernelPreprocess(int width, int height,png_byte *row_pointers, unsigned char *image, int type) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int r = index / width;
	int c = index % width;
	if (index >= width * height) {
		return;
	}
	png_byte* ptr;
	if (type == 0) {
		ptr = &(row_pointers[r * width * 3  + c * 3]);
	}
	else {
		ptr = &(row_pointers[r * width * 4  + c * 4]);
	}
	image[index] = 0.114 * ptr[0] + 0.587 * ptr[1] + 0.299 * ptr[2];
}

__global__ void kernelMin(int width, int height, unsigned char *image, unsigned char *subset,int bsize) {
	//printf("hello there\n");
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int start = bsize * index;
	int end = bsize * (index + 1) > width * height ? width * height : bsize * (index + 1);
	unsigned char min = 255;
	for (int i = start; i < end; i++) {
		if (image[i] < min) {
			min = image[i];
		}
	}
	subset[index] = min;
}
__global__ void kernelMax(int width, int height, unsigned char *image, unsigned char *subset,int bsize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int start = bsize * index;
	int end = bsize * (index + 1) > width * height ? width * height : bsize * (index + 1);
	unsigned char max = 0;
	for (int i = start; i < end; i++) {
		if (image[i] > max) {
			max = image[i];
		}
	}
	subset[index] = max;
}

__global__ void kernelBelow(int width, int height, int threshold,unsigned char *image, int *subset, int *subset_count,int bsize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int start = bsize * index;
	int end = bsize * (index + 1) > width * height ? width * height : bsize * (index + 1);	
	int sum = 0;
	int count = 0;
	for (int i = start; i < end; i++) {
		if (image[i] <= threshold) {
			sum += image[i];
			count++;
		}
	}
	subset[index] = sum;
	subset_count[index] = count;
}

__global__ void kernelAbove(int width, int height, unsigned char threshold,unsigned char *image, int *subset, int *subset_count,int bsize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int start = bsize * index;
	int end = bsize * (index + 1) > width * height ? width * height : bsize * (index + 1);	
	int sum = 0;
	int count = 0;
	for (int i = start; i < end; i++) {
		if (image[i] > threshold) {
			sum += image[i];
			count++;
		}
	}
	subset[index] = sum;
	subset_count[index] = count;
}

__global__ void kernelBinarize(int width, int height, unsigned char threshold, unsigned char *image) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= width * height) {
		return;
	}
	if (image[index] <= threshold) {
		image[index] = 0; 
	}	
	else {
		image[index] = 255;
	}
}

__global__ void kernelReverseBinarize(int width, int height, unsigned char threshold, unsigned char *image) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= width * height) {
		return;
	}
	if (image[index] <= threshold - 10) {
		image[index] = 255; 
	}	
	else {
		image[index] = 0;
	}
}

void
cudaDilation(int w, int h, int xr, int yr){
	unsigned char *a_image;
	cudaMallocManaged(&a_image, w * h * sizeof(unsigned char));
	// for (int i = 0; i < w * h; i++) {
	// 	tempImage[i] = image[i];
	// }
	int isize = w * h;
	dim3 blockDim(256,1);
	dim3 gridDim((isize + blockDim.x - 1) / blockDim.x);
	kernelDilation<<<gridDim,blockDim>>>(w,h,xr,yr,tempImage,a_image);
	cudaDeviceSynchronize();
	for (int i = 0; i < w * h; i++) {
		tempImage[i] = a_image[i];
	}
	cudaFree(a_image);
	//return tempImage;
}

void
cudaPreprocess(int w, int h, png_bytep * row_pointers,int type) {
	//printf("in cudaPreProcess\n");
	//printf("type : %d\n", type);
	int isize = w * h;
	png_byte *rows;
	cudaMallocManaged(&rows, isize * 3 * sizeof(png_byte));
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < 3 * w; j++) {
			rows [3 * w * i + j] = row_pointers[i][j];

		}
	}
	dim3 blockDim(256,1);
	dim3 gridDim((w * h + blockDim.x - 1) / blockDim.x);
	kernelPreprocess<<<gridDim, blockDim>>>(w,h,rows,tempImage,type);
	cudaDeviceSynchronize();
	cudaFree(rows);

}

void cudaBinarize(int w, int h, unsigned char *input_image, int reverse) {
	// unsigned char *a_image;
	// cudaMallocManaged(&a_image,w * h * sizeof(unsigned char));
	// for (int i = 0; i < w * h; i++) {
	// 	a_image[i] = image[i];
	// }
	int numBlocks = 256;
	int blockSize = w * h / 256 + 1;

	unsigned char max_pixel = 0;
	unsigned char min_pixel = 255;	
	unsigned char*subset2;
	cudaMallocManaged(&subset2,numBlocks * sizeof(unsigned char));
	int *subset;
	cudaMallocManaged(&subset,numBlocks * sizeof(int));
	int *subset_count;

	cudaMallocManaged(&subset_count,numBlocks * sizeof(int));
	dim3 blockDim(numBlocks,1);
	dim3 gridDim((numBlocks + blockDim.x - 1) / blockDim.x);

	kernelMin<<<gridDim,blockDim>>>(w,h,input_image,subset2,blockSize);
	cudaDeviceSynchronize();
	for (int i = 0; i < numBlocks; i++) {
		if (subset2[i] < min_pixel) {
			min_pixel = subset2[i];
		}
	}

	kernelMax<<<gridDim,blockDim>>>(w,h,input_image,subset2,blockSize);
	cudaDeviceSynchronize();
	for (int i = 0; i < numBlocks; i++) {
		if (subset2[i] > max_pixel) {
			max_pixel = subset2[i];
		}
	}
	// 	for (int i = 0; i < numBlocks;i ++) {
	// 	printf("%d\n",subset[i]);
	// }
	//printf("to here b\n");
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
		kernelBelow<<<gridDim,blockDim>>>(w,h,curr_threshold,input_image,subset,subset_count,blockSize);
		cudaDeviceSynchronize();
		for (int i = 0; i < numBlocks; i++) {
			below_threshold_sum += subset[i];
			below_threshold_count += subset_count[i];
		}

		//printf("%d %d\n",below_threshold_sum,below_threshold_count);
		//for (int i = 0; i < numBlocks; i++) {
		//	printf("%d ",subset[numBlocks]);
		//}
		//printf("\n");
		//for (int i = 0; i < numBlocks; i++) {
		//	printf("%d ",subset_count[numBlocks]);
		//}
		//printf("\n");
		//printf("\n");
		// printf("%d %d\n",below_threshold_sum,below_threshold_count);
		// for (int i = 0; i < numBlocks; i++) {
		// 	printf("%d ",subset[numBlocks]);
		// }
		// printf("\n");
		// for (int i = 0; i < numBlocks; i++) {
		// 	printf("%d ",subset_count[numBlocks]);
		// }
		// printf("\n");
		// printf("\n");

		kernelAbove<<<gridDim,blockDim>>>(w,h,curr_threshold,input_image,subset,subset_count,blockSize);
		cudaDeviceSynchronize();
		for (int i = 0; i < numBlocks; i++) {
			above_threshold_sum += subset[i];
			above_threshold_count += subset_count[i];
		}

		//printf("%d %d\n",above_threshold_sum,above_threshold_count);
		//printf("\n");
		prev_threshold = curr_threshold;
		below_threshold_avg = below_threshold_count == 0 ? max_pixel : below_threshold_sum / below_threshold_count;
		above_threshold_avg = above_threshold_count == 0 ? min_pixel : above_threshold_sum / above_threshold_count;
		curr_threshold = (unsigned char)(((int) (below_threshold_avg + above_threshold_avg)) / 2);

	}
	dim3 blockDim2(256,1);
	dim3 gridDim2((w * h + blockDim2.x - 1) / blockDim2.x);
	if (reverse) {
		kernelReverseBinarize<<<gridDim2,blockDim2>>>(w,h,curr_threshold,input_image);
	} else {
		kernelBinarize<<<gridDim2,blockDim2>>>(w,h,curr_threshold,input_image);
	}
	
	cudaDeviceSynchronize();
	cudaFree(subset);
	cudaFree(subset2);
	cudaFree(subset_count);

}

__device__ unsigned long long int cudaConvolution(unsigned char *input_image, int row, int col, int width, int height) {
	// do convolution in both x-direction and y-direction
	int horizontal_total = 0;
	int vertical_total = 0;

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

	return (unsigned long long int)(sqrtf(horizontal_total * horizontal_total + vertical_total * vertical_total));
}

__global__ void kernelSobelFindLimit(int width, int height, unsigned char *input_image,
	                                 unsigned long long int *max_val, unsigned long long int *min_val) {
	__shared__ unsigned long long int shared_min_val;// = ULLONG_MAX;
	__shared__ unsigned long long int shared_max_val;// = 0;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadIdx.x == 0) {
		shared_min_val = ULLONG_MAX;
		shared_max_val = 0;
	}
	__syncthreads();
	int row = index / width;
	int col = index % width;
	unsigned long long int result = cudaConvolution(input_image, row, col, width, height);
	atomicMax(&shared_max_val, result);
	atomicMin(&shared_min_val, result);
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicMax(max_val, shared_max_val);
		atomicMin(min_val, shared_min_val);
	}
}

__global__ void kernelSobel(int width, int height, unsigned char *input_image, unsigned char *output_image,
	                        unsigned long long int max_val, unsigned long long int min_val) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int row = index / width;
	int col = index % width;
	unsigned long long int result = cudaConvolution(input_image, row, col, width, height);

	output_image[row * width + col] = MAX_BRIGHTNESS * (result - min_val) / (max_val - min_val);
}

void cudaSobel(int width, int height) {
	//printf("in cudaSobel\n");
	unsigned char *output_image;

	unsigned long long int *max_val;// = 0;
	unsigned long long int *min_val;// = ULLONG_MAX;
	cudaMallocManaged(&max_val, sizeof(unsigned long long int));
	cudaMallocManaged(&min_val, sizeof(unsigned long long int));
	//printf("in here 1\n");
	*max_val = 0;
	*min_val = ULLONG_MAX;
	cudaMallocManaged(&output_image, width * height * sizeof(unsigned char));
	//printf("in here\n");
	dim3 blockDim(256,1);
	dim3 gridDim((width * height + blockDim.x - 1) / blockDim.x);
	//printf("about to find limit\n");
	kernelSobelFindLimit<<<gridDim,blockDim>>>(width, height, tempImage, max_val, min_val);
	cudaDeviceSynchronize();
	//printf("max_val is %d, min_val is %d\n", (int)max_val, (int)min_val);
	kernelSobel<<<gridDim,blockDim>>>(width, height, tempImage, output_image, *max_val, *min_val);

	//unsigned char *result = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	cudaMemcpy(tempImage, output_image, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	cudaFree(output_image);
	cudaFree(max_val);
	cudaFree(min_val);
	//printf("finished sobel\n");
}

__device__ unsigned int find_root(unsigned int *union_find, unsigned int label) {
	while (union_find[label] != label) {
		label = union_find[label];
	}
	return label;
}

unsigned int host_find_root(unsigned int *union_find, unsigned int label) {
	while (union_find[label] != label) {
		label = union_find[label];
	}
	return label;
}

__global__ void kernelBlockSegment(unsigned int width, unsigned int height, unsigned char *image, unsigned int *labels) {
	unsigned int curr, left_upper, upper, right_upper, left;
	unsigned int left_upper_label, upper_label, right_upper_label, left_label, min_label;
	unsigned int block_index = blockIdx.y * gridDim.x + blockIdx.x;
	unsigned int offset = block_index * BLOCK_HEIGHT * BLOCK_WIDTH;
	unsigned int count = 1;
	unsigned int upper_limit = min(blockIdx.y * blockDim.y, height - 1);
	unsigned int lower_limit = min((blockIdx.y + 1) * blockDim.y, height);
	unsigned int left_limit = min(blockIdx.x * blockDim.x, width - 1);
	unsigned int right_limit = min((blockIdx.x + 1) * blockDim.x, width);
	__shared__ unsigned int union_find[BLOCK_HEIGHT * BLOCK_WIDTH + 1];

	if (threadIdx.x == 0 && threadIdx.y == 0) { // only the first index will do the segmentation
		//printf("offset is %d\n", offset);
		for (int i = 0; i < BLOCK_WIDTH * BLOCK_HEIGHT + 1; i ++) { // initial labels don't have parents
			union_find[i] = i;
		}

		
		//printf("upper: %d, lower: %d, left: %d, right: %d\n", (int)upper_limit, (int)lower_limit,
		//	(int)left_limit, (int)right_limit);

		for (unsigned int row = upper_limit; row < lower_limit; row ++) {
			for (unsigned int col = left_limit; col < right_limit; col ++) {
				curr = row * width + col;

				if (image[curr] == 0) {
					labels[curr] = 0;
					continue;
				}

				left_upper_label = UINT_MAX;
				upper_label = UINT_MAX;
				right_upper_label = UINT_MAX;
				left_label = UINT_MAX;

				// top left
				if (row != upper_limit && col != left_limit) {
					left_upper = (row - 1) * width + col - 1;
					if (image[left_upper] == image[curr]) {
						left_upper_label = find_root(union_find, labels[left_upper]);
					}
				}

				// top
				if (row != upper_limit) {
					upper = (row - 1) * width + col;
					if (image[upper] == image[curr]) {
						upper_label = find_root(union_find, labels[upper]);
					}
				}

				// top right
				if (row != upper_limit && col != right_limit - 1) {
					right_upper = (row - 1) * width + col + 1;
					if (image[right_upper] == image[curr]) {
						right_upper_label = find_root(union_find, labels[right_upper]);
					}
				}

				// left
				if (col != left_limit) {
					left = row * width + col - 1;
					if (image[left] == image[curr]) {
						left_label = find_root(union_find, labels[left]);
					}
				}

				min_label = min(min(min(left_upper_label, upper_label), right_upper_label), left_label);

				if (min_label == UINT_MAX) {
					labels[curr] = count;
					count ++;
				} else {
					labels[curr] = min_label;
					if (left_upper_label != UINT_MAX && left_upper_label > min_label) {
						union_find[left_upper_label] = min_label;
					}

					if (upper_label != UINT_MAX && upper_label > min_label) {
						union_find[upper_label] = min_label;
					}

					if (right_upper_label != UINT_MAX && right_upper_label > min_label) {
						union_find[right_upper_label] = min_label;
					}

					if (left_label != UINT_MAX && left_label > min_label) {
						union_find[left_label] = min_label;
					}
				}

			}
		}
	}

	__syncthreads();
	
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	if (row >= 0 && col >= 0 && row < height && col < width) {
		unsigned int label = labels[row * width + col];
		labels[row * width + col] = find_root(union_find, label) + offset;
	}

	if (threadIdx.x == 0 && threadIdx.y == 0) {

		int indicator[BLOCK_WIDTH * BLOCK_HEIGHT];
		int num_seg = 0;
		unsigned int min_label = UINT_MAX;
		unsigned int max_label = 0;

		for (int i = 0; i < BLOCK_HEIGHT * BLOCK_WIDTH; i ++) {
			indicator[i] = 0;
		}

		for (unsigned int row = upper_limit; row < lower_limit; row ++) {
			for (unsigned int col = left_limit; col < right_limit; col ++) {
				unsigned int actual_index = row * width + col;
				unsigned int label = labels[actual_index] - offset;

				if (image[actual_index] && !indicator[label - 1]) {
					min_label = min(min_label, label + offset);
					max_label = max(max_label, label + offset);
					num_seg += 1;
					indicator[label - 1] = 1;
				}
			}
		}
	}
	
}

__global__ void kernelResolveEdges(int width, int height, unsigned char *input_image, unsigned int *labels, unsigned int *union_find) {
	int row, col, index, left_index, left_lower_index, left_upper_index, upper_index, right_upper_index;
	unsigned int label;

	if (threadIdx.x == 0) { // leftmost column
		row = blockDim.y * blockIdx.y + threadIdx.y;
		col = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("checking row %d, col%d\n", (int)row, (int)col);
		if (row >= 0 && row < height && col != 0 && col < width && input_image[row * width + col]) {
			index = row * width + col;
			label = labels[index];
			left_index = row * width + col - 1;
			if (input_image[left_index]) {
				atomicMin(union_find + label, labels[left_index]);
			}

			if (row != 0) {
				left_upper_index = (row - 1) * width + col - 1;
				if (input_image[left_upper_index]) {
					atomicMin(union_find + label, labels[left_upper_index]);
				}
			}

			if (row != height - 1) {
				left_lower_index = (row + 1) * width + col - 1;
				if (input_image[left_lower_index]) {
					atomicMin(union_find + label, labels[left_lower_index]);
				}
			}
		}
	}
	if (threadIdx.y == 0) { // topmost row
		row = blockDim.y * blockIdx.y + threadIdx.y;
		col = blockDim.x * blockIdx.x + threadIdx.x;
		//printf("checking row %d, col%d\n", (int)row, (int)col);
		if (row != 0 && col >= 0 && col < width && row < height && input_image[row * width + col]) {
			index = row * width + col;
			//printf("checking index %d\n", index);
			label = labels[index];
			upper_index = (row - 1) * width + col;
			if (input_image[upper_index]) { // upper pixel
				atomicMin(union_find + label, labels[upper_index]);
			}
			//printf("finished finding upper\n");

			if (col != 0) { // left upper pixel
				left_upper_index = (row - 1) * width + col - 1;
				if (input_image[left_upper_index]) {
					atomicMin(union_find + label, labels[left_upper_index]);
				}
			}
			//printf("finished finding left_upper\n");
			if (col != width - 1) { // right upper pixel
				right_upper_index = (row - 1) * width + col + 1;
				if (input_image[right_upper_index]) {
					atomicMin(union_find + label, labels[right_upper_index]);
				}
			}
			//printf("finished finding right_upper\n");
		}
	}
}

__global__ void kernelUpdateLabel(unsigned int width, unsigned int height, unsigned char *input_image, unsigned int *labels, unsigned int *union_find) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int index, label;

	if (row >= 0 && row < height && col >= 0 && col < width) {
		index = row * width + col;
		if (input_image[index]) {
			label = labels[index];
			while (union_find[label] != label) {
				label = union_find[label];
			}
			labels[index] = label;
		}
	}
}

int *cudaSegment(int width, int height, unsigned char *input_image, int *num_seg) {
	//unsigned char *cuda_image;
	unsigned int *labels;
	unsigned int *union_find;

	//cudaMallocManaged(&cuda_image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	//cudaMemcpy(cuda_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMallocManaged(&labels, width * height * sizeof(unsigned int));

	dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	kernelBlockSegment<<<gridDim, blockDim>>>(width, height, input_image, labels);
	cudaDeviceSynchronize();

	unsigned int rounded_width = (width + blockDim.x - 1) / blockDim.x * blockDim.x;
	unsigned int rounded_height = (height + blockDim.y - 1) / blockDim.y * blockDim.y;
	cudaMallocManaged(&union_find, rounded_width * rounded_height * sizeof(unsigned int));
	for (unsigned int i = 0; i < rounded_width * rounded_height; i ++) {
		union_find[i] = i;
	}
	
	//printf("about to resolve edges\n");
	//kernelResolveEdges<<<gridDim, blockDim>>>(width, height, cuda_image, labels, union_find);
	//cudaDeviceSynchronize();

	unsigned int curr, left_upper, left, left_lower, upper, right_upper;
	unsigned int min_label, curr_label, left_upper_label, left_label, left_lower_label, upper_label, right_upper_label;
	for (unsigned int x = blockDim.x; x < width; x += blockDim.x) {
		for (unsigned int y = 0; y < height; y ++) {
			curr = y * width + x;
			if (input_image[curr] == 0) continue;

			curr_label = host_find_root(union_find, labels[curr]);

			left_upper_label = UINT_MAX;
			left_lower_label = UINT_MAX;
			left_label = UINT_MAX;

			// left
			left = y * width + x - 1;
			if (input_image[left]) {
				left_label = host_find_root(union_find, labels[left]);
			}

			// top left
			if (y != 0) {
				left_upper = (y - 1) * width + x - 1;
				if (input_image[left_upper]) {
					left_upper_label = host_find_root(union_find, labels[left_upper]);
				}
			}

			// lower left
			if (y != height - 1) {
				left_lower = (y + 1) * width + x - 1;
				if (input_image[left_lower]) {
					left_lower_label = host_find_root(union_find, labels[left_lower]);
				}
			}

			min_label = std::min(std::min(std::min(left_upper_label, left_label), left_lower_label), curr_label);

			if (min_label < curr_label) {
				union_find[curr_label] = min_label;
			}

			if (left_upper_label != UINT_MAX && min_label < left_upper_label) {
				union_find[left_upper_label] = min_label;
			}

			if (left_label != UINT_MAX && min_label < left_label) {
				union_find[left_label] = min_label;
			}

			if (left_lower_label != UINT_MAX && min_label < left_lower_label) {
				union_find[left_lower_label] = min_label;
			}
		}
	}

	//printf("finished all vertical edges\n");
	for (unsigned int y = blockDim.y; y < height; y += blockDim.y) {
		for (unsigned int x = 0; x < width; x ++) {
			curr = y * width + x;
			if (input_image[curr] == 0) continue;

			curr_label = host_find_root(union_find, labels[curr]);
			left_upper_label = UINT_MAX;
			upper_label = UINT_MAX;
			right_upper_label = UINT_MAX;

			// upper
			upper = (y - 1) * width + x;
			if (input_image[upper]) {
				upper_label = host_find_root(union_find, labels[upper]);
			}

			// top left
			if (x != 0) {
				left_upper = (y - 1) * width + x - 1;
				if (input_image[left_upper]) {
					left_upper_label = host_find_root(union_find, labels[left_upper]);
				}
			}

			// top right
			if (x != width - 1) {
				right_upper = (y - 1) * width + x + 1;
				if (input_image[right_upper]) {
					right_upper_label = host_find_root(union_find, labels[right_upper]);
				}
			}

			min_label = min(min(min(upper_label, left_upper_label), right_upper_label), curr_label);

			if (min_label < curr_label) {
				//printf("%d point to %d\n", curr_label, min_label);
				union_find[curr_label] = min_label;
			}

			if (left_upper_label != UINT_MAX && min_label < left_upper_label) {
				//printf("%d point to %d\n", left_upper_label, min_label);
				union_find[left_upper_label] = min_label;
			}

			if (upper_label != UINT_MAX && min_label < upper_label) {
				///printf("%d point to %d\n", upper_label, min_label);
				union_find[upper_label] = min_label;
			}

			if (right_upper_label != UINT_MAX && min_label < right_upper_label) {
				//printf("%d point to %d\n", right_upper_label, min_label);
				union_find[right_upper_label] = min_label;
			}
		}
	}

	//printf("finished edge resolve\n");
	
	kernelUpdateLabel<<<gridDim, blockDim>>>(width, height, input_image, labels, union_find);
	cudaDeviceSynchronize();
	//printf("finished all computation\n");
	
	int *segments = (int*)calloc(sizeof(int), rounded_width * rounded_height);

	int count = 1;
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			unsigned int label = labels[row * width + col];
			if (!segments[label]) {
				segments[label] = count;
				count ++;
			}
		}
	}

	int x1[count - 1];
	int y1[count - 1];
	int x2[count - 1];
	int y2[count - 1];

	for (int i = 0; i < count - 1; i ++) {
		x1[i] = INT_MAX;
		x2[i] = INT_MAX;
		y2[i] = INT_MAX;
		y1[i] = INT_MAX;
	}

	//printf("finding limits\n");
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			

			if (input_image[row * width + col]) {
				//num_seg += 1;
				//indicator[label] = 1;
				unsigned int label = labels[row * width + col];

				if (x1[segments[label]] == INT_MAX) {
					//printf("new label:%d\n", label);
					x1[segments[label]] = col;
					x2[segments[label]] = col;
					y1[segments[label]] = row;
					y2[segments[label]] = row;
				} else {
					x1[segments[label]] = std::min(col, x1[segments[label]]);
					x2[segments[label]] = std::max(col, x2[segments[label]]);
					y1[segments[label]] = std::min(row, y1[segments[label]]);
					y2[segments[label]] = std::max(row, y2[segments[label]]);
				}

			}
		}
	}
	free(segments);

	///printf("about to write image\n");
	*num_seg = 0;
	int *segments1;
	for (int i = 0; i < count - 1; i ++) {
		int left_coord = std::max(0, x1[i] - 2);
		int right_coord = std::min(width, x2[i] + 2);
		int upper_coord = std::max(0, y1[i] - 2);
		int lower_coord = std::min(height, y2[i] + 2);
	
		int rwidth = right_coord - left_coord;
		int rheight = lower_coord - upper_coord;
		if (rheight / rwidth >= 1 && rheight / rwidth <= 5) {
			if (rwidth * rheight >= width * height / 35) {
				*num_seg = *num_seg + 1;
			}
		}
	}
	
	segments1 = (int*)malloc(sizeof(int) * 4 * (*num_seg));
	int count1 = 0;
	for (int i = 0; i < count - 1; i ++) {
		int left_coord = std::max(0, x1[i] - 2);
		int right_coord = std::min(width, x2[i] + 2);
		int upper_coord = std::max(0, y1[i] - 2);
		int lower_coord = std::min(height, y2[i] + 2);
	
		int rwidth = right_coord - left_coord;
		int rheight = lower_coord - upper_coord;
		if (rwidth != 0 && rheight != 0 && rheight / rwidth >= 1 && rheight / rwidth <= 5) {
			if (rwidth * rheight >= width * height / 35) {
				segments1[count1 * 4] = left_coord;
				segments1[count1 * 4 + 1] = right_coord;
				segments1[count1 * 4 + 2] = upper_coord;
				segments1[count1 * 4 + 3] = lower_coord;
				count1 ++;
			}
		}
	}
	cudaFree(union_find);
	cudaFree(labels);
	return segments1;
}

unsigned char *cudaExtractRegion(int x1, int x2, int y1, int y2, int width, int height, unsigned char *original_image) {
	int cwidth = x2 - x1;
	int cheight = y2 - y1;

	unsigned char *output_image;
	cudaMallocManaged(&output_image, sizeof(unsigned char) * cwidth * cheight);
    for (int y = y1; y < y2; y ++) {
		for (int x = x1; x < x2; x ++) {
			output_image[(y - y1) * cwidth + (x - x1)] = original_image[y * width + x];
		}
	}
	return output_image;
}

void cudaDraw(int *segments, int num_seg, int width, int height, unsigned char *input_image) {

	for (int i = 0; i < num_seg; i ++) {
		int x1 = std::max(0, segments[4 * i] - 2);
		int x2 = std::min(width, segments[4 * i + 1] + 2);
		int y1 = std::max(0, segments[4 * i + 2] - 2);
		int y2 = std::min(height, segments[4 * i + 3]);

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

unsigned char *cudaFindRegion(int width, int height, int *output_width, int *output_height) {
	//unsigned char *cuda_image;
	unsigned int *labels;
	unsigned int *union_find;

	//cudaMallocManaged(&cuda_image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	//cudaMemcpy(cuda_image, image, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMallocManaged(&labels, width * height * sizeof(unsigned int));

	dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	kernelBlockSegment<<<gridDim, blockDim>>>(width, height, tempImage, labels);
	cudaDeviceSynchronize();

	unsigned int rounded_width = (width + blockDim.x - 1) / blockDim.x * blockDim.x;
	unsigned int rounded_height = (height + blockDim.y - 1) / blockDim.y * blockDim.y;
	cudaMallocManaged(&union_find, rounded_width * rounded_height * sizeof(unsigned int));
	for (unsigned int i = 0; i < rounded_width * rounded_height; i ++) {
		union_find[i] = i;
	}
	
	//printf("about to resolve edges\n");
	//kernelResolveEdges<<<gridDim, blockDim>>>(width, height, cuda_image, labels, union_find);
	//cudaDeviceSynchronize();

	unsigned int curr, left_upper, left, left_lower, upper, right_upper;
	unsigned int min_label, curr_label, left_upper_label, left_label, left_lower_label, upper_label, right_upper_label;
	for (unsigned int x = blockDim.x; x < width; x += blockDim.x) {
		for (unsigned int y = 0; y < height; y ++) {
			curr = y * width + x;
			if (tempImage[curr] == 0) continue;

			curr_label = host_find_root(union_find, labels[curr]);

			left_upper_label = UINT_MAX;
			left_lower_label = UINT_MAX;
			left_label = UINT_MAX;

			// left
			left = y * width + x - 1;
			if (tempImage[left]) {
				left_label = host_find_root(union_find, labels[left]);
			}

			// top left
			if (y != 0) {
				left_upper = (y - 1) * width + x - 1;
				if (tempImage[left_upper]) {
					left_upper_label = host_find_root(union_find, labels[left_upper]);
				}
			}

			// lower left
			if (y != height - 1) {
				left_lower = (y + 1) * width + x - 1;
				if (tempImage[left_lower]) {
					left_lower_label = host_find_root(union_find, labels[left_lower]);
				}
			}

			min_label = std::min(std::min(std::min(left_upper_label, left_label), left_lower_label), curr_label);

			if (min_label < curr_label) {
				//printf("%d point to %d\n", curr_label, min_label);
				union_find[curr_label] = min_label;
			}

			if (left_upper_label != UINT_MAX && min_label < left_upper_label) {
				//printf("%d point to %d\n", left_upper_label, min_label);
				union_find[left_upper_label] = min_label;
			}

			if (left_label != UINT_MAX && min_label < left_label) {
				//printf("%d point to %d\n", left_label, min_label);
				union_find[left_label] = min_label;
			}

			if (left_lower_label != UINT_MAX && min_label < left_lower_label) {
				//printf("%d point to %d\n", left_lower_label, min_label);
				union_find[left_lower_label] = min_label;
			}
		}
	}

	//printf("finished all vertical edges\n");
	for (unsigned int y = blockDim.y; y < height; y += blockDim.y) {
		for (unsigned int x = 0; x < width; x ++) {
			curr = y * width + x;
			if (tempImage[curr] == 0) continue;

			curr_label = host_find_root(union_find, labels[curr]);
			left_upper_label = UINT_MAX;
			upper_label = UINT_MAX;
			right_upper_label = UINT_MAX;

			// upper
			upper = (y - 1) * width + x;
			if (tempImage[upper]) {
				upper_label = host_find_root(union_find, labels[upper]);
			}

			// top left
			if (x != 0) {
				left_upper = (y - 1) * width + x - 1;
				if (tempImage[left_upper]) {
					left_upper_label = host_find_root(union_find, labels[left_upper]);
				}
			}

			// top right
			if (x != width - 1) {
				right_upper = (y - 1) * width + x + 1;
				if (tempImage[right_upper]) {
					right_upper_label = host_find_root(union_find, labels[right_upper]);
				}
			}

			min_label = min(min(min(upper_label, left_upper_label), right_upper_label), curr_label);

			if (min_label < curr_label) {
				//printf("%d point to %d\n", curr_label, min_label);
				union_find[curr_label] = min_label;
			}

			if (left_upper_label != UINT_MAX && min_label < left_upper_label) {
				//printf("%d point to %d\n", left_upper_label, min_label);
				union_find[left_upper_label] = min_label;
			}

			if (upper_label != UINT_MAX && min_label < upper_label) {
				//printf("%d point to %d\n", upper_label, min_label);
				union_find[upper_label] = min_label;
			}

			if (right_upper_label != UINT_MAX && min_label < right_upper_label) {
				//printf("%d point to %d\n", right_upper_label, min_label);
				union_find[right_upper_label] = min_label;
			}
		}
	}

	//printf("finished edge resolve\n");
	
	kernelUpdateLabel<<<gridDim, blockDim>>>(width, height, tempImage, labels, union_find);
	cudaDeviceSynchronize();
	//printf("finished all computation\n");
	
	int *segments = (int*)calloc(sizeof(int), rounded_width * rounded_height);

	int count = 1;
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			unsigned int label = labels[row * width + col];
			if (!segments[label]) {
				segments[label] = count;
				count ++;
			}
		}
	}

	int x1[count - 1];
	int y1[count - 1];
	int x2[count - 1];
	int y2[count - 1];

	for (int i = 0; i < count - 1; i ++) {
		x1[i] = INT_MAX;
		x2[i] = INT_MAX;
		y2[i] = INT_MAX;
		y1[i] = INT_MAX;
	}

	//printf("finding limits\n");
	for (int row = 0; row < height; row ++) {
		for (int col = 0; col < width; col ++) {
			

			if (tempImage[row * width + col]) {
				//num_seg += 1;
				//indicator[label] = 1;
				unsigned int label = labels[row * width + col];

				if (x1[segments[label]] == INT_MAX) {
					//printf("new label:%d\n", label);
					x1[segments[label]] = col;
					x2[segments[label]] = col;
					y1[segments[label]] = row;
					y2[segments[label]] = row;
				} else {
					x1[segments[label]] = std::min(col, x1[segments[label]]);
					x2[segments[label]] = std::max(col, x2[segments[label]]);
					y1[segments[label]] = std::min(row, y1[segments[label]]);
					y2[segments[label]] = std::max(row, y2[segments[label]]);
				}

			}
		}
	}

	free(segments);
	int suitable = 0;
	int max_num_seg = -1;
	int best_x1, best_x2, best_y1, best_y2;
	int num_seg1, num_seg2;

	for (int i = 0; i < count - 1; i ++) {
		int l1 = x1[i];//std::max(0, x1[i] - 2);
		int l2 = x2[i];//std::min(width, x2[i] + 2);
		int r1 = y1[i];//std::max(0, y1[i] - 2);
		int r2 = y2[i];//std::min(height, y2[i] + 2);
		
		int cwidth = l2 - l1;
		int cheight = r2 - r1;
		if (cwidth < 10 * cheight && 2 * cwidth > 3 * cheight &&
			cwidth * cheight * 50> width * height &&
			cwidth * cheight * 10 < width * height) {
			suitable ++;

		    printf("testing suitable region %d: x1: %d, x2: %d, y1: %d, y2: %d ...\n", suitable, l1, l2, r1, r2);
		    unsigned char *output_image1 = cudaExtractRegion(l1, l2, r1, r2, width, height, originalImage);
		    unsigned char *output_image2 = cudaExtractRegion(l1, l2, r1, r2, width, height, original2Image);
		    num_seg1 = 0;
		    num_seg2 = 0;
		    cudaBinarize(cwidth, cheight, output_image1, 0);
		    cudaBinarize(cwidth, cheight, output_image2, 1);
		    int *best_segments1 = cudaSegment(cwidth, cheight, output_image1, &num_seg1);
		    int *best_segments2 = cudaSegment(cwidth, cheight, output_image2, &num_seg2);
		    int curr_num_seg = max(num_seg1, num_seg2);
		    printf("suitable region %d has maximum %d segments\n", suitable, curr_num_seg);
		    if (max_num_seg == -1 || curr_num_seg > max_num_seg) {
		    	best_x1 = l1;
		    	best_x2 = l2;
		    	best_y1 = r1;
		    	best_y2 = r2;
		    	max_num_seg = curr_num_seg;
		    }
		    cudaFree(output_image1);
		    cudaFree(output_image2);
		    free(best_segments2);
		    free(best_segments1);
			
		}
	}

	cudaFree(union_find);
	cudaFree(labels);
	printf("total suitable region: %d, outputting the best one...\n", suitable);

	if (suitable == 0) {
		return NULL;
	}

	*output_width = best_x2 - best_x1;
	*output_height = best_y2 - best_y1;
	unsigned char *output_image1 = cudaExtractRegion(best_x1, best_x2, best_y1, best_y2, width, height, originalImage);
	unsigned char *output_image2 = cudaExtractRegion(best_x1, best_x2, best_y1, best_y2, width, height, original2Image);
	cudaBinarize(*output_width, *output_height, output_image1, 0);
	cudaBinarize(*output_width, *output_height, output_image2, 1);
	num_seg1 = 0;
	num_seg2 = 0;
	int *best_segments1 = cudaSegment(*output_width, *output_height, output_image1, &num_seg1);
	int *best_segments2 = cudaSegment(*output_width, *output_height, output_image2, &num_seg2);
	if (num_seg1 > num_seg2) {
		cudaDraw(best_segments1, num_seg1, *output_width, *output_height, output_image1);
		printf("number of number segments is %d\n", num_seg1);
		cudaFree(output_image2);
		free(best_segments1);
		free(best_segments2);
		return output_image1;
	}
	cudaDraw(best_segments2, num_seg2, *output_width, *output_height, output_image2);
	printf("number of number segments is %d\n", num_seg2);
	cudaFree(output_image1);
	free(best_segments2);
	free(best_segments1);
	return output_image2;
}

unsigned char*
cudaMain(int width, int height, png_bytep *row_pointers, int *output_width, int *output_height, int type, int dilation) {
	int msec;

	cudaMallocManaged(&tempImage, width * height * sizeof(unsigned char));
	clock_t start = clock();
	cudaPreprocess(width, height, row_pointers, type);
	//seq_process_file(width,height,row_pointers,type);
	clock_t time1 = clock();
	msec = (time1 - start) * 1000 / CLOCKS_PER_SEC;
	printf("preprocess png file takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	
	cudaMallocManaged(&originalImage, width * height * sizeof(unsigned char));
	cudaMemcpy(originalImage, tempImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	cudaMallocManaged(&original2Image, width * height * sizeof(unsigned char));
	cudaMemcpy(original2Image, tempImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToDevice);
	//cudaBinarize(width, height, originalImage, 1);
	// *output_width = width;
	// *output_height = height;
	// return originalImage;
	clock_t time2 = clock();
	msec = (time2 - time1) * 1000 / CLOCKS_PER_SEC;
	printf("binarizing original image in both ways takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	
	//cudaBinarize(width, height, original2Image, 0);

	cudaSobel(width, height);
	clock_t time3 = clock();
	msec = (time3 - time2) * 1000 / CLOCKS_PER_SEC;
	printf("sobel edge detection takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	cudaBinarize(width, height, tempImage, 0);
	clock_t time4 = clock();
	msec = (time4 - time3) * 1000 / CLOCKS_PER_SEC;
	printf("binarzing takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	cudaDilation(width, height, max(2,height / 400), max(5,width / 150));
	// *output_width = width;
	// *output_height = height;
	// return tempImage;
	clock_t time5 = clock();
	msec = (time5 - time4) * 1000 / CLOCKS_PER_SEC;
	printf("dilation takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	if (dilation) {
		*output_width = width;
		*output_height = height;
		return tempImage;
	}
	
	unsigned char *output_image = cudaFindRegion(width, height, output_width, output_height);
	clock_t time6 = clock();
	msec = (time6 - time5) * 1000 / CLOCKS_PER_SEC;
	printf("finding the location and determing number segments takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);
	if (output_image == NULL) {
		printf("failed to find region, outputting dilation image only...\n");
		*output_width = width;
		*output_height = height;
		return tempImage;
	}

	return output_image;
	/*
	if (coordinates == NULL) {
		*output_width = width;
		*output_height = height;
		return tempImage;
	}
	clock_t time7 = clock();
	msec = (time7 - time6) * 1000 / CLOCKS_PER_SEC;
	printf("finding the location of original image takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	unsigned char *output_image1 = cudaExtractRegion(coordinates, width, height, originalImage);
	unsigned char *output_image2 = cudaExtractRegion(coordinates, width, height, original2Image);
	clock_t time8 = clock();
	msec = (time8 - time7) * 1000 / CLOCKS_PER_SEC;
	printf("Extracting plate region from both original images takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	int num_seg1 = cudaSegment(*output_width, *output_height, output_image1);
	clock_t time9 = clock();
	msec = (time9 - time8) * 1000 / CLOCKS_PER_SEC;
	printf("finding number segments of output image 1 takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	int num_seg2 = cudaSegment(*output_width, *output_height, output_image2);
	clock_t time10 = clock();
	msec = (time10 - time9) * 1000 / CLOCKS_PER_SEC;
	printf("finding number segments of output image 2 takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	msec = (time10 - start) * 1000 / CLOCKS_PER_SEC;
	printf("The whole process takes %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	printf("Number of number segments detected in original image 1: %d\n", num_seg1);
	printf("Number of number segments detected in original image 2: %d\n", num_seg2);
	if (num_seg1 >= num_seg2) {
		cudaFree(output_image2);
		return output_image1;
	}

	cudaFree(output_image1);
	return output_image2;*/
}

