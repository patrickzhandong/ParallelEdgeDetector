#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <png.h>
#include "cudaprocess.h"
#include <algorithm>

void dilation(int xradius,int yradius,int type, int width, int height, unsigned char *input_image) {
	unsigned char output_image[width * height];
	//horizontal
	if (type == 0) {
		for (int i = 0; i < height; i ++) {
			for (int j = 0; j < width; j++) {
				output_image[i * width + j] = 0;
				for (int k = j - yradius; k <= j + yradius; k++) {
					if (k >= 0 && k < width && input_image[i * width + k] == 255) {
						output_image[i * width + j] = 255;
						break;
					}
				}
				
			}
		}
	}
	//vertical
	if (type == 1) {
		for (int i = 0; i < height; i ++) {
			for (int j = 0; j < width; j++) {
				output_image[i * width + j] = 0;
				for (int k = i - xradius; k <= i + xradius; k++) {
					if (k >= 0 && k < height && input_image[k * width + j] == 255) {
						output_image[i * width + j] = 255;
						break;
					}
				}

			}
				
		}
	}
	//square
	if (type == 2) {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				output_image[i * width + j] = 0;
				for (int k = i - xradius; k <= i + xradius; k++) {
					for (int l = j - yradius; l <= j + yradius; l++) {
						if (k >= 0 && k < height &&
							l >= 0 && l < width &&
							input_image[k * width + l] == 255) {
							output_image[i * width + j] = 255;
							break;
						}
					}
					if (output_image[i * width + j] == 255) {
						break;
					}
				}
			}
		}
	}
	for (int i = 0; i < height * width; i++) {
		input_image[i] = output_image[i];
	}
}

void erosion(int radius,int width, int height, unsigned char *input_image) {
	unsigned char output_image[width * height];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			output_image[i * width + j] = 255;
			for (int k =  i - radius; k <= i + radius; k++) {
				for (int l = j - radius; l <= j + radius; l++) {
					if (k >= 0 && k < height &&
						l >= 0 && l < width &&
						input_image[k * width + l] == 0) {
						output_image[i * width + j] = 0;
						break;
					}
				}
				if (output_image[i * width + j] == 0) {
					break;
				}
			}
		}
	}
	for (int i = 0; i < height * width; i++) {
		input_image[i] = output_image[i];
	}
}
void location(int width, int height, unsigned char *input_image) {
	//erosion(1,width,height,input_image);
	dilation(std::max(2, height / 400), std::max(5, width / 150),2,width,height,input_image);
	//erosion(5,width,height,input_image);
	//dilation(10,0,width,height,input_image);

}