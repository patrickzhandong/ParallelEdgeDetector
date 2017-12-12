
<img src="/title.jpeg">

## A 418 Final Project by Junye(Hans) Chen and Zhan(Patrick) Dong



[Proposal](https://github.com/patrickzhandong/ParallelPlateRecoginition/blob/master/proposal.pdf)
[Checkpoint](https://github.com/patrickzhandong/ParallelPlateRecoginition/blob/master/CheckpointReport.pdf)

### Summary

______

In this project, we implement parallel license plate recognition using CUDA on the GPU of GHC machines. The program takes in a PNG file as input, and outputs the plate region with bounding boxes identifying the characters on the plate. We compare the parallel solution with the sequential solution on CPU and observe a good speedup. It also has a satisfying accuracy. 



### Background

______

#### Program Overview

In this project, we implement our program from scratch based on the algorithm from the paper written by Mahesh Babu K and M V Raghunadh[1]. We implement a multi-step algorithm involving many classical image processing algorithms, and use our own ideas to do the parallelization for each step. The general process of our program is shown in the flow chart below:

<p align="center">
  <img height="350" src="https://github.com/patrickzhandong/ParallelPlateRecoginition/blob/master/program1%20(1).jpg" />

</p>

The input of our program is a PNG file that contains a photo of the back of a car with a license plate, and the ideal output is a PGM file (grayscale of the original image) that contains the plate, with bounding boxes around character segments. Throughout the process, we keep making changes on the array representing the image, trying to extracting the important regions and getting the information we want.

#### Computation, Locality and Dependencies Analysis
The entire algorithm is, as shown above, quite computation intensive. Basic operations like applying a kernel to every pixel in the image is extremely suitable for CUDA, since we execute simple calculations for a large amount of times, and GPU is really good at such operations. Although these simple operations don’t require that much time by default, there are some relatively complicated operations like binarization and image segmentation that provide us with a lot of potentials for speedup. These parts will be the major source of our speedup. 
The locality of most parts of this program is good, since we are only looking at neighboring elements in the array most of the time. However, we use a union find data structure to represent connected components in the image during segmentation. This requires a lot of pointer chasing, and we have to access the array in a chaotic order. That may result in very bad locality, with lots of cache misses.
The most parts of the program also have good dependencies requirements. In many parts of the program, every pixel can be processed at the same time. But in the segmentation part, the result of each pixel depends on the three pixels on its top and the pixel on its left, so we need to find a wise way to parallelize the segmentation step.
From the analysis above, we can observe that the segmentation part is the most challenging part in our program, with bad localities and strong dependencies. It is also the most computation intensive part, so it also provides a lot of potentials for speedup. A good parallelization of this step will result in a very satisfying general speedup.

### Approach

____

For this project, we use the programming language C and C++. We also use built-in Linux libpng for reading and processing PNG files, and CUDA to implement parallelization. 
We run all our experiments on GHC machines with GPU Nvidia GeForce GTX 1080. For the sequential part of the code, we have Intel(R) Xeon(R) CPU E5-1660 v4. 
We write most parts of our codes from scratch, and implement most of the image processing operations on our own, so that we can parallelize without any constraints.
Since our project consists of different image-processing steps, we believe that CUDA is the most suitable way for parallelization. We mainly use CUDA in two ways. For some of the steps, we generate a lot of threads, and deal with the operations of 1 pixel on 1 thread. For the rest of the steps, we break the entire graph into blocks, and generate corresponding number of threads, where each thread takes care of a block. We use arrays with size of the number of threads to get partial results from each block, and combine the subset sums and results, obtaining our final results.
Our entire program consists of several steps, and we parallelize each step to ensure maximum overall speedup.


# Schedule
Nov.20 - Nov. 23: Continue working on the implementation for character segmentation, improve its accuracy. Working on the minor steps after edge detection to find the plate region. 

Nov.24 - Nov. 26: Working on character recognition with existing library, start parallelizing edge detection.Continue working on the minor steps. Start parallelizing the preprocessing step.

Nov.27 - Nov. 30: Continue working on parallelizing edge detection.Continue parallelizing the preprocessing step.

Dec.1 - Dec. 3: Start parallelizing the segmentation step.Continue parallelizing the segmentation step.Trying different preprocessing techniques to improve the accuracy. Start working on character recognition (If have time).

Dec.4 - Dec.7:Continue working on the work undone.Working  on character recognition(If have time).

Dec.8 - Dec.12: Wrap up the algorithm.Do the speed tests.Write the Report.


