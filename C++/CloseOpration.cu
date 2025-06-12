#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

__global__ static void erosion(unsigned char* d_src, unsigned char* d_dst, int len, int width, int height) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	int wth = (len - 1) / 2;

	if (i > 0 && i < width && j > 0 && j < height) {
		d_dst[3 * (j*width + i)] = d_src[3 * (j*width + i)];
		d_dst[3 * (j*width + i) + 1] = d_src[3 * (j*width + i) + 1];
		d_dst[3 * (j*width + i) + 2] = d_src[3 * (j*width + i) + 2];
	}

	if (i > wth && i < width - wth && j > wth && j < height - wth) {
		for (int w = -wth; w < wth + 1; w++) {
			for (int h = -wth; h < wth + 1; h++) {
				if (d_src[3 * ((j + h)*width + i + w)] < d_dst[3 * (j*width + i)])
					d_dst[3 * (j*width + i)] = d_src[3 * ((j + h)*width + i + w)];
				if (d_src[3 * ((j + h)*width + i + w) + 1] < d_dst[3 * (j*width + i) + 1])
					d_dst[3 * (j*width + i) + 1] = d_src[3 * ((j + h)*width + i + w) + 1];
				if (d_src[3 * ((j + h)*width + i + w) + 2] < d_dst[3 * (j*width + i) + 2])
					d_dst[3 * (j*width + i) + 2] = d_src[3 * ((j + h)*width + i + w) + 2];
			}
		}
	}
}

__global__ static void dilation(unsigned char* d_src, unsigned char* d_dst, int len, int width, int height) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	int wth = (len - 1) / 2;

	if (i > 0 && i < width && j > 0 && j < height) {
		//d_dst[j*width + i] = d_src[j*width + i];
		d_dst[3 * (j*width + i)] = d_src[3 * (j*width + i)];
		d_dst[3 * (j*width + i) + 1] = d_src[3 * (j*width + i) + 1];
		d_dst[3 * (j*width + i) + 2] = d_src[3 * (j*width + i) + 2];
	}

	if (i > wth && i < width - wth && j > wth && j < height - wth) {
		for (int w = -wth; w < wth + 1; w++) {
			for (int h = -wth; h < wth + 1; h++) {
				//if (d_src[(j + h)*width + i + w] > d_dst[j*width + i])
				//	d_dst[j*width + i] = d_src[(j + h)*width + i + w];
				if (d_src[3 * ((j + h)*width + i + w)] > d_dst[3 * (j*width + i)])
					d_dst[3 * (j*width + i)] = d_src[3 * ((j + h)*width + i + w)];
				if (d_src[3 * ((j + h)*width + i + w) + 1] > d_dst[3 * (j*width + i) + 1])
					d_dst[3 * (j*width + i) + 1] = d_src[3 * ((j + h)*width + i + w) + 1];
				if (d_src[3 * ((j + h)*width + i + w) + 2] > d_dst[3 * (j*width + i) + 2])
					d_dst[3 * (j*width + i) + 2] = d_src[3 * ((j + h)*width + i + w) + 2];
			}
		}
	}
}

extern "C" void preprocess(unsigned char* d_src, unsigned char* d_tmp1, unsigned char* d_dst, int w, int h) {
	dim3 dimBlock(16, 16);
	dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
	dilation << <dimGrid, dimBlock, 0 >> > (d_src, d_tmp1, 3, w, h);
	erosion << <dimGrid, dimBlock, 0 >> > (d_tmp1, d_dst, 3, w, h);
	
}

//extern "C"
//void CallErosion(uchar* d_src, uchar* d_dst, int len, int width, int height) {
//	cudaError_t err;
//	erosion << <Blocks, Threads >> > (d_src, d_dst, len, width, height);
//	err = cudaGetLastError();
//	if (cudaSeccess != err)
//		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
//}
//
//extern "C"
//void CallDilation(uchar* d_src, uchar* d_dst, int len, int width, int height) {
//	cudaError_t err;
//	dilation << <Blocks, Threads >> > (d_src, d_dst, len, width, height);
//	err = cudaGetLastError();
//	if (cudaSeccess != err)
//		fprintf(stderr, "Failed (error code %s)!\n", cudaGetErrorString(err));
//}