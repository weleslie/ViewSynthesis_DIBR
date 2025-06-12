#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

__global__ static void backwarp(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int width, int height, float delta) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    int count = j * width;
    int dis = (int)floor((disp[(count + i)] * -delta));

    if (i > 0 && i < width && j > 0 && j < height && i + dis >= 0 && i + dis < width) {
        d_dst[(count + i) * 3] = d_src[(count + i + dis) * 3];
        d_dst[(count + i) * 3 + 1] = d_src[(count + i + dis) * 3 + 1];
        d_dst[(count + i) * 3 + 2] = d_src[(count + i + dis) * 3 + 2];
    }
}

extern "C" void BackWarp(unsigned char* d_src, signed char* disp, unsigned char* d_dst, int w, int h, float delta) {
	dim3 dimBlock(32, 32);
	dim3 dimGrid((w + dimBlock.x - 1) / dimBlock.x, (h + dimBlock.y - 1) / dimBlock.y);
	backwarp << <dimGrid, dimBlock, 0 >> > (d_src, disp, d_dst, w, h, delta);
}
