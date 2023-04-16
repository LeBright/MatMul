#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
using namespace std;

const unsigned int Left_Row = 453;
const unsigned int Left_Col =222;
const unsigned int Right_Row = 222;
const unsigned int Right_Col = 357;
bool CPU = false;
bool GPU = false;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>MatrixFDD;

__device__ int my_ceil(float A) {
	if (A > (int)A)
		return (int)A + 1;
	else
		return (int)A;
}
__global__ void MatMul(float* left, float* right, float* answer, unsigned int left_size, unsigned int left_stride, unsigned int right_size, unsigned int right_stride, unsigned int answer_size) {

	unsigned int block_num = my_ceil((float)right_stride / 32);
	unsigned int block_i = blockIdx.x / block_num;
	unsigned int block_j = blockIdx.x % block_num;

	unsigned int thread_i = threadIdx.x >> 5;
	unsigned int thread_j = threadIdx.x % 32u;

	float Cij = 0;
	for (int m = 0; m < my_ceil((float)left_stride / 32); m++){

		__shared__ float left_block[1024];
		__shared__ float right_block[1024];

		if (thread_i+(block_i<<5) < left_size / left_stride && thread_j+(m<<5) < left_stride) {
			left_block[threadIdx.x] = left[((block_i << 5) + thread_i) * left_stride + (m << 5) + thread_j];
		}
		else {
			left_block[threadIdx.x] = 0;
		}
		if (thread_i+(m<<5) < right_size / right_stride && thread_j+(block_j<<5) < right_stride) {
			right_block[threadIdx.x] = right[((m << 5) + thread_i) * right_stride + (block_j << 5) + thread_j];
		}
		else {
			right_block[threadIdx.x] = 0;
		}
		__syncthreads();

#pragma unroll
		for (int k = 0; k < 32; k++)
		{
			Cij += left_block[(thread_i << 5) + k] * right_block[(k << 5) + thread_j];
		}
		__syncthreads();
	}
	if(thread_i+(block_i<<5) < answer_size / right_stride && thread_j+(block_j<<5) < right_stride)
		answer[((block_i << 5) + thread_i) * right_stride + (block_j << 5) + thread_j] = Cij;
}

int main(){

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	CPU = false;
	GPU = true;

	// Create matrices
	MatrixFDD left = Eigen::MatrixXf::Random(Left_Row, Left_Col);
	MatrixFDD right = Eigen::MatrixXf::Random(Right_Row, Right_Col);
	MatrixFDD answer_cpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);
	MatrixFDD answer_gpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);
	cout << "The matrices have been created!" << endl;

	// CPU
	if (CPU) {
		clock_t start_CPU = clock();

		for (int i = 0; i < Left_Row; i++) {
			for (int j = 0; j < Right_Col; j++) {
				float temp = 0;
				for (int k = 0; k < Left_Col; k++) {
					temp += left(i, k) * right(k, j);
				}
				answer_cpu(i, j) = temp;
			}
		}

		clock_t end_CPU = clock();
		cout <<"CPU time: " << end_CPU - start_CPU << endl;
	}

	// GPU
	if (GPU) {
		clock_t start_GPU = clock();

		float* left_device;
		float* right_device;
		float* answer_device;
		cudaMalloc((void**)&left_device, sizeof(float) * Left_Row * Left_Col);
		cudaMalloc((void**)&right_device, sizeof(float) * Right_Row * Right_Col);
		cudaMalloc((void**)&answer_device, sizeof(float) * Left_Row * Right_Col);

		float* left_ptr = left.data();
		cudaMemcpy(left_device, left_ptr, sizeof(float) * Left_Row * Left_Col, cudaMemcpyHostToDevice);
		float* right_ptr = right.data();
		cudaMemcpy(right_device, right_ptr, sizeof(float) * Right_Row * Right_Col, cudaMemcpyHostToDevice);
		float* answer_ptr = answer_gpu.data();
		cudaMemcpy(answer_device, answer_ptr, sizeof(float) * Left_Row * Right_Col, cudaMemcpyHostToDevice);

		int gridsize = ceil((float)Left_Row / 32) * ceil((float)Right_Col / 32);
		MatMul << <gridsize, 1024,8192 >> > (left_device, right_device, answer_device, Left_Row * Left_Col, Left_Col, Right_Row * Right_Col, Right_Col,Left_Row*Right_Col);
		for (int i = 0; i < Left_Row; i++)
		{
			cudaMemcpy(answer_gpu.data() + i * Right_Col, answer_device + i * Right_Col, sizeof(float) * Right_Col, cudaMemcpyDeviceToHost);
		}

		clock_t end_GPU = clock();
		cout << "GPU time: " << end_GPU - start_GPU << endl;
	}
	cout << answer_gpu.isApprox(left * right) << endl;

	return 0;
}

