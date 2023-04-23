#include <iostream>
#include <stdio.h>
#include <Eigen/Core>
#include <ctime>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <fstream>

using namespace std;

unsigned int Left_Row = 128;
unsigned int Left_Col = 128;
unsigned int Right_Row = 128;
unsigned int Right_Col = 128;
bool CPU = false;
bool GPU = false;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>MatrixFDD_Col;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>MatrixFDD_Row;

__global__ void MatMul(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {

	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	const unsigned int tx_gmem2smem = threadIdx.x >> 3;
	const unsigned int ty_gmem2smem = threadIdx.x % 8u;
	unsigned int offset_gmem2smem[4];
	offset_gmem2smem[0] = (tx_gmem2smem << 2) * K + ty_gmem2smem;
	offset_gmem2smem[1] = offset_gmem2smem[0] + K;
	offset_gmem2smem[2] = offset_gmem2smem[1] + K;
	offset_gmem2smem[3] = offset_gmem2smem[2] + K;

	const unsigned int tx_smem2reg = threadIdx.x >> 4;
	const unsigned int ty_smem2reg = threadIdx.x % 16u;

	float* A_begin = A + (bx << 7) * K;
	float* B_begin = B + (by << 7);
	float* A_end = A_begin + K;
	float* C_begin = C + (bx << 7) + (by << 7);

	float4 A_reg[2] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };
	float4 B_reg[2] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };
	float4 panel_1[4] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };
	float4 panel_2[4] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };
	float4 panel_3[4] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };
	float4 panel_4[4] = { {make_float4(0.f, 0.f, 0.f, 0.f)} };

	for (float* A_ptr = A_begin, *B_ptr = B_begin; A_ptr != A_end; A_ptr += 8, B_ptr += N << 3) {
		__shared__ float4 Asmem[8][32];
		__shared__ float4 Bsmem[8][32];

		Asmem[ty_gmem2smem][tx_gmem2smem].x = *(A_ptr + offset_gmem2smem[0]);
		Asmem[ty_gmem2smem][tx_gmem2smem].y = *(A_ptr + offset_gmem2smem[1]);
		Asmem[ty_gmem2smem][tx_gmem2smem].z = *(A_ptr + offset_gmem2smem[2]);
		Asmem[ty_gmem2smem][tx_gmem2smem].w = *(A_ptr + offset_gmem2smem[3]);

		Bsmem[ty_gmem2smem][tx_gmem2smem] = *((float4*)(B_ptr + ty_gmem2smem * N + (tx_gmem2smem << 2)));

		__syncthreads();
#pragma unroll
		for (int i = 0; i < 8; i++) {
			A_reg[0] = *(&Asmem[i][0] + tx_smem2reg);
			A_reg[1] = *(&Asmem[i][0] + tx_smem2reg + 16);
			B_reg[0] = *(&Bsmem[i][0] + ty_smem2reg);
			B_reg[1] = *(&Bsmem[i][0] + ty_smem2reg + 16);

			panel_1[0].x += A_reg[0].x * B_reg[0].x;
			panel_1[0].y += A_reg[0].x * B_reg[0].y;
			panel_1[0].z += A_reg[0].x * B_reg[0].z;
			panel_1[0].w += A_reg[0].x * B_reg[0].w;
			panel_1[1].x += A_reg[0].y * B_reg[0].x;
			panel_1[1].y += A_reg[0].y * B_reg[0].y;
			panel_1[1].z += A_reg[0].y * B_reg[0].z;
			panel_1[1].w += A_reg[0].y * B_reg[0].w;
			panel_1[2].x += A_reg[0].z * B_reg[0].x;
			panel_1[2].y += A_reg[0].z * B_reg[0].y;
			panel_1[2].z += A_reg[0].z * B_reg[0].z;
			panel_1[2].w += A_reg[0].z * B_reg[0].w;
			panel_1[3].x += A_reg[0].w * B_reg[0].x;
			panel_1[3].y += A_reg[0].w * B_reg[0].y;
			panel_1[3].z += A_reg[0].w * B_reg[0].z;
			panel_1[3].w += A_reg[0].w * B_reg[0].w;

			panel_2[0].x += A_reg[0].x * B_reg[1].x;
			panel_2[0].y += A_reg[0].x * B_reg[1].y;
			panel_2[0].z += A_reg[0].x * B_reg[1].z;
			panel_2[0].w += A_reg[0].x * B_reg[1].w;
			panel_2[1].x += A_reg[0].y * B_reg[1].x;
			panel_2[1].y += A_reg[0].y * B_reg[1].y;
			panel_2[1].z += A_reg[0].y * B_reg[1].z;
			panel_2[1].w += A_reg[0].y * B_reg[1].w;
			panel_2[2].x += A_reg[0].z * B_reg[1].x;
			panel_2[2].y += A_reg[0].z * B_reg[1].y;
			panel_2[2].z += A_reg[0].z * B_reg[1].z;
			panel_2[2].w += A_reg[0].z * B_reg[1].w;
			panel_2[3].x += A_reg[0].w * B_reg[1].x;
			panel_2[3].y += A_reg[0].w * B_reg[1].y;
			panel_2[3].z += A_reg[0].w * B_reg[1].z;
			panel_2[3].w += A_reg[0].w * B_reg[1].w;

			panel_3[0].x += A_reg[1].x * B_reg[0].x;
			panel_3[0].y += A_reg[1].x * B_reg[0].y;
			panel_3[0].z += A_reg[1].x * B_reg[0].z;
			panel_3[0].w += A_reg[1].x * B_reg[0].w;
			panel_3[1].x += A_reg[1].y * B_reg[0].x;
			panel_3[1].y += A_reg[1].y * B_reg[0].y;
			panel_3[1].z += A_reg[1].y * B_reg[0].z;
			panel_3[1].w += A_reg[1].y * B_reg[0].w;
			panel_3[2].x += A_reg[1].z * B_reg[0].x;
			panel_3[2].y += A_reg[1].z * B_reg[0].y;
			panel_3[2].z += A_reg[1].z * B_reg[0].z;
			panel_3[2].w += A_reg[1].z * B_reg[0].w;
			panel_3[3].x += A_reg[1].w * B_reg[0].x;
			panel_3[3].y += A_reg[1].w * B_reg[0].y;
			panel_3[3].z += A_reg[1].w * B_reg[0].z;
			panel_3[3].w += A_reg[1].w * B_reg[0].w;

			panel_4[0].x += A_reg[1].x * B_reg[1].x;
			panel_4[0].y += A_reg[1].x * B_reg[1].y;
			panel_4[0].z += A_reg[1].x * B_reg[1].z;
			panel_4[0].w += A_reg[1].x * B_reg[1].w;
			panel_4[1].x += A_reg[1].y * B_reg[1].x;
			panel_4[1].y += A_reg[1].y * B_reg[1].y;
			panel_4[1].z += A_reg[1].y * B_reg[1].z;
			panel_4[1].w += A_reg[1].y * B_reg[1].w;
			panel_4[2].x += A_reg[1].z * B_reg[1].x;
			panel_4[2].y += A_reg[1].z * B_reg[1].y;
			panel_4[2].z += A_reg[1].z * B_reg[1].z;
			panel_4[2].w += A_reg[1].z * B_reg[1].w;
			panel_4[3].x += A_reg[1].w * B_reg[1].x;
			panel_4[3].y += A_reg[1].w * B_reg[1].y;
			panel_4[3].z += A_reg[1].w * B_reg[1].z;
			panel_4[3].w += A_reg[1].w * B_reg[1].w;

		}
	}
	__syncthreads();
	float* C_begin1 = C_begin + (tx_smem2reg << 2) * N + (ty_smem2reg << 2);
	float* C_begin2 = C_begin1 + 64;
	float* C_begin3 = C_begin + ((tx_smem2reg << 2) + 64) * N + (ty_smem2reg << 2);
	float* C_begin4 = C_begin3 + 64;

	*C_begin1 = panel_1[0].x;
	*(C_begin1 + 1) = panel_1[0].y;
	*(C_begin1 + 2) = panel_1[0].z;
	*(C_begin1 + 3) = panel_1[0].w;
	*(C_begin1 + N) = panel_1[1].x;
	*(C_begin1 + N + 1) = panel_1[1].y;
	*(C_begin1 + N + 2) = panel_1[1].z;
	*(C_begin1 + N + 3) = panel_1[1].w;
	*(C_begin1 + 2 * N) = panel_1[2].x;
	*(C_begin1 + 2 * N + 1) = panel_1[2].y;
	*(C_begin1 + 2 * N + 2) = panel_1[2].z;
	*(C_begin1 + 2 * N + 3) = panel_1[2].w;
	*(C_begin1 + 3 * N) = panel_1[3].x;
	*(C_begin1 + 3 * N + 1) = panel_1[3].y;
	*(C_begin1 + 3 * N + 2) = panel_1[3].z;
	*(C_begin1 + 3 * N + 3) = panel_1[3].w;


	*C_begin2 = panel_2[0].x;
	*(C_begin2 + 1) = panel_2[0].y;
	*(C_begin2 + 2) = panel_2[0].z;
	*(C_begin2 + 3) = panel_2[0].w;
	*(C_begin2 + N) = panel_2[1].x;
	*(C_begin2 + N + 1) = panel_2[1].y;
	*(C_begin2 + N + 2) = panel_2[1].z;
	*(C_begin2 + N + 3) = panel_2[1].w;
	*(C_begin2 + 2 * N) = panel_2[2].x;
	*(C_begin2 + 2 * N + 1) = panel_2[2].y;
	*(C_begin2 + 2 * N + 2) = panel_2[2].z;
	*(C_begin2 + 2 * N + 3) = panel_2[2].w;
	*(C_begin2 + 3 * N) = panel_2[3].x;
	*(C_begin2 + 3 * N + 1) = panel_2[3].y;
	*(C_begin2 + 3 * N + 2) = panel_2[3].z;
	*(C_begin2 + 3 * N + 3) = panel_2[3].w;

	*C_begin3 = panel_3[0].x;
	*(C_begin3 + 1) = panel_3[0].y;
	*(C_begin3 + 2) = panel_3[0].z;
	*(C_begin3 + 3) = panel_3[0].w;
	*(C_begin3 + N) = panel_3[1].x;
	*(C_begin3 + N + 1) = panel_3[1].y;
	*(C_begin3 + N + 2) = panel_3[1].z;
	*(C_begin3 + N + 3) = panel_3[1].w;
	*(C_begin3 + 2 * N) = panel_3[2].x;
	*(C_begin3 + 2 * N + 1) = panel_3[2].y;
	*(C_begin3 + 2 * N + 2) = panel_3[2].z;
	*(C_begin3 + 2 * N + 3) = panel_3[2].w;
	*(C_begin3 + 3 * N) = panel_3[3].x;
	*(C_begin3 + 3 * N + 1) = panel_3[3].y;
	*(C_begin3 + 3 * N + 2) = panel_3[3].z;
	*(C_begin3 + 3 * N + 3) = panel_3[3].w;

	*C_begin4 = panel_4[0].x;
	*(C_begin4 + 1) = panel_4[0].y;
	*(C_begin4 + 2) = panel_4[0].z;
	*(C_begin4 + 3) = panel_4[0].w;
	*(C_begin4 + N) = panel_4[1].x;
	*(C_begin4 + N + 1) = panel_4[1].y;
	*(C_begin4 + N + 2) = panel_4[1].z;
	*(C_begin4 + N + 3) = panel_4[1].w;
	*(C_begin4 + 2 * N) = panel_4[2].x;
	*(C_begin4 + 2 * N + 1) = panel_4[2].y;
	*(C_begin4 + 2 * N + 2) = panel_4[2].z;
	*(C_begin4 + 2 * N + 3) = panel_4[2].w;
	*(C_begin4 + 3 * N) = panel_4[3].x;
	*(C_begin4 + 3 * N + 1) = panel_4[3].y;
	*(C_begin4 + 3 * N + 2) = panel_4[3].z;
	*(C_begin4 + 3 * N + 3) = panel_4[3].w;

}

int main() {

	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	//cout << prop.sharedMemPerBlock << endl;

	CPU = false;
	GPU = true;
	ofstream ofile("report.txt");
	for (int i = 0; i < 8; i++) {
		// Create matrices
		MatrixFDD_Row left = Eigen::MatrixXf::Random(Left_Row, Left_Col);
		MatrixFDD_Row right = Eigen::MatrixXf::Random(Right_Row, Right_Col);
		MatrixFDD_Row answer_cpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);
		MatrixFDD_Row answer_gpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);
		cout << "The matrices have been created!" << endl;

		// CPU
		if (CPU) {
			DWORD start_CPU = GetTickCount();

			for (int i = 0; i < Left_Row; i++) {
				for (int j = 0; j < Right_Col; j++) {
					float temp = 0;
					for (int k = 0; k < Left_Col; k++) {
						temp += left(i, k) * right(k, j);
					}
					answer_cpu(i, j) = temp;
				}
			}

			DWORD end_CPU = GetTickCount();
			cout << "CPU time: " << end_CPU - start_CPU << endl;
		}

		// GPU
		if (GPU) {

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

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

			dim3 gridsize((Left_Row + 127) / 128, (Right_Col + 127) / 128);

			cudaEventRecord(start);
			MatMul << <gridsize, 256 >> > (left_device, right_device, answer_device, Left_Row, Right_Col, Left_Col);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			for (int i = 0; i < Left_Row; i++)
			{
				cudaMemcpy(answer_gpu.data() + i * Right_Col, answer_device + i * Right_Col, sizeof(float) * Right_Col, cudaMemcpyDeviceToHost);
			}

			float time;
			cudaEventElapsedTime(&time, start, stop);
			cout << "it: " << i << endl;

			ofile << "(" << Left_Row << ", " << Left_Col << ") * (" << Right_Row << ", " << Right_Col << ") gpu time: " << time << " ms" << endl;
		}
		Left_Row *= 2;
		Left_Col *= 2;
		Right_Row *= 2;
		Right_Col *= 2;
	}
	return 0;
}

