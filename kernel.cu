#include <iostream>
#include <fstream>
#include <stdio.h>
#include <Eigen/Core>
#include <ctime>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

// Tool for change float_to_float4
union float_float4 {
	float4 f4;
	struct {
		float x;
		float y;
		float z;
		float w;
	};
};

// Mat size
unsigned int Left_Row = 256;
unsigned int Left_Col = 256;
unsigned int Right_Row = 256;
unsigned int Right_Col = 256;

// MatTrans block
__constant__ const unsigned int Block_tile_t_x = 64;
__constant__ const unsigned int Block_tile_t_y = 64;
__constant__ const unsigned int Num_threads_transpose = 256;

// MatMul block and K 
__constant__ const unsigned int Block_tile_mm_x = 128;
__constant__ const unsigned int Block_tile_mm_y = 128;
__constant__ const unsigned int K_tile = 16;
__constant__ const unsigned int Num_threads_matmul = 256;
__constant__ const unsigned int Areg_num = 2;
__constant__ const unsigned int Breg_num = 2;

// Block_block
// For example: 
// block-128*128-float,thread-256
//      ⬇cut into
// 4*block_block-16*16-float4, thread-256
__constant__ const unsigned int Block_block_x = Block_tile_mm_x / Areg_num / 4;
__constant__ const unsigned int Block_block_y = Block_tile_mm_y / Breg_num / 4;

bool CPU = false;
bool GPU = false;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>MatrixFDD_Row;

// Transpose
__global__ void __launch_bounds__(256) MatTrans(float* A, float* B, unsigned int M, unsigned int N) {

	const unsigned int num_per_thread = Block_tile_t_x * Block_tile_t_y / blockDim.x;

	// Index for loading global to shared
	const unsigned int block_row = Block_tile_t_y / num_per_thread;
	const unsigned int g2s_tx = threadIdx.x / block_row;
	const unsigned int g2s_ty = threadIdx.x % block_row;
	const unsigned int g2s_locate_smem_y = g2s_ty * num_per_thread;
	const unsigned int g2s_locate_ori = blockIdx.x * Block_tile_t_x * N + blockIdx.y * Block_tile_t_y + g2s_tx * N + g2s_locate_smem_y;

	// Index for loading shared to global
	const unsigned int s2g_tx = threadIdx.x / Block_tile_t_y;
	const unsigned int s2g_ty = threadIdx.x % Block_tile_t_y;
	const unsigned int s2g_locate_smem_x = s2g_tx * num_per_thread;
	const unsigned int s2g_locate_tar = blockIdx.y * Block_tile_t_y * M + blockIdx.x * Block_tile_t_x + s2g_ty * M + s2g_locate_smem_x;

	__shared__ float smem[Block_tile_t_x][Block_tile_t_y];

	// Load global to shared
#pragma unroll
	for (int i = 0; i < num_per_thread; i++)
	{
		smem[g2s_tx][g2s_locate_smem_y + i] = A[g2s_locate_ori + i];
	}

	__syncthreads();

	// Load shared to global
#pragma unroll
	for (int i = 0; i < num_per_thread; i++)
	{
		B[s2g_locate_tar + i] = smem[s2g_locate_smem_x + i][s2g_ty];
	}

}

// Multiply
__global__ void __launch_bounds__(256) MatMul(float* A, float* B, float4* C, unsigned int M, unsigned int N, unsigned int K) {

	// BlockID
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	// Index for loading data from global to shared
	const unsigned int load_num_per_thread_A = Block_tile_mm_x * K_tile / Num_threads_matmul;
	const unsigned int load_g2s_A_x = threadIdx.x / (Block_tile_mm_x / load_num_per_thread_A);
	const unsigned int load_g2s_A_y = threadIdx.x % (Block_tile_mm_x / load_num_per_thread_A);
	const unsigned int load_num_per_thread_B = Block_tile_mm_y * K_tile / Num_threads_matmul;
	const unsigned int load_g2s_B_x = threadIdx.x / (Block_tile_mm_y / load_num_per_thread_B);
	const unsigned int load_g2s_B_y = threadIdx.x % (Block_tile_mm_y / load_num_per_thread_B);

	// Shared memory
	__shared__ float4 Asmem[K_tile][Block_tile_mm_x >> 2];
	__shared__ float4 Bsmem[K_tile][Block_tile_mm_y >> 2];

	// Tool for change 4f to f4
	float_float4 Tool;

	// Ptr
	float* A_ptr = A + bx * Block_tile_mm_x;
	float* B_ptr = B + by * Block_tile_mm_y;
	const unsigned int K_loop = K / K_tile;

	// Reg
	float Areg[Areg_num][4];
	float Breg[Breg_num][4];
	const unsigned int num_panel = Block_tile_mm_x * Block_tile_mm_y / 16 / Num_threads_matmul;
	float4 Panel[num_panel][4];
	const unsigned int Asmem_idx = threadIdx.x / Block_block_y;
	const unsigned int Bsmem_idx = threadIdx.x % Block_block_y;

	// Initialize Panel
#pragma unroll
	for (int InitPanel_i = 0; InitPanel_i < num_panel; InitPanel_i++) {
#pragma unroll
		for (int InitPanel_j = 0; InitPanel_j < 4; InitPanel_j++) {
			Panel[InitPanel_i][InitPanel_j] = make_float4(0.f, 0.f, 0.f, 0.f);
		}
	}

	// K-loop
	for (int k = 0; k < K_loop; k++, A_ptr += K_tile * M, B_ptr += K_tile * N) {

		// Load A to Asmem
		float* Atemp = A_ptr + load_g2s_A_x * M + load_g2s_A_y * load_num_per_thread_A;
#pragma unroll
		for (int A2Asem = 0; A2Asem < load_num_per_thread_A / 4; A2Asem += 1, Atemp += 4) {
			Tool.x = *Atemp;
			Tool.y = *(Atemp + 1);
			Tool.z = *(Atemp + 2);
			Tool.w = *(Atemp + 3);
			Asmem[load_g2s_A_x][load_g2s_A_y * (load_num_per_thread_A / 4) + A2Asem] = Tool.f4;
		}

		// Load B to Bsmem
		float* Btemp = B_ptr + load_g2s_B_x * N + load_g2s_B_y * load_num_per_thread_B;
#pragma unroll
		for (int B2Bsmem = 0; B2Bsmem < load_num_per_thread_B / 4; B2Bsmem += 1, Btemp += 4) {
			Tool.x = *Btemp;
			Tool.y = *(Btemp + 1);
			Tool.z = *(Btemp + 2);
			Tool.w = *(Btemp + 3);
			Bsmem[load_g2s_B_x][load_g2s_B_y * (load_num_per_thread_B / 4) + B2Bsmem] = Tool.f4;
		}

		__syncthreads();

		// Calculate
#pragma unroll
		for (int K_tile_row = 0; K_tile_row < K_tile; K_tile_row++) {

			// Load Asmem to Areg
#pragma unroll
			for (int Areg_idx = 0; Areg_idx < Areg_num; Areg_idx++) {
				Tool.f4 = Asmem[K_tile_row][Asmem_idx + Areg_idx * Block_block_x];
				Areg[Areg_idx][0] = Tool.x;
				Areg[Areg_idx][1] = Tool.y;
				Areg[Areg_idx][2] = Tool.z;
				Areg[Areg_idx][3] = Tool.w;
			}

			// Load Bsmem to Breg
#pragma unroll
			for (int Breg_idx = 0; Breg_idx < Breg_num; Breg_idx++) {
				Tool.f4 = Bsmem[K_tile_row][Bsmem_idx + Breg_idx * Block_block_x];
				Breg[Breg_idx][0] = Tool.x;
				Breg[Breg_idx][1] = Tool.y;
				Breg[Breg_idx][2] = Tool.z;
				Breg[Breg_idx][3] = Tool.w;
			}

			// Calculate Panel
			for (int Panel_id = 0; Panel_id < num_panel; Panel_id++) {
				const unsigned int Areg_idx = Panel_id / Breg_num;
				const unsigned int Breg_idx = Panel_id % Breg_num;
#pragma unroll
				for (int row = 0; row < 4; row++) {
					Panel[Panel_id][row].x += Areg[Areg_idx][row] * Breg[Breg_idx][0];
					Panel[Panel_id][row].y += Areg[Areg_idx][row] * Breg[Breg_idx][1];
					Panel[Panel_id][row].z += Areg[Areg_idx][row] * Breg[Breg_idx][2];
					Panel[Panel_id][row].w += Areg[Areg_idx][row] * Breg[Breg_idx][3];
				}
			}
		}
		__syncthreads();
	}

	// Write Panel to C
	for (int Panel_id = 0; Panel_id < num_panel; Panel_id++) {
		const unsigned int panel_x = Panel_id / Breg_num;
		const unsigned int panel_y = Panel_id % Breg_num;
		const unsigned int tx = threadIdx.x / (Block_tile_mm_y / Breg_num / 4);
		const unsigned int ty = threadIdx.x % (Block_tile_mm_y / Breg_num / 4);
		float4* C_begin_block = C + bx * Block_tile_mm_x * N / 4 + by * Block_tile_mm_y / 4;
		float4* C_begin_block_block = C_begin_block + panel_x * (Block_tile_mm_x / Areg_num) * N / 4 + panel_y * (Block_tile_mm_y / Breg_num / 4);
		float4* C_panel = C_begin_block_block + tx * N + ty;
#pragma unroll
		for (int row = 0; row < 4; row++) {
			*(C_panel + row * N / 4) = Panel[Panel_id][row];
		}
	}

}

int main() {

	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	//cout << prop.multiProcessorCount << endl;

	CPU = false;
	GPU = true;

	while (Left_Col <= 12800) {
		// Create matrices
		MatrixFDD_Row left = Eigen::MatrixXf::Random(Left_Row, Left_Col);
		MatrixFDD_Row left_trans = Eigen::MatrixXf::Zero(Left_Col, Left_Row);
		MatrixFDD_Row right = Eigen::MatrixXf::Random(Right_Row, Right_Col);
		MatrixFDD_Row answer_cpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);
		MatrixFDD_Row answer_gpu = Eigen::MatrixXf::Zero(Left_Row, Right_Col);

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
			float* left_trans_device;
			float* right_device;
			float4* answer_device;
			cudaMalloc((void**)&left_device, sizeof(float) * Left_Row * Left_Col);
			cudaMalloc((void**)&left_trans_device, sizeof(float) * Left_Col * Left_Row);
			cudaMalloc((void**)&right_device, sizeof(float) * Right_Row * Right_Col);
			cudaMalloc((void**)&answer_device, sizeof(float4) * Left_Row * Right_Col / 4);

			float* left_ptr = left.data();
			cudaMemcpy(left_device, left_ptr, sizeof(float) * Left_Row * Left_Col, cudaMemcpyHostToDevice);
			float* left_trans_ptr = left_trans.data();
			cudaMemcpy(left_trans_device, left_trans_ptr, sizeof(float) * Left_Row * Left_Col, cudaMemcpyHostToDevice);
			float* right_ptr = right.data();
			cudaMemcpy(right_device, right_ptr, sizeof(float) * Right_Row * Right_Col, cudaMemcpyHostToDevice);
			float* answer_ptr = answer_gpu.data();
			cudaMemcpy(answer_device, answer_ptr, sizeof(float) * Left_Row * Right_Col, cudaMemcpyHostToDevice);



			dim3 gridsize_trans((Left_Row + Block_tile_t_x - 1) / Block_tile_t_x, (Left_Col + Block_tile_t_y - 1) / Block_tile_t_y);
			dim3 gridsize_matmul((Left_Row + Block_tile_mm_x - 1) / Block_tile_mm_x, (Right_Col + Block_tile_mm_y - 1) / Block_tile_mm_y);
			cudaEventRecord(start);
			MatTrans << <gridsize_trans, Num_threads_transpose >> > (left_device, left_trans_device, Left_Row, Left_Col);
			MatMul << <gridsize_matmul, Num_threads_matmul >> > (left_trans_device, right_device, answer_device, Left_Row, Right_Col, Left_Col);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			for (int i = 0; i < Left_Row; i++)
			{
				cudaMemcpy(answer_gpu.data() + i * Right_Col, answer_device + i * Right_Col / 4, sizeof(float) * Right_Col, cudaMemcpyDeviceToHost);
			}

			float time;
			cudaEventElapsedTime(&time, start, stop);
			cout << "(" << Left_Row << "," << Left_Col << ")*(" << Right_Row << "," << Right_Col << ")  " << time << "ms" << endl;
			cudaFree(left_device);
			cudaFree(left_trans_device);
			cudaFree(right_device);
			cudaFree(answer_device);
		}
		Left_Row += 512;
		Left_Col += 512;
		Right_Row += 512;
		Right_Col += 512;
	}
	return 0;
}

