
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

void matmulCPU(float* a, float* b, float* r, int n);
__global__ void matmulGPU(float* a, float* b, float* r, int n);


int main(int argc, char* argv[]) {
	
	
	int N = 1024;
	printf("N : %d\n", N);

	//Total size
	size_t sz = sizeof(float) * N * N;
	
	LARGE_INTEGER freq, start, end;
	

	float *h_a = (float*)malloc(sz);
	float* h_b = (float*)malloc(sz);
	float* h_r = (float*)malloc(sz);
	srand(time(NULL));
	for (int i = 0; i < N * N; i++) {
		h_a[i] = (float)(rand() % 100);
		h_b[i] = (float)(rand() % 100);
		h_r[i] = 0.;
	}

	float *d_a, *d_b, *d_r;
	cudaMalloc((void**)&d_a, sz);
	cudaMalloc((void**)&d_b, sz);
	cudaMalloc((void**)&d_r, sz);
	float *h_result_global = (float*)malloc(sz);

	/***************
	CPU Matrix multiplication
	***************/
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);
	matmulCPU(h_a, h_b, h_r, N);
	QueryPerformanceCounter(&end);
	double diff = (float)(end.QuadPart - start.QuadPart) / (float)freq.QuadPart;
	printf("CPU elapsend time : %lf\n", diff);


	/************************
	GPU Matrix multiplication_global
	*************************/
	int threads_width = 16;
	int grid_width = N % threads_width ? N / threads_width + 1 : N / threads_width;

	dim3 dim_threads(threads_width, threads_width);
	dim3 dim_grid(grid_width, grid_width);
	QueryPerformanceCounter(&start);
	cudaMemcpy(d_a, h_a, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sz, cudaMemcpyHostToDevice);
	//kernel call
	matmulGPU << < dim_grid, dim_threads >> > (d_a, d_b, d_r, N);

	//wait until kernel is over
	cudaDeviceSynchronize();
	cudaMemcpy(h_result_global, d_r, sz, cudaMemcpyDeviceToHost);
	QueryPerformanceCounter(&end);
	diff = (float)(end.QuadPart - start.QuadPart) / (float)freq.QuadPart;
	printf("GPU elapsend time : %lf\n", diff);

	/**********************
	Verification
	***********************/
	for (int i = 0; i < N * N; i++) {
		if ((h_r[i] - h_result_global[i])/h_result_global[i]>0) {
			printf("failed at %d, h_result_global \n", i);
			printf("%lf\n",  h_result_global[i]);
			break;
		}
	}
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_r);
	free(h_result_global);
	free(h_a);
	free(h_b);
	free(h_r);
	return 0;
}
void matmulCPU(float *a, float *b, float *r, int n) {
	int i = 0, j = 0, x = 0;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			float sum = 0.f;
#pragma unroll 2
			for (x = 0; x < n; x++) {
				sum += a[j * n + x] * b[x * n + i];
			}
			r[j * n + i] = sum;
		}
	}
}

__global__ void matmulGPU(float* a, float* b, float* r, int n) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= n || y >= n)
		return;
	float sum = 0;
#pragma unroll 2
	for (int i = 0; i < n; i++) {
		sum += (a[y * n + i] * b[i * n + x]);
	}
	r[y * n + x] = sum;
}