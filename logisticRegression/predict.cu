/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include "cuda_runtime.h"
#include "utilities.h"
#include <cublas_v2.h>

extern "C" {
#include "predict.h"
}

#include "predict_kernel.cuh"

static cublasHandle_t getCublasHandle() {
	static cublasHandle_t cublasHandle = 0;

	if (!cublasHandle) {
		cublasCreate(&cublasHandle);
		cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
	}

	return cublasHandle;
}

// This form accepts host buffers, and creates device buffers from them
void predictKernelHost(double *h_testData, double *h_model, double intercept,
		double *h_score, int numSamples, int numFeatures, cudaStream_t stream) {

	double *d_test = NULL, *d_model = NULL, *d_score = NULL;
	double *h_value = NULL;
	int i = 0;

	h_value = (double *) malloc(sizeof(double) * numSamples);

	for (i = 0; i < numSamples; i++) {
		h_value[i] = 0.0;
	}

	checkCudaErrors(mallocBest((void **) &d_score, sizeof(double) * numSamples));
	checkCudaErrors(mallocBest((void **) &d_test,sizeof(double) * numSamples * numFeatures));
	checkCudaErrors(mallocBest((void **) &d_model, sizeof(double) * numFeatures));

	checkCudaErrors(cudaMemcpyAsync(d_score, h_score, sizeof(double) * numSamples, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_test, h_testData, sizeof(double) * numSamples * numFeatures, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_model, h_model, sizeof(double) * numFeatures, cudaMemcpyHostToDevice, stream));

	checkCudaErrors(cudaStreamSynchronize(stream));
	// call the device form with the device pointers
	predictKernelDevice(d_test, d_model, intercept, d_score, numSamples, numFeatures, stream);

	// copy results back to the host
	checkCudaErrors(cudaMemcpyAsync(h_score, d_score, sizeof(double) * numSamples, cudaMemcpyDeviceToHost, stream));

	checkCudaErrors(cudaStreamSynchronize(stream));

	// free previously allocated device memory
	freeBest(d_test);
	freeBest(d_model);
	freeBest(d_score);
}

// This form accepts device buffer pointers, which must have been set up previously
void predictKernelDevice(double *d_test, double *d_model, double intercept, double *d_score, int numSamples, int numFeatures, cudaStream_t stream) {
	double *d_zero = NULL, *d_one = NULL, *d_dp = NULL;
	double h_const = 0.0;
	
	//printf("Hello from the predictKernel()!. NumSamples=%d NumFeatures=%d", numSamples, numFeatures);
	
	// allocate temporary buffers
	checkCudaErrors(mallocBest((void **) &d_dp, sizeof(double) * numSamples));
	checkCudaErrors(mallocBest((void **) &d_zero, sizeof(double)));
	checkCudaErrors(mallocBest((void **) &d_one, sizeof(double)));

	// copy constants to device
	checkCudaErrors(cudaMemcpyAsync(d_dp, d_score, sizeof(double) * numSamples, cudaMemcpyDeviceToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_zero, &h_const, sizeof(double), cudaMemcpyHostToDevice, stream));
	h_const = 1.0;
	checkCudaErrors(cudaMemcpyAsync(d_one, &h_const, sizeof(double), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	// tell cublas to use our stream
	cublasHandle_t handle = getCublasHandle();
	cublasSetStream(handle, stream);
	/* First do the dot-product */
	cublasDgemv(handle, CUBLAS_OP_T, numFeatures, numSamples, d_one, d_test, numFeatures, d_model, 1, d_zero, d_dp, 1);

	int threads = 256;
	int threadBlocks = ((numSamples % threads) == 0) ? (numSamples / threads) : (numSamples / threads) + 1;
	checkCudaErrors(cudaStreamSynchronize(stream));
	score_kernel<<<threadBlocks, threads, 0, stream>>>(numSamples, d_dp, intercept, d_score);

	// free buffers
	freeBest(d_zero);
	freeBest(d_one);
	freeBest(d_dp);
}
