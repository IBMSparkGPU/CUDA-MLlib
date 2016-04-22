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
#include <pthread.h>
#include "utilities.h"

struct GPUState {
	int deviceCount; // number of GPUs to use
	int deviceToUse; // GPU to use (round-robin)
	pthread_mutex_t initLock;
} gpustate = {-1,-1,PTHREAD_MUTEX_INITIALIZER};

// like cudaMalloc, but tried GPU, and if fails, mallocs on the host instead and registers
// memory allocated via this method should be freed with freeBest
cudaError_t mallocBest ( void **devPtr, size_t size ) {
	cudaError_t returnVal = cudaSuccess;
	if (cudaMalloc(devPtr, size) != cudaSuccess) {
		//fprintf(stderr, "Unable to malloc %i bytes on the device - using host memory\n", size);
		void* h_ptr;
		returnVal = cudaHostAlloc(&h_ptr, size, cudaHostAllocMapped | cudaHostAllocWriteCombined);
		if (returnVal == cudaSuccess) checkCudaErrors(cudaHostGetDevicePointer(devPtr, h_ptr, 0));
	}
	return returnVal;
}

// like cudaFree, but can also handle host pointers.  To be used with mallocBest
cudaError_t freeBest ( void *devPtr ) {
	cudaPointerAttributes attributes;
	checkCudaErrors(cudaPointerGetAttributes (&attributes, devPtr));
	//fprintf(stderr, "freeing memory of type %i\n", attributes.memoryType);
	if (attributes.memoryType == cudaMemoryTypeDevice)
		checkCudaErrors(cudaFree(devPtr));
	else
		checkCudaErrors(cudaFreeHost(devPtr));

	return cudaSuccess;
}

// returns which GPU to run on, or -1 if no GPUs are available
int get_gpu() {
	if (gpustate.deviceCount == 1)
		return 0; // return immediately for the common case of 1 GPU
	else if (gpustate.deviceCount > 1) { // multiple GPUs
		int newval, oldval;
		do {
			oldval = gpustate.deviceToUse;
			if (oldval == gpustate.deviceCount-1)
				newval = 0;
			else
				newval = oldval+1;
		} while (!__sync_bool_compare_and_swap(&gpustate.deviceToUse, oldval, newval));
	}
	else if (gpustate.deviceCount == -1) { // not yet initialized... run initialization
		pthread_mutex_lock(&gpustate.initLock);
		// check if another thread already completed initialization
		if (gpustate.deviceCount != -1) {
			pthread_mutex_unlock(&gpustate.initLock);
			return get_gpu();
		}
		// continue with initialization
		if (cudaGetDeviceCount(&gpustate.deviceCount)) {
			fprintf(stderr, "Cuda Error in GetDeviceCount: %s\n", cudaGetErrorString(cudaGetLastError()));
			gpustate.deviceCount = 0;
		}
		else if (gpustate.deviceCount <= 0)
			gpustate.deviceCount = 0;
		else
			gpustate.deviceToUse = 0;

		for (int deviceID=0; deviceID<gpustate.deviceCount; deviceID++) {
			cudaSetDevice(deviceID);
			cudaDeviceReset();
		}
		pthread_mutex_unlock(&gpustate.initLock);
	}

	return gpustate.deviceToUse;
}
