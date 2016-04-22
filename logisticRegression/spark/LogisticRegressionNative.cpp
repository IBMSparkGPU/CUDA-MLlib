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

#include <cuda_runtime.h>
#include <ctype.h>
#include <assert.h>

#include "org_apache_spark_mllib_classification_LogisticRegressionNative.h"
#include "utilities.h"
#include "predict.h"

JNIEXPORT jdouble JNICALL Java_org_apache_spark_mllib_classification_LogisticRegressionNative_predictPoint
    (JNIEnv *env, jobject obj, jdoubleArray data, jdoubleArray weights, jdouble intercept) {

	// the kernel is written to take multiple data sets and produce a set of results, but we're going
	// to run it as multiple parallel kernels, each producing a single result instead
	double *d_dataBuffer, *d_weightsBuffer, *d_score;
	int dataCount, dataLen, whichGPU;
	jdouble h_score, *h_dataBuffer, *h_weightsBuffer;
	cudaStream_t stream;

	// select a GPU for *this* specific dataset
	whichGPU = get_gpu();
	checkCudaErrors(cudaSetDevice(whichGPU));
	checkCudaErrors(cudaStreamCreate(&stream));

	// get a pointer to the raw input data, pinning them in memory
	dataCount = env->GetArrayLength(data);
	dataLen = dataCount*sizeof(double);
	assert(dataCount == env->GetArrayLength(weights));
	h_dataBuffer = (jdouble*) env->GetPrimitiveArrayCritical(data, 0);
	h_weightsBuffer = (jdouble*) env->GetPrimitiveArrayCritical(weights, 0);

	// copy input data to the GPU memory
	// TODO: It may be better to access host memory directly, skipping the copy.  Investigate.
	checkCudaErrors(mallocBest((void**)&d_dataBuffer, dataLen));
	checkCudaErrors(mallocBest((void**)&d_weightsBuffer, dataLen));
	checkCudaErrors(cudaMemcpyAsync(d_dataBuffer, h_dataBuffer, dataLen, cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemcpyAsync(d_weightsBuffer, h_weightsBuffer, dataLen, cudaMemcpyHostToDevice, stream));
	// synchronize before unpinning, and also because there is a device-device transfer in predictKernelDevice
	checkCudaErrors(cudaStreamSynchronize(stream));
	// un-pin the host arrays, as we're done with them
	env->ReleasePrimitiveArrayCritical(data, h_dataBuffer, 0);
	env->ReleasePrimitiveArrayCritical(weights, h_weightsBuffer, 0);

	// allocate storage for the result
	checkCudaErrors(mallocBest((void**)&d_score, sizeof(double)));

	// run the kernel, to produce a result
	predictKernelDevice(d_dataBuffer, d_weightsBuffer, intercept, d_score, 1, dataCount, stream);

	checkCudaErrors(cudaStreamSynchronize(stream));

	// copy result back to host
	checkCudaErrors(cudaMemcpyAsync(&h_score, d_score, sizeof(double), cudaMemcpyDeviceToHost, stream));

	checkCudaErrors(cudaStreamSynchronize(stream));

	// Free the GPU buffers
	checkCudaErrors(freeBest(d_dataBuffer));
	checkCudaErrors(freeBest(d_weightsBuffer));
	checkCudaErrors(freeBest(d_score));

	checkCudaErrors(cudaStreamDestroy(stream));

	return h_score;
}
