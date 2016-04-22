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
#include "org_apache_spark_ml_recommendation_CuMFJNIInterface.h"
#include "cuda/als.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../../utilities.h"

JNIEXPORT jobjectArray JNICALL Java_org_apache_spark_ml_recommendation_CuMFJNIInterface_doALSWithCSR
  (JNIEnv * env, jobject obj, jint m, jint n, jint f, jint nnz, jdouble lambda, jobjectArray sortedSrcFactors, jintArray csrRow, jintArray csrCol, jfloatArray csrVal){
	//checkCudaErrors(cudaSetDevice(1));
	//use multiple GPUs
	//select a GPU for *this* specific dataset
   int whichGPU = get_gpu();
   checkCudaErrors(cudaSetDevice(whichGPU));
   cudaStream_t cuda_stream;
   cudaStreamCreate(&cuda_stream);
   /* check correctness
    int csrRowlen = env->GetArrayLength(csrRow);
	int csrCollen = env->GetArrayLength(csrCol);
    int csrVallen = env->GetArrayLength(csrVal);
	assert(csrRowlen == m + 1);
	assert(csrCollen == nnz);
	assert(csrVallen == nnz);
	*/
	int* csrRowIndexHostPtr;
	int* csrColIndexHostPtr;
	float* csrValHostPtr;
	/*
	printf("csrRow of len %d: ", len);
    for (int i = 0; i < len; i++) {
	  printf("%d ", body[i]);
    }
	printf("\n");
	*/
	//calculate X from thetaT
	float* thetaTHost;
	cudacall(cudaMallocHost( (void** ) &thetaTHost, n * f * sizeof(thetaTHost[0])) );
	//to be returned
	float* XTHost;
	cudacall(cudaMallocHost( (void** ) &XTHost, m * f * sizeof(XTHost[0])) );
	
	int numSrcBlocks = env->GetArrayLength(sortedSrcFactors);
	//WARNING: ReleaseFloatArrayElements and DeleteLocalRef are important; 
	//Otherwise result is correct but performance is bad
	int index = 0;
	for(int i = 0; i < numSrcBlocks; i++){
		jobject factorsPerBlock = env->GetObjectArrayElement(sortedSrcFactors, i);
		int numFactors = env->GetArrayLength((jobjectArray)factorsPerBlock);
		for(int j = 0; j < numFactors; j++){
			jobject factor = env->GetObjectArrayElement((jobjectArray)factorsPerBlock, j);
			jfloat *factorfloat = (jfloat *) env->GetPrimitiveArrayCritical( (jfloatArray)factor, 0);	
			memcpy(thetaTHost + index*f, factorfloat, sizeof(float)*f);
			index ++;
			env->ReleasePrimitiveArrayCritical((jfloatArray)factor, factorfloat, 0);
			env->DeleteLocalRef(factor);
		}
		env->DeleteLocalRef(factorsPerBlock);
	}
	// get a pointer to the raw input data, pinning them in memory
	csrRowIndexHostPtr = (jint*) env->GetPrimitiveArrayCritical(csrRow, 0);
	csrColIndexHostPtr = (jint*) env->GetPrimitiveArrayCritical(csrCol, 0);
	csrValHostPtr = (jfloat*) env->GetPrimitiveArrayCritical(csrVal, 0);

	/*
	printf("thetaTHost of len %d: \n", n*f);
    for (int i = 0; i < n*f; i++) {
	  printf("%f ", thetaTHost[i]);
    }
	printf("\n");
	*/
	int * d_csrRowIndex = 0;
	int * d_csrColIndex = 0;
	float * d_csrVal = 0;

	cudacall(cudaMalloc((void** ) &d_csrRowIndex,(m + 1) * sizeof(float)));
	cudacall(cudaMalloc((void** ) &d_csrColIndex, nnz * sizeof(float)));
	cudacall(cudaMalloc((void** ) &d_csrVal, nnz * sizeof(float)));
	cudacall(cudaMemcpyAsync(d_csrRowIndex, csrRowIndexHostPtr,(size_t ) ((m + 1) * sizeof(float)), cudaMemcpyHostToDevice, cuda_stream));
	cudacall(cudaMemcpyAsync(d_csrColIndex, csrColIndexHostPtr,(size_t ) (nnz * sizeof(float)), cudaMemcpyHostToDevice, cuda_stream));
	cudacall(cudaMemcpyAsync(d_csrVal, csrValHostPtr,(size_t ) (nnz * sizeof(float)),cudaMemcpyHostToDevice, cuda_stream));
	cudaStreamSynchronize(cuda_stream);

	// un-pin the host arrays, as we're done with them
	env->ReleasePrimitiveArrayCritical(csrRow, csrRowIndexHostPtr, 0);
	env->ReleasePrimitiveArrayCritical(csrCol, csrColIndexHostPtr, 0);
	env->ReleasePrimitiveArrayCritical(csrVal, csrValHostPtr, 0);

	printf("\tdoALSWithCSR with m=%d,n=%d,f=%d,nnz=%d,lambda=%f \n.", m, n, f, nnz, lambda);
	try{
		doALSWithCSR(cuda_stream, d_csrRowIndex, d_csrColIndex, d_csrVal, thetaTHost, XTHost, m, n, f, nnz, lambda, 1);
	}
	catch (thrust::system_error &e) {
		printf("CUDA error during some_function: %s", e.what());
		
	}
	jclass floatArrayClass =  env->FindClass("[F");
	jobjectArray output = env->NewObjectArray(m, floatArrayClass, 0);
    for (int i = 0; i < m; i++) {
		jfloatArray col = env->NewFloatArray(f);
		env->SetFloatArrayRegion(col, 0, f, XTHost + i*f);
		env->SetObjectArrayElement(output, i, col);
		env->DeleteLocalRef(col);
    }
	cudaFreeHost(thetaTHost);
	cudaFreeHost(XTHost);
	//TODO: stream create and destroy expensive?
	checkCudaErrors(cudaStreamSynchronize(cuda_stream));
	checkCudaErrors(cudaStreamDestroy(cuda_stream));
	cudaCheckError();
	return output;
 }
 
 JNIEXPORT void JNICALL Java_org_apache_spark_ml_recommendation_CuMFJNIInterface_testjni
  (JNIEnv * env, jobject obj){
	  printf("*******in native code of testjni ...\n");
	  
  }
