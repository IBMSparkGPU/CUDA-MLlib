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

#include "../include/cuda_checking.h"
#include "../include/opt_cuda.h"


int getGPUCount(){

 int device_count=0;
 checkCudaErrors(cudaGetDeviceCount(&device_count));

 if (device_count < 0){
   device_count = 0;
 }

 return device_count;
}

void setGPUDevice(int id){
   checkCudaErrors(cudaSetDevice(id));
}

int getCurrentGPU(){
   int current_device=-1;

   checkCudaErrors(cudaGetDevice(&current_device));
   
   return current_device;
}


double *allocateDeviceMemory(int len){
 double *d_region = NULL;

// checkCudaErrors(mallocBest((void **)&d_region, len*sizeof(double)));
 checkCudaErrors(cudaMalloc((void **)&d_region, len*sizeof(double)));

 cudaDeviceSynchronize();
 cudaError_t error = cudaGetLastError();
 if(error != cudaSuccess)
  {
   // print the CUDA error message and exit
   printf("CUDA error in allocateDeviceMemory(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
   exit(0);
 }

 return d_region;
}

void freeDeviceMemory(double *region){
 if (region != NULL){
// checkCudaErrors(freeBest(region));
  checkCudaErrors(cudaFree(region));

 cudaDeviceSynchronize();
 cudaError_t error = cudaGetLastError();
 if(error != cudaSuccess)
  {
   // print the CUDA error message and exit
   printf("CUDA error in freeDeviceMemory(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
   exit(0);
 }

 }
}

void copyFromHostToDevice(double *d_region, double *h_region, int len){
 checkCudaErrors(cudaMemcpy(d_region, h_region, len*sizeof(double), cudaMemcpyHostToDevice));

 cudaDeviceSynchronize();
 cudaError_t error = cudaGetLastError();
 if(error != cudaSuccess)
  {
   // print the CUDA error message and exit
   printf("CUDA error in copyFromHostToDevice(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
   exit(0);
 }

}

void copyFromDeviceToHost(double *h_region, double *d_region, int len){
 checkCudaErrors(cudaMemcpy(h_region, d_region, len*sizeof(double), cudaMemcpyDeviceToHost));
 
 cudaDeviceSynchronize();
 cudaError_t error = cudaGetLastError();
 if(error != cudaSuccess)
  {
   // print the CUDA error message and exit
   printf("CUDA error in copyFromHDeviceToHost(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
   exit(0);
 }

}

void copyFromDeviceToDevice(double *dest, double *src, int len){
 checkCudaErrors(cudaMemcpy(dest, src, len*sizeof(double), cudaMemcpyDeviceToDevice));

 cudaDeviceSynchronize();
 cudaError_t error = cudaGetLastError();
 if(error != cudaSuccess)
  {
   // print the CUDA error message and exit
   printf("CUDA error in copyFromHDeviceToDevice(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
   exit(0);
 }


}

double *kernel_getZero()
{
  double *d_answer = NULL;
  double  h_answer = 0.0;
  d_answer = allocateDeviceMemory(1);
  copyFromHostToDevice(d_answer, &h_answer, 1);

  return d_answer;
}


double *kernel_getOne()
{
  double *d_answer = NULL;
  double  h_answer = 1.0;
  d_answer = allocateDeviceMemory(1);
  copyFromHostToDevice(d_answer, &h_answer, 1);
  return d_answer;
}




void kernel_matrixTimesVector(const double *matrixIn, int rows, int cols,
                              const double *vectorIn,          // cols
                              double       *vectorOut,         // rows
                                                          cublasHandle_t cublasHandle)
{

  double* deviceOne = kernel_getOne();
  double* deviceZero = kernel_getZero();

  checkCublasErrors(cublasDgemv(cublasHandle,
                           CUBLAS_OP_T,
                           cols, rows,
                           deviceOne,
                           matrixIn,  cols,
                           vectorIn,  1,
                           deviceZero,
                           vectorOut, 1));

  freeDeviceMemory(deviceOne);
  freeDeviceMemory(deviceZero);
}



// vectorOut = vectorIn x matrixIn
// All three are allocation in device memory
// The matrix is in row major order (while cublas assumes column major)

void kernel_vectorTimesMatrix(const double *vectorIn, // rows
                              const double *matrixIn, int rows, int cols,
                              double       *vectorOut, // cols
                                                          cublasHandle_t cublasHandle)
{

  double* deviceOne = kernel_getOne();
  double* deviceZero = kernel_getZero();

  // Reverse the order of multiplication bacause cublas assumes column-major order
  checkCublasErrors(cublasDgemv(cublasHandle,
                           CUBLAS_OP_N,
                           cols, rows,
                           deviceOne,
                           matrixIn,  cols,
                           vectorIn,  1,
                           deviceZero,
                           vectorOut, 1));

  freeDeviceMemory(deviceOne);
  freeDeviceMemory(deviceZero);


}

// ---------------- set_vector_to_zero ----------------------------

__global__ void kernel_set_vector_to_zero(double *d_vec, int dimension)
{

 int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < dimension){
    d_vec[gid] = 0;
  }
}

void cuda_set_vector_to_zero(double * d_vec, int dimension)
{
 int nBlocks=0, nThreads = 0;

 nThreads = 512;

  if (dimension < nThreads){
     nBlocks = 1;
     nThreads = dimension;
  }
  else{
     if ((dimension % nThreads) == 0){
         nBlocks = dimension/nThreads;
     }
     else{
         nBlocks = (dimension/nThreads)+1;
     }
  }

  kernel_set_vector_to_zero<<<nBlocks, nThreads>>>(d_vec, dimension);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_set_vector_to_zero(): %s GPU=%d \n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }

}

// ------------------------ function_to_be_minimized -----------------------------

__global__ void kernel_logit_term(double     * d_product,         // elements: N
                               double     * d_y,
                               int           N,
                               int           dimension)
{

  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;


  if (gid < N){

  double dp = d_product[gid];
  if (dp < 100.00) {
    d_product[gid] = log(1.0 + exp(dp)) - d_y[gid] * dp;
  } else {
    d_product[gid] =               dp   - d_y[gid] * dp;
  }
  }
}

__global__ void kernel_divide(double     * d_product,         // elements: N
                           double     * d_rv,
                           int           numIterations,
                           int           N)
{

  extern __shared__ double partial_sum[]; // one element per thread

  double sum = 0;

  for (int i = 0; i < numIterations; ++i) {
    int idx = i*blockDim.x + threadIdx.x;
    if (idx < N) sum += d_product[idx];
  }
  partial_sum[threadIdx.x] = sum;

  __syncthreads();

  if (threadIdx.x == 0) {
    for (int i = 1; i < blockDim.x; ++i) {
      sum += partial_sum[i];
    }
    *d_rv = sum/(double)N;
  }
}



__global__ void kernel_add_regularization_term(double     * d_input_vector,
                                            int           dimension,
                                            double       regularization_parameter,
                                            double     * d_rv)
{
  if (threadIdx.x == 0) {
    double sum = 0;
    for (int i = 1; i < dimension; ++i) {
      sum += 0.5 * d_input_vector[i] * d_input_vector[i] * regularization_parameter;
    }
    *d_rv += sum;
  }
}

double cuda_function_to_be_minimized(double     * d_input_vector,  // elements:     dimension
                                   double     * d_x,             // elements: N x dimension
                                   double     * d_y,             // elements: N
                                   double       regularization_parameter,
                                   int           N,
                                   int           dimension,
                                                                   cublasHandle_t cublasHandle)
{
  double   h_rv = 0; // returned value
  double * d_rv           = allocateDeviceMemory(1);
  double * d_product      = allocateDeviceMemory(N); 

  int nThreads = 512;
  int nBlocks = 0;
  int maxThreads = 1024;

  if (N < nThreads){
     nBlocks = 1;
     nThreads = N;
  }
  else{
     if ((N % nThreads) == 0){
         nBlocks = N/nThreads;
     }
     else{
         nBlocks = (N/nThreads)+1;
     }
  }

  kernel_matrixTimesVector(d_x, N, dimension, d_input_vector, d_product, cublasHandle);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_matrixTimesVector(): %s GPU=%d\\n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }



  kernel_logit_term<<<nBlocks, nThreads>>>(d_product, d_y, N, dimension);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_logit_term(): %s GPU=%d\\n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


  int threads = N;
  if (threads > maxThreads) threads = maxThreads;
  kernel_divide<<<1, threads, threads*sizeof(double)>>>(d_product, d_rv, (N+threads-1)/threads, N);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_divide(): %s GPU=%d\\n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }

  kernel_add_regularization_term<<<1, 1>>>(d_input_vector, dimension, regularization_parameter, d_rv);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_add_regularization_term(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


  copyFromDeviceToHost(&h_rv, d_rv, 1);

  freeDeviceMemory(d_rv);
  freeDeviceMemory(d_product);

  return h_rv;
}



// ------------------------ gradient_of_function_to_be_minimized -----------------------------

__global__ void kernel_logit_term_grad(double     * d_product,         // elements: N
                                    double     * d_y,
                                    double     * d_coef,
                                    int           N)
{
  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < N){

  double dp = d_product[gid];
  if (dp < 100.00) {
      d_coef[gid] = 1.0 - 1.0 / (1.0 + exp(dp)) - d_y[gid];
    } else {
      d_coef[gid] = 1.0                         - d_y[gid];
    }
 }
}



__global__ void kernel_add_regularization_term_grad(double     * d_output_gradient, // elements:    dimension
                                                 double     * d_input_vector,
                                                 int        dimension,
                                                 int           N,
                                                 double       regularization_parameter)
{

  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < dimension){

  d_output_gradient[gid] /= N;

  if (gid != 0) {
    d_output_gradient[gid] += d_input_vector[gid] * regularization_parameter;
  }
 }

}

void cuda_gradient_of_function_to_be_minimized(double     * d_output_gradient, // elements:     dimension
                                               double     * d_input_vector,    // elements:     dimension
                                               double     * d_x,               // elements: N x dimension
                                               double     * d_y,               // elements: N
                                               double     regularization_parameter,
                                               int        N,
                                               int        dimension,
                                               cublasHandle_t cublasHandle)
{
  int nBlocks = 0; 
  int nThreads = 0;

  double *d_product = allocateDeviceMemory(N);
  double *d_coef = allocateDeviceMemory(N);

  nThreads = 512; 

  if ((N % nThreads) == 0){
      nBlocks = N/nThreads;
  }
  else{
      nBlocks = (N/nThreads)+1;
  }

  kernel_matrixTimesVector(d_x, N, dimension, d_input_vector, d_product, cublasHandle);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_matrixTimesVector()::cuda_gradient_of_function_to_be_minimized(): %s GPU=%d\n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


  kernel_logit_term_grad<<<nBlocks, nThreads>>>(d_product, d_y, d_coef, N);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_logit_term_grad()::cuda_gradient_of_function_to_be_minimized: %s GPU=%d \n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


  kernel_vectorTimesMatrix(d_coef, d_x, N, dimension, d_output_gradient, cublasHandle);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_vectorTimesMatrix()::cuda_gradient_of_function_to_be_minimized: %s GPU=%d \n", cudaGetErrorString(error), getCurrentGPU());
    exit(-1);
  }

  if (dimension < nThreads){
     nBlocks = 1;
     nThreads = dimension;
  }
  else{
     if ((dimension % nThreads) == 0){
         nBlocks = dimension/nThreads;
     }
     else{
         nBlocks = (dimension/nThreads)+1;
     }
  }


  kernel_add_regularization_term_grad<<<nBlocks, nThreads>>>(d_output_gradient, d_input_vector, dimension, N, regularization_parameter);

  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_add_regularization_term_grad()::cuda_gradient_of_function_to_be_minimized: %s GPU=%d \n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


#if 0
  double *h_output_gradient = (double *) malloc(dimension * sizeof(double));

  cudaMemcpy(h_output_gradient, d_output_gradient, dimension*sizeof(double), cudaMemcpyDeviceToHost);

  int i=0;
  for(i=0; i < dimension; i++){
  printf("h_output_gradient[%d]=%15.10f \n", i, h_output_gradient[i]);
  }
#endif

 
  freeDeviceMemory(d_product);
  freeDeviceMemory(d_coef);
 
}

// -------------------- vec_equals_vec1_plus_alpha_times_vec2 --------------------

__global__ void kernel_vec_equals_vec1_plus_alpha_times_vec2(double      *vec,
                                                          double      *vec1,
                                                          double       alpha,
                                                          double      *d_a1,
                                                          double      *vec2,
                                                          int numElements)
{
  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < numElements){
  double a = alpha;
  if (d_a1) a *= *d_a1;

   vec[gid] = vec1[gid] + a * vec2[gid];
  }
}




void cuda_vec_equals_vec1_plus_alpha_times_vec2(double * d_vec,
                                              double * d_vec1,
                                              double   alpha,
                                              double * d_a1,
                                              double * d_vec2,
                                              int numElements)
{
  int nBlocks = 0;
  int nThreads = 0;

  nThreads = 512;

  if (numElements < nThreads){
     nBlocks = 1;
     nThreads = numElements;
  }
  else{
     if ((numElements % nThreads) == 0){
         nBlocks = numElements/nThreads;
     }
     else{
         nBlocks = (numElements/nThreads)+1;
     }
  }

  kernel_vec_equals_vec1_plus_alpha_times_vec2<<<nBlocks, nThreads>>>(d_vec, d_vec1, alpha, d_a1, d_vec2,numElements);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_vec_equals_vec1_plus_alpha_times_vec2(): %s GPU=%d \n", cudaGetErrorString(error), getCurrentGPU());
    exit(-1);
  }

}


// -------------------- vec_equals_alpha_times_vec2 --------------------

__global__ void kernel_vec_equals_minus_vec1(double      *vec,
                                          double      *vec1, 
                                          int numElements)
{

  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < numElements){
    vec[gid] = -vec1[gid];
  }
}

void cuda_vec_equals_minus_vec1(double * d_vec,
                              double * d_vec1,
                              int       numElements)
{

  int nBlocks = 0;
  int nThreads = 0;

  nThreads = 512;
  if (numElements < nThreads){
     nBlocks = 1;
     nThreads = numElements;
  }
  else{
     if ((numElements % nThreads) == 0){
         nBlocks = numElements/nThreads;
     }
     else{
         nBlocks = (numElements/nThreads)+1;
     }
  }

  kernel_vec_equals_minus_vec1<<<nBlocks, nThreads>>>(d_vec, d_vec1, numElements);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error in kernel_vec_equals_minus_vec1(): %s GPU=%d \n", cudaGetErrorString(error), getCurrentGPU());
    exit(-1);
  }

}

// -------------------- euclidean_norm -----------------------

__global__ void kernel_euclidean_norm(const double      *vec,
                                   int                 numElements,
                                   double            *answer)
{
  extern __shared__ double square[]; // one element per thread

  int i = threadIdx.x; // numElements assumed to fit into one block
  square[i] = vec[i] * vec[i];

  __syncthreads();

  if (i == 0) {
    double sum = 0;
    for (int j = 0; j < numElements; ++j) {
      sum += square[j];
    }
    *answer = sqrt(sum);
  }
}


double cuda_euclidean_norm(const double * d_vec,
                         int             numElements)
{
  // This code relies on numElements <= max threads allowed per block

  double  h_answer=0.0;

  double *d_answer = allocateDeviceMemory(1);

  kernel_euclidean_norm<<<1, numElements, numElements*sizeof(double)>>>(d_vec, numElements, d_answer);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit    
    printf("CUDA error in kernel_euclidian_norm(): %s GPU=%d \n", cudaGetErrorString(error),getCurrentGPU());
    exit(-1);
  }


  copyFromDeviceToHost(&h_answer, d_answer, 1);

  freeDeviceMemory(d_answer);

  return h_answer;
}


// -------------------- mult_vector_by_number --------------------

__global__ void kernel_mult_vector_by_number(double      *vec,
                                          double       alpha,
                                          int numElements)
{
  int iam = threadIdx.x;
  int bid = blockIdx.x;
  int threads_in_block = blockDim.x;
  int gid = bid*threads_in_block + iam;

  if (gid < numElements){
    vec[gid] *= alpha;
  }

}

void cuda_mult_vector_by_number(double * d_vec,
                              double alpha,
                              int       numElements)
{

  int nBlocks = 0;
  int nThreads = 0;

  nThreads = 512;

  if (numElements < nThreads){
     nBlocks = 1;
     nThreads = numElements;
  }
  else{
     if ((numElements % nThreads) == 0){
         nBlocks = numElements/nThreads;
     }
     else{
         nBlocks = (numElements/nThreads)+1;
     }
  }

  kernel_mult_vector_by_number<<<nBlocks, nThreads>>>(d_vec, alpha, numElements);

  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit    
    printf("CUDA error in kernel_mult_vector_by_number(): %s GPU=%d\n", cudaGetErrorString(error), getCurrentGPU());
    exit(-1);
  }


}

// -------------------- dot_product -----------------------

__global__ void kernel_dot_product(const double * vec1,
                                const double * vec2,
                                int             numElements,
                                double       * answer)
{
  extern __shared__ double products[]; // one element per thread

  int i = threadIdx.x; // numElements assumed to fit into one block
  products[i] = vec1[i] * vec2[i];

  __syncthreads();

  if (i == 0) {
    double sum = 0;
    for (int j = 0; j < numElements; ++j) {
      sum += products[j];
    }
    *answer = sum;
  }
}


void cuda_dot_product(const double * d_vec1,
                    const double * d_vec2,
                    double       * d_answer,
                    int             numElements,
                    cublasHandle_t cublasHandle)
{
       checkCublasErrors(cublasDdot(cublasHandle, numElements, d_vec1, 1, d_vec2, 1, d_answer));
}

