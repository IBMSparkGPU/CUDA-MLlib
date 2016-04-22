/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *  
 *  http://www.apache.org/licenses/LICENSE-2.0
 * 
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include "math.h"

#include "../include/opt_cuda.h"
#include "../include/lbfgs.h"

void
initialize_from_file(const char *file_name, int32_t *n, int32_t *dim, double **deviceX, double **deviceY)
{
  FILE *  f = NULL;
  char    buff[256];
  int32_t i,j;
  int32_t   dimension = -1; 
  int32_t   N = -1;
  double    regularization_parameter = -888.888;


  /* --- BEGIN tunable parameters --- */

  regularization_parameter = 0.1;
  f = fopen(file_name, "r");

  /* ---  END  tunable parameters --- */

  if (f == NULL) {
    printf("Can not open input file. Exiting.\n");
    HALT
  }

  dimension = 0;
  while (TRUE) {
    if (fscanf(f, "%s", buff) != 1) {
      HALT
    }
    if ( (buff[0] == 'l') || (buff[0] == 'f') ) {
      dimension++;
    } else {
      break;
    }
  }

  i = 1;
  while (TRUE) {
    if (fscanf(f, "%s", buff) != 1) {
      break;
    }
    i++;
  }
  if (i % dimension != 0) HALT
  N = i / dimension;

  *n = N;
  *dim = dimension;

  printf("Starting the initializing process \n");

  double *givenY = (double *) malloc(sizeof(double)* N);
  double *givenX = (double *) malloc(sizeof(double)* N * dimension);

  rewind(f);
  for (i = 0; i < dimension; i++) {
    if (fscanf(f, "%s", buff) != 1) {
      HALT
    }
  }

  int ij = 0;
  for (i = 0; i < N; i++) {
    if (fscanf(f, "%lf", &(givenY[i])) != 1) {
      HALT
    }
    for (j = 0; j < dimension - 1; j++) {
      if (fscanf(f, "%lf", &(givenX[ij++])) != 1) HALT
    }
    givenX[ij++] = 1.0;
  }
  if (fscanf(f, "%s", buff) == 1) HALT

  fclose(f);

 printf("Allocating GPU memory \n");


// checkCudaErrors(cudaMalloc((void **)&device_Y, N*sizeof(double)));
// checkCudaErrors(cudaMalloc((void **)&device_X, N*dimension*sizeof(double)));

 *deviceY = allocateDeviceMemory(N);
 *deviceX = allocateDeviceMemory(N*dimension);

 printf("Copying the data \n");

// checkCudaErrors(cudaMemcpy(device_Y, givenY, N*sizeof(double), cudaMemcpyHostToDevice)); 
// checkCudaErrors(cudaMemcpy(device_X, givenX, N*dimension*sizeof(double), cudaMemcpyHostToDevice)); 

 copyFromHostToDevice(*deviceY, givenY, N);
 copyFromHostToDevice(*deviceX, givenX, N*dimension);

 free(givenX);
 free(givenY);

}


void
initialize_from_arrays(double *  givenY, 
                       double *  givenX, 
                       double ** deviceX,
                       double ** deviceY,
                       int32_t   givenDimension,
                       int32_t   givenN)
{
  char hostname[1024];
  hostname[1023] = '\0';
  //gethostname(hostname, 1023);

 // printf("\nRunning %s implementation of LBFGS on %s compiled on %s %s; features = %d, samples = %d\n\n", 
 //        getenv("SPARK_IMPLEMENTATION"), hostname, __DATE__, __TIME__, givenDimension, givenN);

 int dimension = givenDimension;
 int N = givenN;

 *deviceY = allocateDeviceMemory(N);
 copyFromHostToDevice(*deviceY, givenY, N);

 *deviceX = allocateDeviceMemory(N*dimension);
 copyFromHostToDevice(*deviceX, givenX, N*dimension);
}



double
function_to_be_minimized(double * input_vector, double *device_X, double *device_Y, double regularization_parameter, int N, int dimension, cublasHandle_t cublasHandle)
{
  double  rv;
                                          
  rv = cuda_function_to_be_minimized(input_vector, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);
  return rv;
}



void
gradient_of_function_to_be_minimized(double * output_gradient, double * input_vector, double *device_X, double *device_Y, double regularization_parameter, int N, int dimension, cublasHandle_t cublasHandle)
{
   cuda_gradient_of_function_to_be_minimized(output_gradient, input_vector, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);
}




