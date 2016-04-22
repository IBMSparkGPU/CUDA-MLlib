/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LBFGS_H
#define LBFGS_H
#include "cublas_v2.h"
#ifdef __cplusplus
#include <stdlib.h>
#include <stdint.h>

#define HALT {printf("HALT: \""__FILE__"\", line %d.\n", __LINE__); exit(-1);}
#define FALSE (0!=0)
#define TRUE (1!=0)

#define DEBUG



/* The following four functions must be provided by user. */

double
function_to_be_minimized(
double * input_vector,
double *device_X, double *device_Y,
double regulerization_parameter, int N, int dimension,
cublasHandle_t cublasHandle);

void
gradient_of_function_to_be_minimized(
double * output_gradient, double * input_vector,
double *device_X, double *device_Y,
double regulerization_parameter, int N, int dimension,
cublasHandle_t cublasHandle);

extern void
initialize_from_file(const char *file_name, int *n, int *dim, double **deviceX, double **deviceY);


/* Minimization by LBFGS algorithm */

extern double lbfgs_from_file(const char *file_name);

extern "C"
#endif

extern
#ifdef __cplusplus
"C"
#endif
void initialize_from_arrays(double *  givenY,
                       double *  givenX,
                       double ** deviceX,
                       double ** deviceY,
                       int   givenDimension,
                       int   givenN);

#ifdef __cplusplus
extern "C"
#endif
void lbfgs(double * minimizing_vector,
      double * minimum, 
      double  convergenceTol,
      int      maxIterations,
      double * device_X,
      double * device_Y,
      double   regulerization_parameter,
      int      N,
      int      dimension,
      double * loss_history_array,
      int      loss_history_array_size);

extern double lbfgs_from_file(const char *file_name);

#ifdef __cplusplus
extern "C" 
#endif
double lbfgs_from_arrays(double *Y,
                         double *X,
                         double *YX,
                         double  convergenceTol,
                         double  regularization_parameter,
                         double *minimizing_vector,
                         double *loss_history_array,
                         int     loss_history_array_size,
                         int     numSamples,
                         int     numFeatures,
                         int     maxIterations);
#endif
