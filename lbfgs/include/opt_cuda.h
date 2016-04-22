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

#ifndef _KERNELS_H_
#define _KERNELS_H_

#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include "utilities.h"

double *allocateDeviceMemory(int len);
void copyFromHostToDevice(double *h_region, double *d_region, int len);
void copyFromDeviceToHost(double *h_region, double *d_region, int len);
void copyFromDeviceToDevice(double *dest, double *src, int len);

#ifdef __cplusplus
extern "C"
#endif
void freeDeviceMemory(double *region);


extern void   cuda_set_vector_to_zero(double * h_vec, int n);
extern void   cuda_vec_equals_vec1_plus_alpha_times_vec2(double * h_vec,
                                                       double * h_vec1,
                                                       double   alpha,
                                                       double * a1,
                                                       double * h_vec2,
                                                       int       numElements);

extern void   cuda_matrix_times_vector(const double *h_matrixIn, int rows, int cols,
                                   const double *h_vectorIn,
                                   double       *h_vectorOut);

extern void   cuda_mult_vector_by_number(double * h_vec,
                                       double   alpha,
                                       int       numElements);

extern void   cuda_vec_equals_minus_vec1(double * h_vec,
                                       double * h_vec1,
                                       int       numElements);
extern double cuda_euclidean_norm(const double * h_vec, int numElements) ;
extern void   cuda_dot_product(const double * h_vec1,
                             const double * h_vec2,
                             double       * d_answer,
                             int             numElements,
                             cublasHandle_t cublasHandle) ;

extern double cuda_function_to_be_minimized(double     * h_input_vector,
                                          double     * x,
                                          double     * y,
                                          double       regularization_parameter,
                                          int           N,
                                          int           dimension,
                                          cublasHandle_t cublasHandle);

extern void   cuda_gradient_of_function_to_be_minimized(double     * h_output_gradient,
                                                      double     * h_input_vector,
                                                      double     * x,
                                                      double     * y,
                                                      double       regularization_parameter,
                                                      int           N,
                                                      int           dimension,
                                                      cublasHandle_t cublasHandle);
extern int getGPUCount();
extern void setGPUDevice(int id);
extern int getCurrentGPU();

#endif

