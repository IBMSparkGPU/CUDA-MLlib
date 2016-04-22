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


#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <float.h> 
#include <math.h> 

#include "../include/opt_cuda.h"
#include "../include/cuda_checking.h"
#include "../include/lbfgs.h"



/* --- BEGIN Local functions definitions --- */


void
set_vector_to_zero(double * vec, int numElements) 
{
  cuda_set_vector_to_zero(vec, numElements);
}

void
mult_vector_by_number(
double * vec, double number, int dimension) {
   cuda_mult_vector_by_number(vec, number, dimension);
}

void
vec_equals_minus_vec1(
double * vec, double * vec1, int dimension) {
  cuda_vec_equals_minus_vec1(vec, vec1, dimension);
}

void
vec_equals_vec1_plus_alpha_times_vec2(double * vec, 
                                      double * vec1, 
                                      double   alpha, double *a1,
                                      double * vec2, int dimension) {
   cuda_vec_equals_vec1_plus_alpha_times_vec2(vec, vec1, alpha, a1, vec2, dimension);
}

void
vec_equals_vec1_minus_vec2(
double * vec, double * vec1, double * vec2, int dimension) {
  vec_equals_vec1_plus_alpha_times_vec2(vec, vec1, -1.0, NULL, vec2, dimension);
}

void
vec_equals_vec1_plus_vec2(
double * vec, double * vec1, double * vec2, int dimension) {
  vec_equals_vec1_plus_alpha_times_vec2(vec, vec1, 1.0, NULL, vec2, dimension);
}

double *
dot_product(
const double * vec1, const double * vec2, int dimension, cublasHandle_t cublasHandle)
{   
  double *x = NULL;
  x= allocateDeviceMemory(1);
  cuda_dot_product(vec1, vec2, x, dimension, cublasHandle);

  return x;
}


double
euclidean_norm(
double * vec, int dimension) {                                    
  return cuda_euclidean_norm(vec, dimension);
}



static void
print_vector(
double * vec, int dimension) {
  int32_t i;
  for (i = 0; i < dimension; i++) {
    printf("_[%d]: %15.10f\n", i, vec[i]);
  }
}

/* ---  END  Local functions definitions --- */

void
lbfgs(double * minimizing_vector_result, 
      double * minimum, 
      double   convergenceTol,
      int      maxIterations,
      double * device_X,
      double * device_Y,
      double   regularization_parameter,
      int      N,
      int      dimension, 
      double * loss_history_array,
      int      loss_history_array_size){
  // Note: all the variables used in the alg. are local to this function.
  // Also all the algorithm code is contained in this function, that is
  // such procedures as line search etc. are not separate functions.
        
  /* --- BEGIN Tunable parameters --- */

  const int32_t history_length = 20; // Use inform. from this many last iter.
  const double  do_not_update_hessian_if_abs_inv_rho_is_less = 1.0e-11;
  #define       NUM_LS_STEPS 40
  const double  line_search_steps[NUM_LS_STEPS] = {
    /* [ 0 ..  4]: */  1.0e-9, 2.0e-9, 4.0e-9,  8.0e-9, 1.6e-8,
    /* [ 5 ..  9]: */  3.2e-8, 6.4e-8, 1.25e-7, 2.5e-7, 5.0e-7,
    /* [10 .. 14]: */  1.0e-6, 2.0e-6, 4.0e-6,  8.0e-6, 1.6e-5,
    /* [15 .. 19]: */  3.2e-5, 6.4e-5, 1.25e-4, 2.5e-4, 5.0e-4,
    /* [20 .. 24]: */  1.0e-3, 2.0e-3, 4.0e-3,  8.0e-3, 1.6e-2,
    /* [25 .. 29]: */  3.2e-2, 6.4e-2, 1.25e-1, 2.5e-1, 5.0e-1,
    /* [30 .. 34]: */  1.0,    2.0,    4.0,     8.0,    1.6e+1,
    /* [34 .. 39]: */  3.2e+1, 6.4e+1, 1.25e+2, 2.5e+2, 5.0e+2
  };
                                         
  cublasHandle_t cublasHandle;
  checkCublasErrors(cublasCreate(&cublasHandle));
  cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);

//  printf("Invoking lbfgs() on gpu %d\n", getCurrentGPU());
                                     
  int32_t tolerance; // This is index to the array above.
  for (tolerance = 0; tolerance < NUM_LS_STEPS; ++tolerance) {
    if (line_search_steps[tolerance] >= convergenceTol) break;
  }
  if (tolerance >= NUM_LS_STEPS) tolerance = NUM_LS_STEPS-1;
                                         

  /* ---  END  Tunable parameters --- */
              

  /* --- BEGIN Workspace (if desired can be safely reset after each iter. --- */
                                         
  int32_t  history_first = 0; // the first history step
  int32_t  i=0;
  int32_t  j=0;
  int32_t  line_search_step_index=0;
  double   tmp=0.0;
  double * tmp_ptr=NULL;

  double * direction_of_descend = allocateDeviceMemory(dimension);
  double * minimizer_candidate  = allocateDeviceMemory(dimension);
  double * gradient             = allocateDeviceMemory(dimension);
  double * tmp_vec              = allocateDeviceMemory(dimension);
                                         
  checkCudaErrors(cudaMemset(direction_of_descend,0, dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(minimizer_candidate,0, dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(gradient,0, dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(tmp_vec,0, dimension*sizeof(double)));


  /* ---  END  Workspace (if desired can be safely reset after each iter. --- */

  /* --- BEGIN These variables must retain values between iterations. --- */

  int       is_first_iter = TRUE;
  int32_t   iter_number=0;

  int32_t   new_line_search_step_index=0;
  double    new_minimum=0.0;
  double *  minimizing_vector     = allocateDeviceMemory(dimension);
  double *  new_minimizing_vector = allocateDeviceMemory(dimension);
  double *  new_gradient          = allocateDeviceMemory(dimension);

  checkCudaErrors(cudaMemset(minimizing_vector,0, dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(new_minimizing_vector,0, dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(new_gradient,0, dimension*sizeof(double)));

  double * s = allocateDeviceMemory(history_length * dimension);
  double * y = allocateDeviceMemory(history_length * dimension);


  checkCudaErrors(cudaMemset(s,0, history_length*dimension*sizeof(double)));
  checkCudaErrors(cudaMemset(y,0, history_length*dimension*sizeof(double)));

  double *  rho =                           // Array[history_length] of double.
    (double *) malloc(sizeof(double) * history_length);
  double *loss_history_queue  =             // Array[history_length] of double
    (double *) malloc(sizeof(double) * loss_history_array_size);

                                         {
    int32_t i;
    for (i = history_length - 1; i >= 0;  i--) {
      rho[i] = 0.0;
      loss_history_queue[i] = 0.0;
    }
  }

   /* set_vector_to_zero(rho, history_length);**********************************************************************************/
  /* ---  END  These variables must retain values between iterations. --- */

  // Initial point will be set inside of the loop by using `is_first_iter' var.
  // It does not take too much runtime but makes it visible what the role of
  // `minimizing_vector', `new_minimizing_vector' etc. is.
  // The termination criterion is checked inside of the loop.
  // Terminate if can not find a new point with a smaller value of obj. func.
  // There are two possibilities for termination:
  // 1. Adjusted gradient is zero. 2. Line search fails.
  // Hence two break statements inside.
  // The code ends after end of this loop (except for memory deallocation).
  while (TRUE) {

    if (is_first_iter) {
      is_first_iter = FALSE;
      set_vector_to_zero(minimizing_vector, dimension);
      gradient_of_function_to_be_minimized(gradient, minimizing_vector, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);
      *minimum = DBL_MAX;
      line_search_step_index = tolerance;
      iter_number = 1;
    } else {
      copyFromDeviceToDevice(minimizing_vector, new_minimizing_vector, dimension);
      copyFromDeviceToDevice(gradient, new_gradient, dimension);
      *minimum = new_minimum;
      line_search_step_index = new_line_search_step_index;
      iter_number++;
    }
    loss_history_queue[(iter_number-1) % loss_history_array_size] = *minimum;

    /* --- BEGIN Find direction of descend. --- */

    // First set `direction_of_descend' to minus gradient.
    vec_equals_minus_vec1(direction_of_descend, gradient, dimension);

    double *h_output = (double *) malloc(dimension * sizeof(double));

    // Multiply `direction_of_descend' by approx. inv. hessian given implicitly.

    set_vector_to_zero(tmp_vec, dimension);
    for (i = history_length - 1; i >= 0; i--) {
      int32_t h = (i + history_first) % history_length; // index into the circular queues s, y, rho
      double *sh = &(s[h*dimension]);
      double *yh = &(y[h*dimension]);
                                               
      tmp_ptr = dot_product(sh, direction_of_descend, dimension, cublasHandle);
      vec_equals_vec1_plus_alpha_times_vec2(tmp_vec, tmp_vec, rho[h], tmp_ptr, sh, dimension);  
      vec_equals_vec1_plus_alpha_times_vec2(direction_of_descend, direction_of_descend, -rho[h], tmp_ptr, yh, dimension);
    }

    for (i = 0; i < history_length; i++) {
      int32_t h = (i + history_first) % history_length; // index into the circular queues s, y, rho    
      double *sh = &(s[h*dimension]);   
      double *yh = &(y[h*dimension]);        
                                          
      tmp_ptr = dot_product(yh, direction_of_descend, dimension, cublasHandle);
      vec_equals_vec1_plus_alpha_times_vec2(direction_of_descend, direction_of_descend, -rho[h], tmp_ptr, sh, dimension);
    }
    vec_equals_vec1_plus_vec2(
      direction_of_descend, direction_of_descend, tmp_vec, dimension);

    // Normalize
    tmp = euclidean_norm(direction_of_descend, dimension);
    if (tmp <= DBL_MIN) break; // This is unlikely to happen. But if so  <------  tolerance??
                               // finish descend and output result.
    mult_vector_by_number(direction_of_descend, 1.0 / tmp, dimension);

    /* ---  END  Find direction of descend. --- */

#if 0
    cudaMemcpy(h_output, direction_of_descend, dimension*sizeof(double), cudaMemcpyDeviceToHost);

    for(i=0; i < dimension; i++){
           printf("3 h_output[%d]=%15.10f \n", i, h_output[i]);
    }
    printf("\n");
#endif


    /* --- BEGIN Line search. --- */

    line_search_step_index -= 1; // Alternative heuristic `-= 2' or `-= 3'.
    if (line_search_step_index < tolerance) {
      line_search_step_index = tolerance;
    }

    vec_equals_vec1_plus_alpha_times_vec2(minimizer_candidate, minimizing_vector,
                                          line_search_steps[line_search_step_index], NULL, direction_of_descend,dimension);              
    tmp = function_to_be_minimized(minimizer_candidate, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);

    if (tmp < *minimum) {
      // The 1st step guess gives improvement. Incr. step while keep improving.
      new_minimum = tmp;
      copyFromDeviceToDevice(new_minimizing_vector, minimizer_candidate, dimension);
      new_line_search_step_index = line_search_step_index;
      while (TRUE) {
        line_search_step_index++;
        if (line_search_step_index >= NUM_LS_STEPS) {
          break; // Maximum step has been reached and is the most beneficial.
        } else {
          vec_equals_vec1_plus_alpha_times_vec2(minimizer_candidate, minimizing_vector,
                                                line_search_steps[line_search_step_index], NULL, direction_of_descend,dimension);              
          tmp = function_to_be_minimized(minimizer_candidate, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);
          if (tmp < new_minimum) {
            new_minimum = tmp;
            copyFromDeviceToDevice(new_minimizing_vector, minimizer_candidate, dimension);
            new_line_search_step_index = line_search_step_index;
          } else {
            // The last increase in step was not beneficial.
            // Do not accept this step, stay with the result of the previous.
            break;
          }
        }
      }
    } else {
      // The 1st step guess does not give improvement.
      // Decrease step until get improvent or get below tolerance.
      while (TRUE) {
        line_search_step_index -= 1;
        if (line_search_step_index >= tolerance) {
          vec_equals_vec1_plus_alpha_times_vec2(minimizer_candidate, minimizing_vector,
                                                line_search_steps[line_search_step_index], NULL, direction_of_descend,dimension);              
          
          tmp = function_to_be_minimized(minimizer_candidate, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);
          if (tmp < *minimum) {
            new_minimum = tmp;
            copyFromDeviceToDevice(new_minimizing_vector, minimizer_candidate, dimension);
            new_line_search_step_index = line_search_step_index;
            // Line search succeeded.
            // line_search_step_index >= tolerance, so the outer big loop
            break;
          }
        } else {
          // Line search failed. This is the main termination point of the
          // whole algorithm. (There is another one which is very unlikely
          // to happen: when norm(direction_of_descend) == 0)
          // Ops, need to break two loops now.
          break;
          // And line_search_step_index remains < tolerance. See 3 lines below.
        }
      }
      if (line_search_step_index < tolerance) {
        break; // Line search failed. Algorithm finished.
      }
      // Line search succeeded.
    }
    // Line search succeeded.
    // `new_minimum', `new_minimizing_vector', and
    // `new_line_search_step_index' have meaningful values.

    gradient_of_function_to_be_minimized(new_gradient, new_minimizing_vector, device_X, device_Y, regularization_parameter, N, dimension, cublasHandle);

#if 0
    printf("new_minimum=%15.10f new_line_search_step_index=%d \n", new_minimum, new_line_search_step_index);
    cudaMemcpy(h_output, new_gradient, dimension*sizeof(double), cudaMemcpyDeviceToHost);

    for(i=0; i < dimension; i++){
           printf("3 h_output[%d]=%15.10f \n", i, h_output[i]);
    }
    printf("\n");
#endif

    /* ---  END  Line search. --- */



    /* --- BEGIN Update history with the new step info. --- */

    // History is a pipe. _[0] drops off
    // and _[history_length - 1] becomes available for the new record.
    // Move only pointers not the whole arrays.
    history_first++;
    int32_t history_last_index = (history_first + history_length - 1)%history_length;
    vec_equals_vec1_minus_vec2(s + history_last_index*dimension, new_minimizing_vector, minimizing_vector,dimension);
    vec_equals_vec1_minus_vec2(y + history_last_index*dimension, new_gradient, gradient,dimension);
    // Set tmp to (1 / rho).
    tmp_ptr = dot_product(s + history_last_index*dimension, 
                          y + history_last_index*dimension, dimension,
						  cublasHandle);
    double h_dot_product;

    copyFromDeviceToHost(&h_dot_product, tmp_ptr, 1);    
                
    if ((h_dot_product <   do_not_update_hessian_if_abs_inv_rho_is_less)
        &&
        (h_dot_product > (-do_not_update_hessian_if_abs_inv_rho_is_less))) {
      // Overwrite the new history entry
      // with a neutral (not affecting hessian) one.
      // Note that if we hit a degeneracy history entries will become
      // neutral one by one and finally we will move off from
      // the degenerate point by a regular gradient descent step.
      set_vector_to_zero(s + history_last_index*dimension, dimension);
      set_vector_to_zero(y + history_last_index*dimension, dimension);
      rho[history_last_index] = 0.0;
    } else {
      // `s' and `y' are already set, it remains to set `rho'.
      rho[history_last_index] = 1.0 / h_dot_product;
    }

    /* ---  END  Update history with the new step info. --- */



    // This is the end of the body of the loop.
    // Continue to the next iteration unless we want to print debug info. first

    int trace_progress_wanted = 0;

     if (trace_progress_wanted){
      if ( (iter_number & 0x0 /*0x3FF*/) == 0) {
        printf("Iteration %d:\n", iter_number);
        printf("Old minimum value = %15.10f\n", *minimum);
        printf("\n");
        // printf("Old minimizing vector:\n");
        // printf("\n");
        // print_vector(minimizing_vector);
        // printf("\n");
        printf(
               "Step length[%d] = %12.9f\n",
               new_line_search_step_index, line_search_steps[new_line_search_step_index]
               );
        // printf("\n");
        // print_history(s, y, rho, history_length);
        printf("\n");
      }
    }

    if (iter_number > maxIterations) break;

  } // End of loop.


  int h = 0;                                           // runs over loss_history_array
  int r = (int) iter_number - loss_history_array_size; // runs over loss_history_queue starting with first recorded minimum
  for (h = 0; h < loss_history_array_size; ++h, ++r) {
    loss_history_array[h] = (r < 0) ? 0 : loss_history_queue[r % loss_history_array_size];
  }
 
  copyFromDeviceToHost(minimizing_vector_result, minimizing_vector, dimension); 
                                         
  free(loss_history_queue);
  freeDeviceMemory(y);
  freeDeviceMemory(s);
  free(rho);
  freeDeviceMemory(new_gradient);
  freeDeviceMemory(new_minimizing_vector);
  freeDeviceMemory(minimizing_vector);
  freeDeviceMemory(tmp_vec);
  freeDeviceMemory(gradient);
  freeDeviceMemory(minimizer_candidate);
  freeDeviceMemory(direction_of_descend);

}


double lbfgs_from_file(const char *file_name)
{

  int32_t   dimension = -1;      
  int32_t   N = -1;
  double    regularization_parameter = -888.888;
  double    *device_X, *device_Y;

  initialize_from_file(file_name,&N,&dimension,&device_X,&device_Y);

  double   minimum;
  double * minimizing_vector =
    (double *) malloc(sizeof(double) * dimension);

  const int loss_history_array_size = 20;
  double    loss_history_array[loss_history_array_size];

  printf("Invoking lbfgs() on gpu %d\n", getCurrentGPU());

  regularization_parameter = 0.1;
  lbfgs(minimizing_vector, &minimum, 0.0001, 1000, device_X, device_Y, regularization_parameter, N, dimension, loss_history_array, loss_history_array_size);

#if 0
    int i;
    printf("\nMINIMIZING VECTOR:\n", minimum);
    for (i = 0; i < dimension; ++i) {
      printf("%f\n", minimizing_vector[i]);
    }

    printf("\nLOSS HISTORY: %f\n", minimum);
    for (i = 0; i < loss_history_array_size; ++i) {
      printf("%f \n", loss_history_array[i]);
    }
#endif


  free(minimizing_vector);
  freeDeviceMemory(device_X);
  freeDeviceMemory(device_Y);
  

  return minimum;
}


double lbfgs_from_arrays(double       *Y,                       // numSamples
                         double       *X,                       // numSamples x numFeatures
                         double       *YX,
                         double        convergenceTol,
                         double        regularization_parameter,
                         double       *minimizing_vector,       //              numFeatures 
                         double       *loss_history_array,
                         int           loss_history_array_size,
                         int           numSamples,
                         int           numFeatures,
                         int           maxIterations)
{
  int i;
  double   minimum;
  double   *device_X, *device_Y;

  initialize_from_arrays(Y, X, &device_X, &device_Y, numFeatures, numSamples);

  lbfgs(minimizing_vector, &minimum, convergenceTol, maxIterations, device_X, device_Y, regularization_parameter, numSamples, numFeatures, loss_history_array, loss_history_array_size);

  freeDeviceMemory(device_X);
  freeDeviceMemory(device_Y);

  return minimum;
}
