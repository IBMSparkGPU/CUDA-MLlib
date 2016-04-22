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
#include <float.h> // DBL_MIN
#include <math.h>  // sqrt(...)
#include <pthread.h>
#include "../include/lbfgs.h"
#include "../include/opt_cuda.h"

typedef struct thread_data_t{
   int id;
   char *file;
} thread_data;


void *thread_function(void *ptr){

   thread_data *data = (thread_data *)ptr;
   int id = data->id;
   const char *filename= data->file;

   printf("Thread %d opening file %s\n", id, filename); 

   setGPUDevice(id);
   double minimum = lbfgs_from_file(filename); 

   printf("Thread %d LBFGS finished. GPU=%d minimum=%15.10f \n", id, getCurrentGPU(), minimum);
}


signed
main(
signed argc, char ** argv) {
  /*
  - Function `lbfgs(...)'
      implicitly takes as input three black box functions:

      1 of 3) dimension_of_domain(...),
      2 of 3) function_to_be_minimized(...),
      3 of 3) gradient_of_function_to_be_minimized(...).

      defined in file `func_to_be_minimized_and_its_grad.c'.

  - Function `lbfgs(...)' produces two pieces of output:

      1 of 2) `minimizing_vector', (Must be pre-allocated)
      2 of 2) `minimum'.
  */

  if (argc != 2) {
    printf("Requires 1 argument, not %d\n", argc-1);
    return 1;
  }


  thread_data *thread_param[2];
  pthread_t threads[2];

  int gpu_count = getGPUCount();
  printf("There are %d GPUs in the system \n", getGPUCount());
  printf("Opening file %s \n", argv[1]);

  for(int i=0; i < gpu_count; i++){
    thread_param[i] = (thread_data *) malloc(sizeof(thread_data));
    thread_param[i]->id = i;
    thread_param[i]->file = argv[1];

    if (pthread_create(&threads[i], NULL, thread_function, (void *) thread_param[i])){
       fprintf(stderr, "Error creating thread\n");
       return 1;
    }
  }

  for(int i=0; i < gpu_count; i++){
    if (pthread_join(threads[i], NULL)){
    fprintf(stderr, "Error joining thread\n");
    return 2;
  } 
  }



#if 0
  double minimum = lbfgs_from_file(argv[1]);

  printf("L-BFGS has finished.\n\n");
  printf("Minimum value = %15.10f\n\n", minimum);
#endif

  return 0;
}


