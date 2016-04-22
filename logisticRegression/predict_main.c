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

#include "predict.h"

#include <sys/time.h>

double mySecond()
{
        struct timeval tp;
        struct timezone tzp;

        gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


int 
main(int argc, char **argv){

int numFeatures = 606;
int numSamples = 43400;
double *testData=NULL, *model=NULL, *result=NULL;
int i=0;

  testData = (double *) malloc(sizeof(double)*numSamples*numFeatures);
  model = (double *) malloc(sizeof(double)*numFeatures);

  result = (double *) malloc(sizeof(double)*numSamples);

  for (i=0; i < numSamples*numFeatures; i++){
     testData[i] = 1.0;
  }

  for (i=0; i < numSamples; i++){
     result[i] = 0.0;
  }

  for (i=0; i < numFeatures; i++){
     model[i] = 1.0;
  }

  double intercept= 0.5;
  predictKernelHost(testData, model,intercept, result, numSamples, numFeatures, 0);

  /* Checking the dot product */
#if 0
  for(i=0; i < numSamples; i++){
     if (result[i] != (numFeatures*1.0)){
       printf("Dot product error. result=%lf \n", result[i]);
       return 0;
     }
  }

  printf("\n Dot product successfully computed \n"); 
#endif

}




