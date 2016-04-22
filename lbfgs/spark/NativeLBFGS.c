
#include "NativeLBFGS.h"
#include "include/lbfgs.h"
#include "include/opt_cuda.h"
#include "utilities.h"
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <pthread.h>
#include "cublas_v2.h"



static float getDeltams(struct timeval *start, struct timeval *end) 
{
  time_t startSec = start->tv_sec;
  suseconds_t startUSec = start->tv_usec;
  
  time_t endSec = end->tv_sec;
  suseconds_t endUSec = end->tv_usec;
  
  return ((endSec - startSec) * 1e6 + (endUSec - startUSec)) / 1e3;
}

// TEMPORARY workaround until we have mallocBest in place
static pthread_mutex_t gpuLocks[8] = {PTHREAD_MUTEX_INITIALIZER};

JNIEXPORT void JNICALL Java_org_apache_spark_mllib_optimization_NativeLBFGS_runNativeLBFGS
  (JNIEnv      *env, 
   jobject      obj, 
   jobjectArray javaDataAsArray, 
   jdouble      javaConvergenceTol,
   jdouble      javaRegParm,
   jdoubleArray javaMinimizingVector, 
   jdoubleArray javaLossHistoryArray,
   jint         javaLossHistoryArraySize,
   jint         javaMaxIterations)
{
  // select a GPU for *this* specific dataset
  // NOTE: The GPU code currently has problems with GPUs other than GPU0,
  // so we're hardcoding GPU0 here until that is resolved.
  int whichGPU = get_gpu();
  pthread_mutex_lock(&gpuLocks[whichGPU]);
  checkCudaErrors(cudaSetDevice(whichGPU));

  struct timeval start, end;
  gettimeofday(&start, NULL);
  
  int i=0;
  int numSamples = (*env)->GetArrayLength(env, javaDataAsArray);

  jobject javaSample = (*env)->GetObjectArrayElement(env, javaDataAsArray, i);
  int numFeatures = (*env)->GetArrayLength(env, (jdoubleArray) javaSample)-1;
  
  jdouble *XX = malloc(sizeof(double) * numSamples * numFeatures);
  jdouble *YY = malloc(sizeof(double) * numSamples);

  for (i=0; i<numSamples; i++) {
    javaSample = (*env)->GetObjectArrayElement(env, javaDataAsArray, i);
    jdouble *sample = (*env)->GetPrimitiveArrayCritical(env, (jdoubleArray) javaSample, 0);
    memcpy(XX + i*numFeatures, sample+1, sizeof(double)*numFeatures);
    (*env)->ReleasePrimitiveArrayCritical(env, javaSample, sample, 0);

    YY[i] = sample[0];
  }

  gettimeofday(&end, NULL);
  printf("JNI timing Get Arrays: %f ms \n", getDeltams(&start, &end));

  gettimeofday(&start, NULL);
  // copy data to GPU

  double *deviceX=NULL, *deviceY=NULL;
  jdouble *minimizingVector = (*env)->GetPrimitiveArrayCritical(env, javaMinimizingVector, 0);
  jdouble *lossHistoryArray = (*env)->GetPrimitiveArrayCritical(env, javaLossHistoryArray, 0);
  initialize_from_arrays(YY, XX, &deviceX, &deviceY, numFeatures, numSamples);
  (*env)->ReleasePrimitiveArrayCritical(env, javaMinimizingVector, minimizingVector, 0);
  (*env)->ReleasePrimitiveArrayCritical(env, javaLossHistoryArray, lossHistoryArray, 0);

  gettimeofday(&end, NULL);
  printf("JNI timing initialize_from_arrays: %f ms \n", getDeltams(&start, &end));

  // run the kernel
  gettimeofday(&start, NULL);
  double   r;
  lbfgs(minimizingVector, &r, javaConvergenceTol, javaMaxIterations, deviceX, deviceY, javaRegParm, numSamples, numFeatures, lossHistoryArray, javaLossHistoryArraySize);

  gettimeofday(&end, NULL);

  pthread_mutex_unlock(&gpuLocks[whichGPU]);
  printf("JNI lbfgs_from_arrays on GPU %i: %f ms \n", whichGPU, getDeltams(&start, &end));


  gettimeofday(&start, NULL);
                                
  free(XX);
  free(YY);

  freeDeviceMemory(deviceX);
  freeDeviceMemory(deviceY);
  
  gettimeofday(&end, NULL);
  printf("JNI cleanup: %f ms \n", getDeltams(&start, &end));
}
