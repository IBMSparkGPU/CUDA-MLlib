#include "NativeCuAcc.h"

#include <string.h>
#include <iostream>
#include <string>

#include "cuAcc_cluster.h"
#include "cuAcc_updater.h"


JNIEXPORT jlong JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_create_1updater
  (JNIEnv *env, jobject, jstring name, jint option)
{
  CSystemInfo tm(__FUNCTION__);

  //CSystemInfo::log("NativeSGD-INFO] CUDA enabled on %s pid=%d\n",CSystemInfo::get_host_info(),getpid());
  //tm.proc_info();

  char* _name = (char*)env->GetStringUTFChars(name,0);

  CSystemInfo::mutex_lock();

  int device  = CCuAcc::get_most_idle_device();

  CUDA_CHECK(cudaSetDevice(device));   
  CUpdater*  pUpdater  = NULL;
  if(!strcmp(_name,"SquaredL2Updater"))   pUpdater = new CL2AdaDeltaUpdater();
  else if(!strcmp(_name,"L1Updater"))     pUpdater = new CL1AdaDeltaUpdater;
  else if(!strcmp(_name,"SimpleUpdater")) pUpdater = new CSimpleUpdater;
  else
  {
    CSystemInfo::log("ERROR: %s is not supported\n",_name);
    assert(false);
  }

  CSystemInfo::mutex_unlock();

  env->ReleaseStringUTFChars(name,_name);

  return  (jlong)pUpdater;
}

JNIEXPORT void JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_destroy_1updater
  (JNIEnv *, jobject, jlong handle)
{
  delete (CUpdater*)handle;
}

JNIEXPORT jdouble JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_updater_1initialize
  (JNIEnv *env, jobject, jlong handle, jdoubleArray x, jdouble lambda, jint option)
{
  CUpdater*  pUpdater  = (CUpdater*)handle;  
  assert(pUpdater);

  double* _x      = (double*)env->GetPrimitiveArrayCritical(x,0);
  double  regVal  = pUpdater->initialize(env->GetArrayLength(x),_x,lambda); 

  env->ReleasePrimitiveArrayCritical(x, _x, 0);  
  return  regVal;
}

JNIEXPORT jdouble JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_updater_1convergence_1compute
  (JNIEnv *env, jobject, jlong handle, jdoubleArray x, jdoubleArray cgrad, jlong mini_batch_size, jdouble step_size, jint iter, jdouble lambda, jdoubleArray convergence_info, jint option)  
{
  CSystemInfo tm(__FUNCTION__);

  CUpdater*  pUpdater  = (CUpdater*)handle;  
  assert(pUpdater);

  double* _x                = (double*)env->GetPrimitiveArrayCritical(x,0);
  double* _g                = (double*)env->GetPrimitiveArrayCritical(cgrad,0); 
  double* _convergence_info = (double*)env->GetPrimitiveArrayCritical(convergence_info,0); 

  tm.timer_check();

  //cudaHostRegister(_x,pUpdater->m_nx*sizeof(double),0);
  //cudaHostRegister(_g,pUpdater->m_nx*sizeof(double),0); 

  double  regVal  = pUpdater->update(_x,_g,mini_batch_size,step_size,iter,lambda,_convergence_info);  

  //cudaHostUnregister(_x);
  //cudaHostUnregister(_g);

  tm.timer_check();

  env->ReleasePrimitiveArrayCritical(convergence_info, _convergence_info, 0);
  env->ReleasePrimitiveArrayCritical(cgrad, _g, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(x, _x, JNI_ABORT); 

  return  regVal;   
}


JNIEXPORT jlong JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_create_1acc_1cluster
  (JNIEnv *env, jobject, jstring name, jdoubleArray data, jdoubleArray y, jint option)
{
  CSystemInfo tm(__FUNCTION__);

  unsigned  ny = env->GetArrayLength(y); 
  unsigned  nx = env->GetArrayLength(data)/ny;

  assert(ny);
  assert(nx);

  char*   _name   = (char*)env->GetStringUTFChars(name,0);
  double* _data   = (double*)env->GetPrimitiveArrayCritical(data,0);
  double* _y      = (double*)env->GetPrimitiveArrayCritical(y,0);


  tm.timer_check();

  CMachineLearning::ALGO algo;
  if(!strcmp(_name,"LinearRegression"))         algo  = CMachineLearning::LINEAR_REGRESSION;
  else if(!strcmp(_name,"LogisticRegression"))  algo  = CMachineLearning::LOGISTIC_REGRESSION;
  else
  {
    CSystemInfo::log("ERROR: %s is not supported\n",_name);
    assert(false);
  } 

  cudaHostRegister(_data, ny*nx*sizeof(double), cudaHostRegisterDefault);
  cudaHostRegister(_y, ny*sizeof(double), cudaHostRegisterDefault);
    
  CCuAccCluster* p_cluster  = new CCuAccCluster(algo,CCuAcc::get_num_device()*2,ny,nx,_data,_y,false);;

  cudaHostUnregister(_data);
  cudaHostUnregister(_y);

  tm.timer_check();

  CCuMemory<double>::showMemInfo();

  env->ReleasePrimitiveArrayCritical(y, _y, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(data, _data, JNI_ABORT);
  env->ReleaseStringUTFChars(name,_name);

  return  (jlong)p_cluster;
}

JNIEXPORT jlong JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_create_1sparse_1acc_1cluster
  (JNIEnv *env, jobject, jstring name, jdoubleArray data, jdoubleArray y, jintArray csr_ridx, jintArray csr_cidx, jint nx, jboolean intercept, jint option)
{
  CSystemInfo tm(__FUNCTION__);

  unsigned  ny  = env->GetArrayLength(csr_ridx)-1;
  unsigned  nnz = env->GetArrayLength(csr_cidx);

  char*   _name       = (char*)env->GetStringUTFChars(name,0);
  double* _data       = data? (double*)env->GetPrimitiveArrayCritical(data,0):NULL;
  double* _y          = (double*)env->GetPrimitiveArrayCritical(y,0);
  int*    _csr_ridx   = (int*)env->GetPrimitiveArrayCritical(csr_ridx,0);
  int*    _csr_cidx   = (int*)env->GetPrimitiveArrayCritical(csr_cidx,0);

  tm.timer_check();

  CMachineLearning::ALGO algo;
  if(!strcmp(_name,"LinearRegression"))         algo  = CMachineLearning::LINEAR_REGRESSION;
  else if(!strcmp(_name,"LogisticRegression"))  algo  = CMachineLearning::LOGISTIC_REGRESSION;
  else
  {
    CSystemInfo::log("ERROR: %s is not supported\n",_name);
    assert(false);
  } 

  if(data)  cudaHostRegister(_data, ny*nx*sizeof(double), cudaHostRegisterDefault);  
  cudaHostRegister(_y, ny*sizeof(double), cudaHostRegisterDefault);
  
  CCuAccCluster* p_cluster  = new CCuAccCluster(algo,CCuAcc::get_num_device()*2,ny,nx,_data,_y,_csr_ridx,_csr_cidx,nnz,intercept);

  if(data)  cudaHostUnregister(_data);
  cudaHostUnregister(_y);

  tm.timer_check();

  CCuMemory<double>::showMemInfo();

  if(data)  env->ReleasePrimitiveArrayCritical(data, _data, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(csr_ridx, _csr_ridx, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(csr_cidx, _csr_cidx, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(y, _y, JNI_ABORT);
  env->ReleaseStringUTFChars(name,_name);

  return  (jlong)p_cluster;
}

JNIEXPORT jstring JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_get_1exec_1id
  (JNIEnv *env, jobject)
{  
  CSystemInfo::mutex_lock();
  const char* uid = CSystemInfo::get_host_info();
  CSystemInfo::mutex_unlock();

  return  env->NewStringUTF(uid);
}

JNIEXPORT void JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1aug
  (JNIEnv *env, jobject, jlong handle, jdoubleArray x, jdouble intercept, jdoubleArray metric)  
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  
   
  double* _x  = (double*)env->GetPrimitiveArrayCritical(x,0);      
  double* _metric  = (double*)env->GetPrimitiveArrayCritical(metric,0);  
 
  //tm.timer_check();
  //needs to be incremental
  *_metric  = p_cluster->aug(_x,intercept); 
  //tm.timer_check();

  env->ReleasePrimitiveArrayCritical(x, _x, JNI_ABORT);   
  env->ReleasePrimitiveArrayCritical(metric, _metric, 0);
}

JNIEXPORT void JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1rmse
  (JNIEnv *env, jobject, jlong handle, jdoubleArray x, jdouble intercept, jdoubleArray metric)  
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  
   
  double* _x  = (double*)env->GetPrimitiveArrayCritical(x,0);      
  double* _metric  = (double*)env->GetPrimitiveArrayCritical(metric,0);  
 
  //tm.timer_check();
  //needs to be incremental
  *_metric  = p_cluster->rmse(_x,intercept); 
  //tm.timer_check();

  env->ReleasePrimitiveArrayCritical(x, _x, JNI_ABORT);   
  env->ReleasePrimitiveArrayCritical(metric, _metric, 0);
}






JNIEXPORT jint JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1evaluate
  (JNIEnv *env, jobject, jlong handle, jdouble mini_batch_fraction, jdouble weight_sum, jdoubleArray x, jdoubleArray g, jdoubleArray fx, jint option)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  switch(CCuAccCluster::get_set_cluster_state(p_cluster,CCuAccCluster::ACTVE)){
  case CCuAccCluster::IDLE:
    break;
  case CCuAccCluster::DELETED:
  case CCuAccCluster::INVALID:
    assert(false);
  case CCuAccCluster::ACTVE: 
    return  0;
  }


  //CSystemInfo tm(__FUNCTION__);
   
  double* _x  = (double*)env->GetPrimitiveArrayCritical(x,0);      
  double* _g  = (double*)env->GetPrimitiveArrayCritical(g,0); 
  double* _fx = (double*)env->GetPrimitiveArrayCritical(fx,0); 
 
  //tm.timer_check();
  //needs to be incremental
  *_fx  += p_cluster->evaluate(env->GetArrayLength(g),_x,_g,weight_sum); 
  //tm.timer_check();

  env->ReleasePrimitiveArrayCritical(x, _x, JNI_ABORT);   
  env->ReleasePrimitiveArrayCritical(fx, _fx, 0);
  env->ReleasePrimitiveArrayCritical(g, _g, 0);
  
  return 1;
}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_destroy_1acc_1cluster
  (JNIEnv *env, jobject, jlong handle)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  switch(CCuAccCluster::get_set_cluster_state(p_cluster,CCuAccCluster::DELETED)){
  case CCuAccCluster::ACTVE:
    assert(false);
  case CCuAccCluster::IDLE:     
    return  1;
  case CCuAccCluster::DELETED:
  case CCuAccCluster::INVALID:
    return 0;
  }
  assert(false);

  return  0;
}

JNIEXPORT void JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_reset_1acc_1cluster
  (JNIEnv *env, jobject, jlong handle)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  //allow race condition
  CCuAccCluster::set_cluster_state(p_cluster,CCuAccCluster::IDLE);
}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1weighten
  (JNIEnv *env, jobject, jlong handle, jdoubleArray weight, jdoubleArray weight_info)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  switch(CCuAccCluster::get_set_cluster_state(p_cluster,CCuAccCluster::ACTVE)){
  case CCuAccCluster::IDLE:
    break;
  case CCuAccCluster::DELETED:
  case CCuAccCluster::INVALID:
    assert(false);
  case CCuAccCluster::ACTVE: 
    return  0;
  }

  double* _weight_info = (double*)env->GetPrimitiveArrayCritical(weight_info,0);

  if(weight)
  {
    double* _weight    = (double*)env->GetPrimitiveArrayCritical(weight,0);
    p_cluster->weighten(_weight,_weight_info,_weight_info+1);
    env->ReleasePrimitiveArrayCritical(weight, _weight, JNI_ABORT);
  }
  else
  {
    p_cluster->weighten(NULL,_weight_info,_weight_info+1);
  }

   env->ReleasePrimitiveArrayCritical(weight_info, _weight_info, JNI_ABORT);

  return  1;

}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1summarize
    (JNIEnv *env, jobject, jlong handle, jdouble weight_sum, jdoubleArray data_sum, jdoubleArray data_sq_sum, jdoubleArray label_info)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  switch(CCuAccCluster::get_set_cluster_state(p_cluster,CCuAccCluster::ACTVE)){
  case CCuAccCluster::IDLE:
    break;
  case CCuAccCluster::DELETED:
  case CCuAccCluster::INVALID:
    assert(false);
  case CCuAccCluster::ACTVE: 
    return  0;
  }

  double* _sum    = (double*)env->GetPrimitiveArrayCritical(data_sum,0);
  double* _data_sq_sum = (double*)env->GetPrimitiveArrayCritical(data_sq_sum,0); 
  double* _label_info  = (double*)env->GetPrimitiveArrayCritical(label_info,0); 

  p_cluster->summarize(weight_sum,_sum,_data_sq_sum,_label_info);

  env->ReleasePrimitiveArrayCritical(data_sum, _sum, 0);
  env->ReleasePrimitiveArrayCritical(data_sq_sum, _data_sq_sum, 0); 
  env->ReleasePrimitiveArrayCritical(label_info, _label_info, 0); 

  return  1;
}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_acc_1cluster_1standardize
  (JNIEnv *env, jobject, jlong handle, jdoubleArray data_mean, jdoubleArray data_std, jdouble label_mean, jdouble label_std)
{
  CCuAccCluster*  p_cluster  = (CCuAccCluster*)handle;  

  switch(CCuAccCluster::get_set_cluster_state(p_cluster,CCuAccCluster::ACTVE)){
  case CCuAccCluster::IDLE:
    break;
  case CCuAccCluster::DELETED:
  case CCuAccCluster::INVALID:
    assert(false);
  case CCuAccCluster::ACTVE: 
    return  0;
  }

  double* _data_mean  = (double*)env->GetPrimitiveArrayCritical(data_mean,0); 
  double* _data_std   = (double*)env->GetPrimitiveArrayCritical(data_std,0); 

  p_cluster->standardize(_data_mean,_data_std,label_mean,label_std);

  env->ReleasePrimitiveArrayCritical(data_mean, _data_mean, JNI_ABORT); 
  env->ReleasePrimitiveArrayCritical(data_std, _data_std, JNI_ABORT); 

  return  1;
}


///*
// * Class:     org_apache_spark_mllib_optimization_NativeSGD
// * Method:    compress
// * Signature: ([D[IDI)[I
// */
//JNIEXPORT jintArray JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_compress
//  (JNIEnv *, jobject, jdoubleArray, jintArray, jdouble, jint);
//
///*
// * Class:     org_apache_spark_mllib_optimization_NativeSGD
// * Method:    decompress
// * Signature: ([D[IDI)V
// */
//JNIEXPORT void JNICALL Java_org_apache_spark_ml_optim_NativeCuAcc_decompress
//  (JNIEnv *, jobject, jdoubleArray, jintArray, jdouble, jint);