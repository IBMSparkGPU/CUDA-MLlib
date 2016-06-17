#include "cuAcc_updater.h"

#include <math.h>
#include <cblas.h>
  

void CUpdater::write(const char* filename, const unsigned m_nx, const double* previousWeights, const double* x, const double* cgrad, long mini_batch_size, double step_size, int iter, double lambda)
{
  FILE* pFile = fopen(filename,"wb");
  if(!pFile)
  {
    CSystemInfo::log("cannot write to [%s]\n",filename);
    return;
  }

  fwrite(&m_nx,sizeof(unsigned),1,pFile); 

  fwrite(previousWeights,sizeof(double),m_nx,pFile);
  fwrite(x,sizeof(double),m_nx,pFile);
  fwrite(cgrad,sizeof(double),m_nx,pFile);


  fwrite(&mini_batch_size,sizeof(long),1,pFile); 
  fwrite(&step_size,sizeof(double),1,pFile); 

  fwrite(&iter,sizeof(int),1,pFile); 
  fwrite(&lambda,sizeof(double),1,pFile);   

  fclose(pFile);
}

void CUpdater::read(const char* filename, unsigned& m_nx, double*&previousWeights, double*&x, double*&cgrad, long& mini_batch_size, double& step_size, int& iter, double& lambda)
{    
  FILE* pFile = fopen(filename,"rb");
  if(!pFile)
  {
    CSystemInfo::log("cannot read [%s]\n",filename);
    return;
  }

  fread(&m_nx,sizeof(unsigned),1,pFile);

  previousWeights   = new double[m_nx];
  x           = new double[m_nx];
  cgrad       = new double[m_nx];

  fread(previousWeights,sizeof(double),m_nx,pFile);
  fread(x,sizeof(double),m_nx,pFile);
  fread(cgrad,sizeof(double),m_nx,pFile);

  fread(&mini_batch_size,sizeof(long),1,pFile); 
  fread(&step_size,sizeof(double),1,pFile); 

  fread(&iter,sizeof(int),1,pFile); 
  fread(&lambda,sizeof(double),1,pFile);   

  fclose(pFile);
}

void CUpdater::convergence_check(CCuMemory<double>& dev_x, double* convergence_info)
{
  //////////////////////////////////////////////////////////
  //convergence timer_check
  double* solutionVecDiff = convergence_info;
  double* norm_currentBDV = convergence_info+1;
  
  CUBLAS_CHECK(cublasDaxpy(m_cublas_handle,m_nx,&MONE,dev_x.dev(),1,m_dev_prev_x.dev(),1));
  CUBLAS_CHECK(cublasDnrm2(m_cublas_handle,m_nx,m_dev_prev_x.dev(),1,solutionVecDiff));
  CUBLAS_CHECK(cublasDnrm2(m_cublas_handle,m_nx,dev_x.dev(),1,norm_currentBDV));
}

void CUpdater::convergence_check(double* x, double* convergence_info)
{
  //////////////////////////////////////////////////////////
  //convergence timer_check
  double* solutionVecDiff = convergence_info;
  double* norm_currentBDV = convergence_info+1;

  cblas_daxpy(m_nx,-1,x,1,m_dev_prev_x.host(),1);
 
  *solutionVecDiff  = cblas_dnrm2(m_nx,m_dev_prev_x.host(),1);
  *norm_currentBDV  = cblas_dnrm2(m_nx,x,1);
}

double CSimpleUpdater::initialize(unsigned nx, double* x, double lambda)
{

  return  0;
}
 
double CSimpleUpdater::update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info)
{
  CSystemInfo tm(__FUNCTION__);



  return  0;
}

double CL1Updater::initialize(unsigned nx,double* x,double lambda)
{
  m_nx       = nx;
  m_dev_prev_x.resize(m_nx);
  m_dev_prev_x.to_dev(x,m_nx,m_stream,0);

  double  norm;
  CUBLAS_CHECK(cublasDasum(m_cublas_handle,m_nx,m_dev_prev_x.dev(),1,&norm));

  CUDA_CHECK(cudaStreamSynchronize(m_stream));
   
  return  lambda*norm;
}

__global__ void kernel_L1Update_compute(unsigned nx,double* x, const double soft_threshold)
{
  const unsigned  i  = threadIdx.x+blockDim.x*blockIdx.x;

  if(i>=nx)  return;

  const double  v   = x[i];

  if(v>soft_threshold)        x[i] = v-soft_threshold;
  else if(v<-soft_threshold)  x[i] = v+soft_threshold;
  else                        x[i] = 0;
}

double CL1Updater::update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info)
{
  CSystemInfo tm(__FUNCTION__);

  CUDA_CHECK(cudaSetDevice(m_device));

  CCuMemory<double>    dev_x(x,m_nx,m_stream);
  CCuMemory<double>    dev_g(cgrad,m_nx,m_stream); 

  tm.timer_check();

  //////////////////////////////////////////////////
  //L1 update
  //proximal operator to capture L1 regularization
  // w = prox ( w-alpha/batch_size*gradient_sum )

  double  alpha = step_size/sqrt(double(iter));
  double  scale = -alpha/mini_batch_size;

  CUBLAS_CHECK(cublasDaxpy(m_cublas_handle,m_nx,&scale,dev_g.dev(),1,dev_x.dev(),1));

  // for lasso, prox(x) is known that
  // prox (x) = x-k    if x>k
  //            -(x-k) if x<-k
  //            0      otherwise
  kernel_L1Update_compute<<<m_nx/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
    (m_nx,dev_x.dev(),lambda * alpha);

  //now, dev_x is read-only, copy back to host
  //TODO: is this synchronous?
  dev_x.to_host(x,m_nx,m_stream);

  return  compute_convergence_regularization(dev_x,iter,lambda,convergence_info);
}

double CL1Updater::compute_convergence_regularization(CCuMemory<double>& dev_x, int iter, double lambda, double*& convergence_info)
{
  /////////////////////////////////////////////////////////////
  //evaluate the actual regularization value
  double  regVal;
  CUBLAS_CHECK(cublasDasum(m_cublas_handle,m_nx,dev_x.dev(),1,&regVal));  
  if(iter>1)
    convergence_check(dev_x,convergence_info); 
 
  CUDA_CHECK(cudaStreamSynchronize(m_stream)); 

  /////////////////////////////////////////////////////////////
  //keep the last weight for the next iteration
  m_dev_prev_x.from_dev(dev_x.dev(),dev_x.m_count,m_stream,0);

  return  lambda*regVal;
}


double CL1AdaDeltaUpdater::initialize(unsigned nx,double* x,double lambda)
{
  CAdaDelta::initialize(nx);

  return  CL1Updater::initialize(nx,x,lambda); 
}

__global__ void kernel_L1AdaDeltaUpdate_compute(unsigned  nx, double* x, double* g,
  double* avg_squared_g, double* avg_squared_delta_x, const double step_size, const double lambda, const double rho, const double eps)
{
  const unsigned  i  = threadIdx.x+blockDim.x*blockIdx.x;

  if(i>=nx)  return;

  const double  cur_x = x[i];

  avg_squared_g[i] = rho*avg_squared_g[i]+(1-rho)*pow(g[i],2);

  double  alpha = step_size*sqrt(avg_squared_delta_x[i]+eps)/sqrt(avg_squared_g[i]+eps);

  x[i]  = cur_x-alpha*g[i];
  
  double  soft_threshold = lambda * alpha;

  //now apply prox(x)
  const double  v   = x[i];

  if(v>soft_threshold)        x[i] = v-soft_threshold;
  else if(v<-soft_threshold)  x[i] = v+soft_threshold;
  else                        x[i] = 0;

  avg_squared_delta_x[i] = rho*avg_squared_delta_x[i]+(1-rho)*pow(x[i]-cur_x,2);    
}


double CL1AdaDeltaUpdater::update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info)
{
  CSystemInfo tm(__FUNCTION__);

  CUDA_CHECK(cudaSetDevice(m_device));

  CCuMemory<double>    dev_x(x,m_nx,m_stream);
  CCuMemory<double>    dev_g(cgrad,m_nx,m_stream); 

  tm.timer_check();

  //////////////////////////////////////////////////
  //L1 AdaDelta update
  //proximal operator to capture L1 regularization
  // w = prox ( w-alpha/batch_size*gradient_sum )
  // for lasso, it is known that
  // prox (x) = x-k    if x>k
  //            -(x-k) if x<-k
  //            0      otherwise

  double  scale1  = 1.0/mini_batch_size;  
  double  rho     = 0.1;
  double  eps     = 1e-2;   
  
  CUBLAS_CHECK(cublasDscal(m_cublas_handle,m_nx,&scale1,dev_g.dev(),1));   

  kernel_L1AdaDeltaUpdate_compute<<<m_nx/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
    (m_nx,dev_x.dev(),dev_g.dev(),m_dev_avg_squared_g.dev(),m_dev_avg_squared_delta_x.dev(),
    step_size,lambda,rho,eps);

  dev_x.to_host(x,m_nx,m_stream);  

  return  compute_convergence_regularization(dev_x,iter,lambda,convergence_info);
}
 

double CL2Updater::initialize(unsigned nx,double* x,double lambda)
{
  CUDA_CHECK(cudaSetDevice(m_device));
  m_nx       = nx;
  m_dev_prev_x.resize(m_nx);
  m_dev_prev_x.to_dev(x,m_nx,m_stream,0);
  
  double  norm;
  CUBLAS_CHECK(cublasDnrm2(m_cublas_handle,m_nx,m_dev_prev_x.dev(),1,&norm));

  return  0.5*lambda*norm*norm;
}

double CL2Updater::update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info)
{
  CSystemInfo tm(__FUNCTION__);

  CUDA_CHECK(cudaSetDevice(m_device));

  CCuMemory<double>    dev_x(x,m_nx,m_stream);
  CCuMemory<double>    dev_g(cgrad,m_nx,m_stream); 

  tm.timer_check();
  
  ////////////////////////////////////////////////////
  //L2 update
  // w = w - alpha * gradient
  //where
  // gradient = graident_sum/batch_size+lambda/batch_size*w
  //in SPARK, 
  // gradient = graident_sum/batch_size+lambda*w
  //perhaps, it is just a weight factor, so won't matter much with proper lambda
  //now,
  // w = w - alpha [ (gradient_sum/batch_size) + lambda*w ]
  //   = (1-alpha*lambda)w - (alpha/batch_size)*gradient_sum
  //   = scale1*w          + (scale2)*gradient_sum
  double  alpha   = step_size/sqrt(double(iter));
  double  scale1  = 1.0-alpha*lambda; 
  double  scale2  = -alpha/mini_batch_size;
  
  CUBLAS_CHECK(cublasDscal(m_cublas_handle,m_nx,&scale1,dev_x.dev(),1));  
  CUBLAS_CHECK(cublasDaxpy(m_cublas_handle,m_nx,&scale2,dev_g.dev(),1,dev_x.dev(),1));

  //now, dev_x is read-only, copy back to host
  //TODO, is this "A"synchronous?
  dev_x.to_host(x,m_nx,m_stream);  

  return  compute_convergence_regularization(dev_x,iter,lambda,convergence_info);
}

double CL2Updater::compute_convergence_regularization(CCuMemory<double>& dev_x, int iter, double lambda, double*& convergence_info)
{
  /////////////////////////////////////////////////////////////
  //evaluate the actual regularization value
  double  regVal; 
  if(iter>1)
  {
    convergence_check(dev_x,convergence_info);
    regVal  = convergence_info[1];
  }
  else
    CUBLAS_CHECK(cublasDnrm2(m_cublas_handle,m_nx,dev_x.dev(),1,&regVal));

  CUDA_CHECK(cudaStreamSynchronize(m_stream));

  /////////////////////////////////////////////////////////////
  //keep the last weight for the next iteration
  m_dev_prev_x.from_dev(dev_x.dev(),dev_x.m_count,m_stream,0);

  return  0.5*lambda*regVal*regVal;
}

double CL2AdaDeltaUpdater::initialize(unsigned nx,double* x,double lambda)
{
  CAdaDelta::initialize(nx);

  return  CL2Updater::initialize(nx,x,lambda); 
}

__global__ void kernel_L2AdaDeltaUpdate_compute(unsigned  nx, double* x, double* g,
  double* avg_squared_g, double* avg_squared_delta_x, const double step_size, const double rho, const double eps)
{
  const unsigned  i  = threadIdx.x+blockDim.x*blockIdx.x;

  if(i>=nx)  return;

  double  cur_x = x[i];

  avg_squared_g[i] = rho*avg_squared_g[i]+(1-rho)*pow(g[i],2);

  double  alpha = step_size*sqrt(avg_squared_delta_x[i]+eps)/sqrt(avg_squared_g[i]+eps);

  x[i]  = cur_x-alpha*g[i];

  avg_squared_delta_x[i] = rho*avg_squared_delta_x[i]+(1-rho)*pow(x[i]-cur_x,2);   
}


double CL2AdaDeltaUpdater::update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info)
{
  CSystemInfo tm(__FUNCTION__);

  CUDA_CHECK(cudaSetDevice(m_device));

  CCuMemory<double>    dev_x(x,m_nx,m_stream);
  CCuMemory<double>    dev_g(cgrad,m_nx,m_stream); 

  tm.timer_check();

  //////////////////////////////////////////////////////
  // L2 AdaDelta
  // w_(i+1) = w_(i) - {RMS[diff w]_(i-1)/RMS[g]_(i)} grad
  // L2 updater
  // w       = w     - alpha                         [ (gradient_sum/batch_size) + (lambda)w ]
  //                                                 ==================> grad <===============
  // so alpha =  {RMS[w]_(i-1)/RMS[g]_(i)} in L2


  double  scale1  = 1.0/mini_batch_size; 
  double  rho     = 0.25;
  double  eps     = 1e-2;  
  
  CUBLAS_CHECK(cublasDscal(m_cublas_handle,m_nx,&scale1,dev_g.dev(),1));  
  CUBLAS_CHECK(cublasDaxpy(m_cublas_handle,m_nx,&lambda,dev_x.dev(),1,dev_g.dev(),1));

  kernel_L2AdaDeltaUpdate_compute<<<m_nx/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
    (m_nx,dev_x.dev(),dev_g.dev(),m_dev_avg_squared_g.dev(),m_dev_avg_squared_delta_x.dev(),
    step_size,rho,eps);

  dev_x.to_host(x,m_nx,m_stream);  
   
  return  compute_convergence_regularization(dev_x,iter,lambda,convergence_info);
}
 
 