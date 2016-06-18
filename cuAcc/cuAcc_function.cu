#include "radix_sort.h"
#include "cuAcc_function.h"
#include <cblas.h>
#include <cmath>


template<typename T>
__global__ void kernel_set(unsigned size, T* arr, T v)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  arr[idx]  = v;
}

//template<typename T>
//__global__ void kernel_seq_array(unsigned size, T* arr)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  arr[idx]  = idx;
//}

//template<typename T>
//__global__ void kernel_sq_arry(unsigned size, T* dst, T* src)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  dst[idx]  = src[idx]*src[idx];
//}
//
//template<typename T>
//__global__ void kernel_mult_array(unsigned size, T* dst, T* src)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  dst[idx]  *= src[idx];
//}
//
template<typename T>
__global__ void kernel_mult(unsigned size, T* dst, const T* __restrict__ a, const T* __restrict__ b)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  dst[idx]  = a[idx]*b[idx];
}

template<typename T>
__global__ void kernel_inv(unsigned size, T* dst, T* src)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  if(src[idx]) dst[idx] = 1.0/src[idx];
  else         dst[idx] = 0;
}



//template<typename T>
//__global__ void kernel_div_array(unsigned size, T* dst, T* src)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  if(src[idx]) dst[idx]  /= src[idx];
//  else         dst[idx] = 0;
//}

//
//template<typename T>
//__global__ void kernel_sub_array(unsigned size, T* dst, T* src)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  dst[idx]  -=  src[idx];
//}


//template<typename T>
//__global__ void kernel_dec(unsigned size, T* arr, T v)
//{
//  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
//  if(idx>=size)  return;
//
//  arr[idx]  -=  v;
//}

template<typename T>
__global__ void kernel_dec(unsigned size, T* __restrict__ arr, T* __restrict__ v)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  arr[idx]  -=  *v;
}


template<typename T>
__global__ void kernel_inc(unsigned size, T* __restrict__ arr, T* __restrict__ v)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  arr[idx]  +=  *v;
}

template<typename T>
__global__ void kernel_inc(unsigned size, T* arr, T v)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=size)  return;

  arr[idx]  +=  v;
}


__global__ void kernel(float *x, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = sqrt(pow(3.14159,i));
  }
}


/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)

Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
In other words if block_size <= 32, allocate 64*sizeof(T) bytes.
If block_size > 32, allocate block_size*sizeof(T) bytes.
*/
template <class T, unsigned int block_size>
__global__ void kernel_reduce(T* __restrict__ idata, T* __restrict__ odata, unsigned int n)
{
  extern __shared__ T __smem_d[];

  T *sdata = __smem_d;

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const unsigned  tid       = threadIdx.x;  
  const unsigned  grid_size = block_size*2*gridDim.x;
  const bool      is_pow2   = (n&(n-1))==0;

  T sum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger grid_size and therefore fewer elements per thread
  if(is_pow2)
  {
    for(unsigned i= blockIdx.x*block_size*2+threadIdx.x;i<n;i+=grid_size)
    {
      sum += idata[i];
      sum += idata[i+block_size];
    }
  }
  else
  {
    for(unsigned i= blockIdx.x*block_size*2+threadIdx.x;i<n;i+=grid_size)
    {
      sum += idata[i];

      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (i + block_size < n)
        sum += idata[i+block_size];
    }
  }


  // each thread puts its local sum into shared memory
  sdata[tid] = sum;
  __syncthreads();

  // do reduction in shared mem
  if ((block_size >= 512) && (tid < 256))  sdata[tid] = sum = sum + sdata[tid + 256];  __syncthreads();
  if ((block_size >= 256) && (tid < 128))  sdata[tid] = sum = sum + sdata[tid + 128];  __syncthreads();
  if ((block_size >= 128) && (tid <  64))  sdata[tid] = sum = sum + sdata[tid +  64];  __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
  if ( tid < 32 )
  {
    // Fetch final intermediate sum from 2nd warp
    if (block_size>=64) sum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
    {
      sum += __shfl_down(sum, offset);
    }
  }
#else
  // fully unroll reduction within a single warp
  if ((block_size>=64) && (tid < 32))  sdata[tid] = sum = sum + sdata[tid + 32];   __syncthreads();
  if ((block_size>=32) && (tid < 16))  sdata[tid] = sum = sum + sdata[tid + 16];   __syncthreads();
  if ((block_size>=16) && (tid <  8))  sdata[tid] = sum = sum + sdata[tid +  8];   __syncthreads();
  if ((block_size>= 8) && (tid <  4))  sdata[tid] = sum = sum + sdata[tid +  4];   __syncthreads();
  if ((block_size>= 4) && (tid <  2))  sdata[tid] = sum = sum + sdata[tid +  2];   __syncthreads();
  if ((block_size>= 2) && (tid <  1))  sdata[tid] = sum = sum + sdata[tid +  1];   __syncthreads();
#endif

  // write result for this block to global mem
  if (tid == 0) odata[blockIdx.x] = sum;
}

void kernel_reduce_array(double* __restrict__ idata, double* __restrict__ mdata, double* __restrict__ odata, double* __restrict__ one, unsigned n, cudaStream_t stream, cublasHandle_t cbulas_handle, double scale=ONE)
{
  const unsigned  fx_size = n/THREAD_BLOCK_SIZE+1;

  if(fx_size>1)
  {
    kernel_reduce<double,THREAD_BLOCK_SIZE> <<<fx_size, THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE* sizeof(double),stream>>>    
      (idata, mdata, n);

    CUBLAS_CHECK(cublasDgemv(cbulas_handle,CUBLAS_OP_N,1,fx_size,&scale,mdata,1,one,1,
      &ZERO,odata,1));  
  }
  else
  {
    CUBLAS_CHECK(cublasDgemv(cbulas_handle,CUBLAS_OP_N,1,n,&scale,idata,1,one,1,
      &ZERO,odata,1));  
  }
}

void kernel_mult_array(const double* __restrict__ a, const double* __restrict__ b, double* dst, unsigned n, cudaStream_t stream, cublasHandle_t cbulas_handle)
{
  kernel_mult<double><<<n/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,stream>>>(n,dst,a,b);

  //CUBLAS_CHECK(cublasDsbmv(cbulas_handle,CUBLAS_FILL_MODE_LOWER,n,0,&ONE,a,1,b,1,&ZERO,dst,1));
}

std::vector<CCuMemory<double> >  CCuMatrix::m_cache_dev_one_per_device; 

CCuMatrix::CCuMatrix(const int device, const unsigned ncol, const unsigned nrow, const double* y, cudaStream_t stream)
  :m_ncol(ncol),m_nrow(nrow),m_data_one_only(false),m_label_weighted(false)
{ 
  m_dev_y.resize(nrow);
  m_dev_y.to_dev(y,nrow,stream,0);
  m_dev_x_std_inv.resize(ncol);
  m_dev_x_mean.resize(ncol);

  //MUST BE protected by mutex outside
  if(m_cache_dev_one_per_device.empty())
    m_cache_dev_one_per_device.resize(CCuAcc::get_num_device());

  m_p_dev_one = &m_cache_dev_one_per_device[device];  
  if(m_p_dev_one->m_count<nrow)
  {
    CSystemInfo::log("device=%d  re-allocating %.2fK for ONE vector\n",device,nrow*1.30/1000);

    //should not delete it as it has been given out to others
    //need to resize the buffer
    m_p_dev_one->resize(unsigned(nrow*1.30));

    kernel_set<double><<<m_p_dev_one->m_count/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,stream>>>
      (m_p_dev_one->m_count,m_p_dev_one->dev(),1);    
  }
}


void CCuMatrix::weighten(double* weight, CCuMemory<double>* p_dev_tmp_weight_sum, CCuAcc* p_acc)
{  
  assert(weight);
  m_label_weighted  = weight;

  m_dev_w.resize(m_nrow);
  m_dev_w.to_dev(weight,m_nrow,p_acc->m_stream,0); 

  kernel_reduce_array(m_dev_w.dev(),p_dev_tmp_weight_sum->dev(),p_dev_tmp_weight_sum->dev(),m_p_dev_one->dev(),m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);

  p_dev_tmp_weight_sum->to_host(weight,1,p_acc->m_stream);
}


void  CCuDenseMatrix::write(const char* filename, const unsigned ny, const unsigned nx, const double* data, const double* y, const double* weight)
{
  FILE* pFile = fopen(filename,"wb");
  if(!pFile)
  {
    CSystemInfo::log("cannot write to [%s]\n",filename);
    return;
  }

  fwrite(&ny,sizeof(unsigned),1,pFile);
  fwrite(&nx,sizeof(unsigned),1,pFile);


  fwrite(data,sizeof(double),ny*nx,pFile);
  fwrite(y,sizeof(double),ny,pFile);
  fwrite(weight,sizeof(double),nx,pFile);

  fclose(pFile);
}

void  CCuDenseMatrix::read(const char* filename, unsigned& ny, unsigned& nx, double*& data, double*& y, double*& weight)
{    
  FILE* pFile = fopen(filename,"rb");
  if(!pFile)
  {
    CSystemInfo::log("cannot read [%s]\n",filename);
    return;
  }

  fread(&ny,sizeof(unsigned),1,pFile);
  fread(&nx,sizeof(unsigned),1,pFile);

  data    = new double[ny*nx];
  y       = new double[ny];
  weight  = new double[nx];


  fread(data,sizeof(double),ny*nx,pFile);
  fread(y,sizeof(double),ny,pFile);
  fread(weight,sizeof(double),nx,pFile);

  fclose(pFile);
}

void  CCuSparseMatrix::write(const char* filename, const unsigned ny, const unsigned nx, const double* data, const double* y, 
  const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const double* weight)
{
  FILE* pFile = fopen(filename,"wb");
  if(!pFile)
  {
    CSystemInfo::log("cannot write to [%s]\n",filename);
    return;
  }

  fwrite(&ny,sizeof(unsigned),1,pFile);
  fwrite(&nx,sizeof(unsigned),1,pFile);
  fwrite(&nnz,sizeof(unsigned),1,pFile);

  fwrite(data,sizeof(double),nnz,pFile);
  fwrite(y,sizeof(double),ny,pFile);
  fwrite(csr_ridx,sizeof(int),ny+1,pFile);
  fwrite(csr_cidx,sizeof(int),nnz,pFile);

  fwrite(weight,sizeof(double),nx,pFile);

  fclose(pFile);
}

void CCuSparseMatrix::read(const char* filename,  unsigned& ny,  unsigned& nx,  double*& data,  double*& y, 
  int*& csr_ridx,  int*& csr_cidx,  unsigned& nnz,  double*& weight)
{
  FILE* pFile = fopen(filename,"rb");
  if(!pFile)
  {
    CSystemInfo::log("cannot read [%s]\n",filename);
    return;
  }

  fread(&ny,sizeof(unsigned),1,pFile);
  fread(&nx,sizeof(unsigned),1,pFile);
  fread(&nnz,sizeof(unsigned),1,pFile);

  data      = new double[nnz];
  y     = new double[ny];
  weight    = new double[nx];
  csr_ridx  = new int[ny];
  csr_cidx  = new int[nnz];

  fread(data,sizeof(double),nnz,pFile);
  fread(y,sizeof(double),ny,pFile);
  fread(csr_ridx,sizeof(int),ny+1,pFile); 
  fread(csr_cidx,sizeof(int),nnz,pFile); 

  fread(weight,sizeof(double),nx,pFile);

  fclose(pFile);
} 

__global__ void kernel_logit(unsigned ny, double* __restrict__ m, double* __restrict__ y)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=ny)  return;

  const double  v = m[idx];

  if(v==0)          
    y[idx]  = 0.5;
  else
  {
    const double  exp_v = exp(v);
    y[idx]  = 1.0/(1+exp_v);
  }
}


#define EXP_THRESHOLD     (36)
__global__ void kernel_logistic_fx(unsigned ny, double* __restrict__ m2fx, double* __restrict__ y, double* __restrict__ t)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;
  if(idx>=ny)  return;

  const double  v = m2fx[idx];  //this is margin, will be filled with cost

  if(y[idx]>0)  m2fx[idx] = 0;
  else          m2fx[idx] = -v;  

  if(v==0)                  
  {    
    m2fx[idx] +=  0.69314718055994529;
    t[idx]    =  0.5-y[idx]; 
  }
  else if(v>EXP_THRESHOLD)   
  {
    m2fx[idx] +=  v;
    t[idx]    = -y[idx]; 
  }
  else if(v>EXP_THRESHOLD/2)  
  {
    const double  exp_v = exp(v);

    m2fx[idx] +=  v+1/exp_v;
    t[idx]    =  1.0/(1+exp_v)-y[idx];
  }
  else if(v<-EXP_THRESHOLD) 
  {
    t[idx]  = 1.0-y[idx]; 
  }
  else
  {
    const double  exp_v = exp(v);

    m2fx[idx] +=  log1p(exp_v);
    t[idx]    = 1.0/(1+exp_v)-y[idx];
  }   
}

CCuSparseMatrix::CCuSparseMatrix(const int device,const unsigned ny, const unsigned nx, const double* csr_data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, cudaStream_t stream, cusparseHandle_t cusparse_handle)
  :CCuMatrix(device,nx,ny,y,stream),
  m_nnz(nnz)
{  
  m_data_one_only = !csr_data;

  m_dev_csr_data.resize(nnz);  
  m_dev_csr_ridx.resize(ny+1);
  m_dev_csr_cidx.resize(nnz);

  if(csr_data)  m_dev_csr_data.to_dev(csr_data,nnz,stream,0);
  m_dev_csr_ridx.to_dev(csr_ridx,ny+1,stream,0);
  m_dev_csr_cidx.to_dev(csr_cidx,nnz,stream,0);  

  CUSPARSE_CHECK(cusparseCreateMatDescr(&m_data_descr));
  CUSPARSE_CHECK(cusparseSetMatType(m_data_descr,CUSPARSE_MATRIX_TYPE_GENERAL));
  CUSPARSE_CHECK(cusparseSetMatIndexBase(m_data_descr,CUSPARSE_INDEX_BASE_ZERO));  

  assert(y);

  if(csr_ridx[0]!=0)  kernel_inc<int><<<(ny+1)/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,stream>>>(ny+1,m_dev_csr_ridx.dev(),-csr_ridx[0]);     

  //since it is on-set (no data copied from host), initialize arrays with ONE
  m_dev_csc_ridx.resize(nnz);
  m_dev_csc_cidx.resize(nx+1);
  m_dev_csc_data.resize(nnz);

  CUSPARSE_CHECK(cusparseCreateHybMat(&m_hybA));
  CUSPARSE_CHECK(cusparseCreateHybMat(&m_hybT));

  CUSPARSE_CHECK(cusparseDcsr2csc(cusparse_handle,ny,nx,nnz,
    m_dev_csr_data.dev(),m_dev_csr_ridx.dev(),m_dev_csr_cidx.dev(),
    m_dev_csc_data.dev(),m_dev_csc_ridx.dev(),m_dev_csc_cidx.dev(),
    CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO));
}

template<typename T>
__global__ void kernel_standardize_csr_matrix(unsigned size, T* __restrict__ m, int* __restrict__ cidx, T* __restrict__ f)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=size)  return;

  m[idx]  *=  f[cidx[idx]];
}

template<typename T>
__global__ void kernel_standardize_csc_matrix(unsigned size, T* __restrict__ m,int* __restrict__ cidx, T* __restrict__ f)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=size)  return;

  const double  den  = f[idx];

  for(unsigned i=cidx[idx],s=cidx[idx+1];i<s;++i)
    m[i]  *=  den;
}

template<typename T>
__global__ void kernel_count_column_csc_matrix(unsigned size,int* __restrict__ cidx, T* __restrict__ s)
{
  const unsigned  idx  = threadIdx.x+blockDim.x*blockIdx.x;

  if(idx>=size)  return;

  s[idx] = cidx[idx+1]-cidx[idx];
}

#define _USE_HYB_

void CCuSparseMatrix::transform(CCuAcc* p_acc)
{
#ifdef _USE_HYB_
  CUSPARSE_CHECK(cusparseDcsr2hyb(p_acc->m_cusparse_handle,m_nrow,m_ncol,m_data_descr,m_dev_csr_data.dev(),m_dev_csr_ridx.dev(),m_dev_csr_cidx.dev(),m_hybA,0,CUSPARSE_HYB_PARTITION_AUTO));    
  CUSPARSE_CHECK(cusparseDcsr2hyb(p_acc->m_cusparse_handle,m_ncol,m_nrow,m_data_descr,m_dev_csc_data.dev(),m_dev_csc_cidx.dev(),m_dev_csc_ridx.dev(),m_hybT,0,CUSPARSE_HYB_PARTITION_AUTO));        
#endif
}

void CCuSparseMatrix::standardize_orig(CCuAcc* p_acc)
{
  kernel_standardize_csr_matrix<double><<<m_nnz/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,p_acc->m_stream>>>
    (m_nnz,m_dev_csr_data.dev(),m_dev_csr_cidx.dev(),m_dev_x_std_inv.dev());
}

void CCuSparseMatrix::standardize_trans(CCuAcc* p_acc)
{
  kernel_standardize_csc_matrix<double><<<m_ncol/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,p_acc->m_stream>>>
    (m_ncol,m_dev_csc_data.dev(),m_dev_csc_cidx.dev(),m_dev_x_std_inv.dev());
}


void CCuSparseMatrix::summarize(double weight_sum, CCuMemory<double>* p_dev_tmp_data_sum, CCuMemory<double>* p_dev_tmp_data_sq_sum, CCuMemory<double>* p_dev_tmp_label_info, CCuAcc* p_acc)
{ 
  /////////////////////////////////////////////
  //  <------------ x ------------->
  //  ^
  //  |
  //  |
  //  y   need to add up per column
  //  |   => use transposed matrix
  //  |      multiply all-one vector
  //  L
  //    ==========================
  //    sum sum sum sum

  assert(p_dev_tmp_data_sum&&p_dev_tmp_data_sum->m_dev);
  assert(p_dev_tmp_data_sq_sum&&p_dev_tmp_data_sq_sum->m_dev);
  assert(p_dev_tmp_label_info&&p_dev_tmp_label_info->m_dev);
  assert(p_dev_tmp_data_sum->m_count>=m_ncol);
  assert(m_p_dev_one);
  assert(m_p_dev_one->m_count>=m_nrow);

  if(m_data_one_only)
  {
    //set data now
    //if we do in the constructor, it would be "synchronous" due to csr->csc conversion
    kernel_set<double><<<m_nnz/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,p_acc->m_stream>>>(m_nnz,m_dev_csr_data.dev(),1);      
    kernel_set<double><<<m_nnz/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,p_acc->m_stream>>>(m_nnz,m_dev_csc_data.dev(),1);  
  }

  ///////////////////////////////////////////////////////////////////////////
  //data summary
  //scale down data by num_sample
  double  scale = 1/weight_sum;

  if(m_data_one_only&&!m_label_weighted)
  {
    kernel_count_column_csc_matrix<double><<<m_ncol/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,p_acc->m_stream>>>
      (m_ncol,m_dev_csc_cidx.dev(),p_dev_tmp_data_sum->dev());

    CUBLAS_CHECK(cublasDscal(p_acc->m_cublas_handle,m_ncol,&scale,p_dev_tmp_data_sum->dev(),1));   

    p_dev_tmp_data_sq_sum->from_dev(p_dev_tmp_data_sum->dev(),m_ncol,p_acc->m_stream,0);//if all ones, then square sum is identical
  }
  else if(m_data_one_only&&m_label_weighted)
  {
    CUSPARSE_CHECK(cusparseDcsrmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m_ncol,m_nrow,m_nnz,&ONE,
      m_data_descr,m_dev_csc_data.dev(),m_dev_csc_cidx.dev(),m_dev_csc_ridx.dev(),
      m_dev_w.dev(),&ZERO,p_dev_tmp_data_sum->dev())); 

    CUBLAS_CHECK(cublasDscal(p_acc->m_cublas_handle,m_ncol,&scale,p_dev_tmp_data_sum->dev(),1));   

    p_dev_tmp_data_sq_sum->from_dev(p_dev_tmp_data_sum->dev(),m_ncol,p_acc->m_stream,0);//if all ones, then square sum is identical
  }
  else
  {
    CUSPARSE_CHECK(cusparseDcsrmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m_ncol,m_nrow,m_nnz,&ONE,
      m_data_descr,m_dev_csc_data.dev(),m_dev_csc_cidx.dev(),m_dev_csc_ridx.dev(),
      m_dev_w.dev(),&ZERO,p_dev_tmp_data_sum->dev())); 

    //TODO: not optimized yet
    CCuMemory<double> data_squre(m_nnz);

    kernel_mult_array(m_dev_csc_data.dev(),m_dev_csc_data.dev(),data_squre.dev(),m_nnz,p_acc->m_stream,p_acc->m_cublas_handle);

    CUSPARSE_CHECK(cusparseDcsrmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m_ncol,m_nrow,m_nnz,&ONE,
      m_data_descr,data_squre.dev(),m_dev_csc_cidx.dev(),m_dev_csc_ridx.dev(),
      m_p_dev_one->dev(),&ZERO,p_dev_tmp_data_sq_sum->dev())); 

    CUBLAS_CHECK(cublasDscal(p_acc->m_cublas_handle,m_ncol,&scale,p_dev_tmp_data_sum->dev(),1));   
    CUBLAS_CHECK(cublasDscal(p_acc->m_cublas_handle,m_ncol,&scale,p_dev_tmp_data_sq_sum->dev(),1));   
  }



  //////////////////////////////////////////////////////////////////////
  //label summary

  //label sum at p_dev_tmp_label_info[0] 
  double* p_label_sum  = p_dev_tmp_label_info->dev();

  if(m_label_weighted)
  {
    kernel_mult_array(m_dev_y.dev(),m_dev_w.dev(),p_label_sum,m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);

    //weight*label sum at p_dev_tmp_label_info[0]
    kernel_reduce_array(p_label_sum,p_label_sum,p_label_sum,m_p_dev_one->dev(),m_nrow,p_acc->m_stream,p_acc->m_cublas_handle); 
  }
  else
  {
    //label sum at p_dev_tmp_label_info[0]
    kernel_reduce_array(m_dev_y.dev(),p_label_sum,p_label_sum,m_p_dev_one->dev(),m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);  
  }

  //label sq_sum at p_dev_tmp_label_info[1] 
  double* p_label_sq_sum  = p_dev_tmp_label_info->dev()+1;

  //make a square first
  kernel_mult_array(m_dev_y.dev(),m_dev_y.dev(),p_label_sq_sum,m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);

  if(m_label_weighted)    //then mult weight
    kernel_mult_array(m_dev_w.dev(),p_label_sq_sum,p_label_sq_sum,m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);

  //add up
  kernel_reduce_array(p_label_sq_sum,p_label_sq_sum,p_label_sq_sum,m_p_dev_one->dev(),m_nrow,p_acc->m_stream,p_acc->m_cublas_handle);  

  //don't scale down label as it is a scala value
  p_dev_tmp_label_info->copy(p_acc->m_stream,2);//label sum and sq_sum
  p_dev_tmp_data_sum->copy(p_acc->m_stream);    //data sum
  p_dev_tmp_data_sq_sum->copy(p_acc->m_stream); //data sq_sum
} 

void CCuSparseMatrix::gemv(unsigned trans, const double* alpha, CCuMemory<double>* x, const double* beta, CCuMemory<double>* y, CCuAcc* p_acc)
{
  if(trans==CUSPARSE_OPERATION_NON_TRANSPOSE)
  {
#ifdef _USE_HYB_
    CUSPARSE_CHECK(cusparseDhybmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,alpha,m_data_descr,m_hybA,x->dev(),beta,y->dev()));
#else
    CUSPARSE_CHECK(cusparseDcsrmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m_nrow,m_ncol,m_nnz,alpha,
      m_data_descr,m_dev_csr_data.dev(),m_dev_csr_ridx.dev(),m_dev_csr_cidx.dev(),
      x->dev(),
      beta,y->dev()));  
#endif
  }
  else
  {
#ifdef _USE_HYB_
    CUSPARSE_CHECK(cusparseDhybmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,alpha,m_data_descr,m_hybT,x->dev(),beta,y->dev()));  
#else
    CUSPARSE_CHECK(cusparseDcsrmv(p_acc->m_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m_ncol,m_nrow,m_nnz,alpha,
      m_data_descr,m_dev_csc_data.dev(),m_dev_csc_cidx.dev(),m_dev_csc_ridx.dev(),
      x->dev(),
      beta,y->dev()));    
#endif
  }
}
 

void CLinearRegression::standardize(double* data_mean, double* data_std, double label_mean, double label_std)
{
  //keep stat in memory
  m_p_matrix->m_label_mean  = label_mean;
  m_p_matrix->m_label_std   = label_std;

  m_p_matrix->m_dev_x_mean.to_dev(data_mean,m_p_matrix->m_ncol,m_stream,0); 
  m_p_matrix->m_dev_x_std_inv.to_dev(data_std,m_p_matrix->m_ncol,m_stream,0);

  kernel_inv<double><<<m_p_matrix->m_ncol/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
    (m_p_matrix->m_ncol,m_p_matrix->m_dev_x_std_inv.dev(),m_p_matrix->m_dev_x_std_inv.dev());   

  //////////////////////////////////////////////////////////
  //standardize data
  m_p_matrix->standardize_trans(this);
  m_p_matrix->transform(this);

  /////////////////////////////////////////////////////////
  //standardize label
  assert(label_std!=0);
  if(label_std!=1)
  {
    double  scale = 1.0/label_std;
    CUBLAS_CHECK(cublasDscal(m_cublas_handle,m_p_matrix->m_nrow,&scale,m_p_matrix->m_dev_y.dev(),1));   
  }
}


void CLinearRegression::evaluate(const double* x, const double weight_sum)
{
  const unsigned  nx      = m_p_matrix->m_ncol;
  const unsigned  ny      = m_p_matrix->m_nrow; 
  const unsigned  fx_size = ny/THREAD_BLOCK_SIZE+1;

  m_p_dev_tmp_x->to_dev(x,nx,m_stream,0);
  m_p_dev_tmp_m->from_dev(m_p_matrix->m_dev_y.dev(),ny,m_stream,0);

  //effective X
  kernel_mult_array(m_p_dev_tmp_x->dev(),m_p_matrix->m_dev_x_std_inv.dev(),m_p_dev_tmp_x->dev(),nx,m_stream,m_cublas_handle);

  double* sum     = m_p_dev_tmp_x->dev()+nx;
  double* offset  = sum+1;

  CUBLAS_CHECK(cublasDgemv(m_cublas_handle,CUBLAS_OP_N,1,nx,&ONE,m_p_dev_tmp_x->dev(),1,m_p_matrix->m_dev_x_mean.dev(),1,&ZERO,sum,1));

  if(m_intercept)
  {
    kernel_set<double><<<1,1,0,m_stream>>>(1,offset,m_p_matrix->m_label_mean/m_p_matrix->m_label_std); 
    CUBLAS_CHECK(cublasDaxpy(m_cublas_handle,1,&MONE,sum,1,offset,1));
  }
  else
    kernel_set<double><<<1,1,0,m_stream>>>(1,offset,0); 

  ///////////main///////////
  //Dw-l => margin
  m_p_matrix->gemv(CUBLAS_OP_N,&ONE,m_p_dev_tmp_x,&MONE,m_p_dev_tmp_m,this);   

  if(m_intercept)
    kernel_inc<double><<<fx_size,THREAD_BLOCK_SIZE,0,m_stream>>>(ny,m_p_dev_tmp_m->dev(),offset);  


  CCuMemory<double>*  p_dev_tmp_wm = NULL;

  //weight*margin
  if(m_p_matrix->m_label_weighted)
  {
    assert(m_p_matrix->m_dev_w.m_count==ny);

    p_dev_tmp_wm = &m_dev_buf_xy1; 

    kernel_mult_array(m_p_dev_tmp_m->dev(),m_p_matrix->m_dev_w.dev(),p_dev_tmp_wm->dev(),ny,m_stream,m_cublas_handle);
  }
  else
  {
    p_dev_tmp_wm = m_p_dev_tmp_m;
  }

  //(Dw-l)D => gradient and scale down gradient
  double  scale = 1/weight_sum;
  m_p_matrix->gemv(CUBLAS_OP_T,&scale,p_dev_tmp_wm,&ZERO,m_p_dev_tmp_g,this);

  //(Dw-l)^2 => cost
  CUBLAS_CHECK(cublasDgemv(m_cublas_handle,CUBLAS_OP_N,1,ny,&HALF,p_dev_tmp_wm->dev(),1,
    m_p_dev_tmp_m->dev(),1,&ZERO,m_p_dev_tmp_fx->dev(),1));    

  //debug();
  //CSystemInfo::log("weight_sum =%f,m_p_dev_tmp_g[0]=%f\n",weight_sum,m_p_dev_tmp_g->m_host[0]);
  //CSystemInfo::log("m_p_dev_tmp_fx[0]=%f\n",m_p_dev_tmp_fx->m_host[0]);

  m_p_dev_tmp_g->copy(m_stream);
  m_p_dev_tmp_fx->copy(m_stream,1);
}

void CLinearRegression::sq_err(const double* w, const double intercept, double* fx, double* label)
{
  //compute squared error
  const unsigned  nx      = m_p_matrix->m_ncol;
  const unsigned  ny      = m_p_matrix->m_nrow; 
  const unsigned  fx_size = ny/THREAD_BLOCK_SIZE+1;

  m_p_dev_tmp_x->to_dev(w,nx,m_stream,0);   
  m_p_dev_tmp_m->from_dev(m_p_matrix->m_dev_y.dev(),ny,m_stream,0);

  if(m_p_matrix->m_label_std!=1)
  {
    double  scale = m_p_matrix->m_label_std;
    CUBLAS_CHECK(cublasDscal(m_cublas_handle,m_p_matrix->m_nrow,&scale,m_p_dev_tmp_m->dev(),1));    
  }

  m_p_matrix->gemv(CUBLAS_OP_N,&ONE,m_p_dev_tmp_x,&MONE,m_p_dev_tmp_m,this); 

  kernel_inc<double><<<fx_size,THREAD_BLOCK_SIZE,0,m_stream>>>(ny,m_p_dev_tmp_m->dev(),intercept);  

  kernel_mult_array(m_p_dev_tmp_m->dev(),m_p_dev_tmp_m->dev(),m_p_dev_tmp_m->dev(),ny,m_stream,m_cublas_handle);
 
  kernel_reduce_array(m_p_dev_tmp_m->dev(),m_p_dev_tmp_m->dev(),m_p_dev_tmp_m->dev(),m_p_matrix->m_p_dev_one->dev(),ny,m_stream,m_cublas_handle);

  m_p_dev_tmp_m->to_host(fx,1,m_stream);
}

void CLogisticRegression::standardize(double* data_mean, double* data_std, double label_mean, double label_std)
{
  //; 

  //float *data;
  //cudaMalloc(&data, 1048576 * sizeof(float));
  //// launch one worker kernel per stream
  //kernel<<<1, 64, 0, stream>>>(data, 1048576); 

  m_p_matrix->m_dev_x_std_inv.to_dev(data_std,m_p_matrix->m_ncol,m_stream,0);

  kernel_inv<double><<<m_p_matrix->m_ncol/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
    (m_p_matrix->m_ncol,m_p_matrix->m_dev_x_std_inv.dev(),m_p_matrix->m_dev_x_std_inv.dev());   

  //////////////////////////////////////////////////////////
  //standardize data
  m_p_matrix->standardize_orig(this);
  m_p_matrix->standardize_trans(this);
  m_p_matrix->transform(this);
}

void CLogisticRegression::evaluate(const double* x, const double weight_sum)
{
  const unsigned  nx      = m_p_matrix->m_ncol;
  const unsigned  ny      = m_p_matrix->m_nrow; 
  const unsigned  fx_size = ny/THREAD_BLOCK_SIZE+1;

  //if m_intercept is given, it can be 1 longer than nx
  m_p_dev_tmp_x->to_dev(x,nx+m_intercept,m_stream,0); 

  //Dw => margin to be used in logistic cost function
  m_p_matrix->gemv(CUSPARSE_OPERATION_NON_TRANSPOSE,&MONE,m_p_dev_tmp_x,&ZERO,m_p_dev_tmp_m,this);   

  if(m_intercept)    //decrease the margin by the last in x
    kernel_dec<double><<<(ny+1)/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>(ny,m_p_dev_tmp_m->dev(),m_p_dev_tmp_x->dev()+nx);

  //for every sample, evaluate gradient and cost using logistic cost function
  //model: h(v) = 1over1pExp(v)
  //cost : H(Dw,l) = -log (h(Dw))   if l==1
  //                 -log (1-h(Dw)) if l==0
  //input : m_p_dev_buf_xy1 => y l
  //        m_p_dev_buf_y1 => margin Dw
  //output: m_p_dev_buf_xy1 <= h(Dw)-l for gradient computation
  //        m_p_dev_buf_y1 <=  H(Dw,I) for cost estimation
  kernel_logistic_fx<<<fx_size,THREAD_BLOCK_SIZE,0,m_stream>>>
    (ny,m_p_dev_tmp_m->dev(),m_p_matrix->m_dev_y.dev(),m_p_dev_tmp_t->dev());    

  //(h(Dw)-l)D => gradient and scale down
  double  scale = 1/weight_sum;
  m_p_matrix->gemv(CUSPARSE_OPERATION_TRANSPOSE,&scale,m_p_dev_tmp_t,&ZERO,m_p_dev_tmp_g,this); 

  if(m_intercept)    //add up m_p_dev_tmp_t to the last in m_p_dev_tmp_g 
    kernel_reduce_array(m_p_dev_tmp_t->dev(),m_p_dev_tmp_t->dev(),m_p_dev_tmp_g->dev()+nx,m_p_matrix->m_p_dev_one->dev(),ny,m_stream,m_cublas_handle,scale);  


  //add up H(Dw,I) for the total cost
  kernel_reduce_array(m_p_dev_tmp_fx->dev(),m_p_dev_tmp_fx->dev(),m_p_dev_tmp_fx->dev(),m_p_matrix->m_p_dev_one->dev(),ny,m_stream,m_cublas_handle);

  m_p_dev_tmp_g->copy(m_stream);
  m_p_dev_tmp_fx->copy(m_stream,1);
}

void CLogisticRegression::predict(const double* w, const double intercept, double* fx, double* label)
{
  const unsigned  nx      = m_p_matrix->m_ncol;
  const unsigned  ny      = m_p_matrix->m_nrow; 
  const unsigned  fx_size = ny/THREAD_BLOCK_SIZE+1;

  m_p_dev_tmp_x->to_dev(w,nx,m_stream,0);  

  if(m_p_matrix->m_dev_x_std_inv.m_count)
  {
    //need to invert first
    kernel_inv<double><<<nx/THREAD_BLOCK_SIZE+1,THREAD_BLOCK_SIZE,0,m_stream>>>
      (nx,m_p_dev_tmp_t->dev(),m_p_matrix->m_dev_x_std_inv.dev());   

    kernel_mult_array(m_p_dev_tmp_x->dev(),m_p_dev_tmp_t->dev(),m_p_dev_tmp_x->dev(),nx,m_stream,m_cublas_handle);
  }

  m_p_matrix->gemv(CUSPARSE_OPERATION_NON_TRANSPOSE,&MONE,m_p_dev_tmp_x,&ZERO,m_p_dev_tmp_m,this);   

  kernel_inc<double><<<fx_size,THREAD_BLOCK_SIZE,0,m_stream>>>
    (ny,m_p_dev_tmp_t->dev(),intercept);    

  kernel_logit<<<fx_size,THREAD_BLOCK_SIZE,0,m_stream>>>
    (ny,m_p_dev_tmp_m->dev(),m_p_dev_tmp_t->dev());  

  m_p_matrix->m_dev_y.to_host(label,ny,m_stream);
  m_p_dev_tmp_t->to_host(fx,ny,m_stream);
}

double CLogisticRegression::aug(double* label, double * fx, unsigned n)
{   
  CSystemInfo t(__FUNCTION__);

  radix_sort<double,double>(fx,label,n);
  t.timer_check();

  std::reverse(fx,fx+n);
  std::reverse(label,label+n); 

  t.timer_check();
  /* Count number of positive and negative examples first */
  unsigned N=0,P=0;
  for(unsigned i = 0; i < n ; i++) 
  {
    if(label[i] == 1) P++;
  }

  N = n-P;

  /* Then calculate the actual are under the ROC curve */
  double    fprev   = INT_MIN;
  double    A       = 0;
  unsigned  FP      = 0, 
    TP      = 0,
    FPprev  = 0, 
    TPprev  = 0;

  double  _fx;
  double  _label; 

  for(unsigned i = 0 ; i < n; i++) 
  {
    _fx     = fx[i];
    _label  = label[i];
    if(_fx != fprev) 
    {
      /* Divide area here already : a bit slower, but gains in precision and avoids overflows */
      assert(FP>=FPprev);

      A       +=  double(FP-FPprev)/N*double(TP+TPprev)/P;
      fprev	  = _fx;
      FPprev	= FP;
      TPprev	= TP;
    }

    int label_one = _label==1;
    TP  +=  label_one;
    FP  +=  1-label_one;
  }

  A +=  double(N-FPprev)/N*double(P+TPprev)/P;
  /* Not the same as Fawcett's original pseudocode though I think his contains a typo */
  t.timer_check();

  return  A*0.5;
}


//
//void CFactorizationMachine::evaluate(const double* x)
//{
//
//}
