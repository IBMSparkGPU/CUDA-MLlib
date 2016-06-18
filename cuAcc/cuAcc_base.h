#ifndef __CUACC_BASE_H__
#define __CUACC_BASE_H__



#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h> 
#include <stdarg.h>
#include <time.h>
#include <pthread.h>
#ifdef WIN32
#include <Windows.h>
#include <process.h>
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <execinfo.h>
#include <cxxabi.h>     
#endif

#include <string>
#include <vector>
#include <algorithm>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <host_defines.h> 


#ifdef WIN32
#undef __ldg
#define __ldg *
#endif

#ifdef _DISABLE_CUDA_
#undef __device__
#define __device__ 
#undef __global__
#define __global__ 
#undef __host__
#define __host__ 
#define __syncthreads __gpu_dummy
__device__ void __syncthreads();
#undef __shared__
#define __shared__
#define __ldg *
#endif


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#define THREAD_BLOCK_SIZE (512)
#define CUDA_CHECK(call) \
{\
  cudaError_t err = (call);\
  if(cudaSuccess != err)\
{\
  fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));fflush(stderr);\
  cudaDeviceReset();\
  exit(EXIT_FAILURE);\
}\
}

#define CUBLAS_CHECK(call) \
{\
  cublasStatus_t status = (call);\
  if(CUBLAS_STATUS_SUCCESS != status)\
{\
  fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\nReason = %s\n", __FILE__, __LINE__, status,CCuAcc::cublas_get_error_enum(status));fflush(stderr);\
  cudaDeviceReset();\
  exit(EXIT_FAILURE);\
}\
}

#define CUSPARSE_CHECK(call) \
{\
  cusparseStatus_t status = (call);\
  if(CUSPARSE_STATUS_SUCCESS != status)\
{\
  fprintf(stderr,"CUSPARSE Error:\nFile = %s\nLine = %d\nCode = %d\nReason = %s\n", __FILE__, __LINE__, status,CCuAcc::cusparse_get_error_enum(status));fflush(stderr);\
  cudaDeviceReset();\
  exit(EXIT_FAILURE);\
}\
}


const double HALF = 0.5;
const double ONE  = 1;
const double MONE = -1;
const double ZERO = 0;


class CSystemInfo : private std::vector<double> {
public:
  static pthread_mutex_t   m_mutex;
 
  bool          m_printed;  
  std::string   m_msg;

#ifdef WIN32
  time_t  m_start;
#else
  struct timespec m_start;
#endif

  CSystemInfo(const char* msg="") : m_printed(false), m_msg(msg)
  {
    timer_reset();
  }

  ~CSystemInfo()
  {
    timer_check();

    if(!m_printed) 
      timer_print(); 
  }

  void    timer_check();
  double  timer_elapsed_time();
  void    timer_reset();
  void    timer_print();

  static int          log(const char * format, ... );  //timer_print log
  static void         proc_info();
  static void         print_stack();
  static const char*  get_host_info();  
  static void   mutex_lock();     //while getting the most idle device, stop others from interverning
  static void   mutex_unlock();
};

/////////////////////////////////////////////////
//top level class for CUDA accelerator
//-controls device assignment (ie get_most_idle_device)
//-other CUDA utility functions (ie errorcode2string)
//-CUDA lib handles/stremas per accelerator
class CCuAcc {
public:
  int               m_device; //assigned device to this accelerator (once set, don't change it)
  cudaStream_t      m_stream; 
  cusparseHandle_t  m_cusparse_handle;
  cublasHandle_t    m_cublas_handle;

  static const char*  cublas_get_error_enum(cublasStatus_t error);   // cuBLAS API errors
  static const char*  cusparse_get_error_enum(cusparseStatus_t error);

  static int    get_num_device();
  static int    get_most_idle_device();

  CCuAcc(int device) : m_device(device)
  {      
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreate(&m_stream));    
    
    CUBLAS_CHECK(cublasCreate(&m_cublas_handle))
    CUBLAS_CHECK(cublasSetStream(m_cublas_handle,m_stream));    

    CUSPARSE_CHECK(cusparseCreate(&m_cusparse_handle));    
    CUSPARSE_CHECK(cusparseSetStream(m_cusparse_handle,m_stream));
  }

  virtual ~CCuAcc()
  {    
    CUDA_CHECK(cudaStreamDestroy(m_stream));
    cublasDestroy(m_cublas_handle);
    cusparseDestroy(m_cusparse_handle);
  }
};

////////////////////////////////////////////////////
//CUDA memory class
//-automatic cudaFree at the end of scope
//-way to get host copy cleanly
#define _USE_PINNED_MEMORY_
#ifdef WIN32
#define _USE_VECTOR_
#endif

template <class T> 
class CCuMemory{
public:
  T*        m_dev;
  unsigned  m_count;
#ifdef _USE_VECTOR_
  std::vector<T> m_host;
#else
  T*      m_host;
#endif

  static void showMemInfo()
  {
    size_t mem_tot  = 0;
    size_t mem_free = 0;    
    cudaMemGetInfo(&mem_free, & mem_tot);

    printf("mem_free=%.2f/%.2f GB \n",mem_free/1024.0/1024/1024,mem_tot/1024.0/1024/1024);
  }

  static T* alloc( unsigned count )
  {
    T* dev;
    CUDA_CHECK(cudaMalloc( (void**)&dev, count*sizeof(T) ));

    if(!dev)
    {
      CSystemInfo::log("failed to allocate %.1f KB on GPU\n",count*sizeof(T)/1024.0);
      CSystemInfo::print_stack();
      assert(false);
    }

    return dev;
  }

  //default
  CCuMemory() : m_dev(NULL), m_host(NULL), m_count(0) {}

  //initialize with a given data in host memory
  CCuMemory(const T* host, unsigned count, cudaStream_t stream) : m_dev(NULL), m_host(NULL)
  {
    m_count = count;
    m_dev   = alloc(m_count);   
    to_dev(host,m_count,stream,0); 
  }

  //initialize with a given value, cannot be used to set values to non one-byte type
  CCuMemory(int v, unsigned count) : m_dev(NULL), m_host(NULL)
  {
    m_count = count;
    m_dev   = alloc(m_count);  
    memset(v);    
  }

  //just allocate memory on device
  CCuMemory(unsigned count) : m_dev(NULL), m_host(NULL)
  {
    m_count = count;
    m_dev   = alloc(m_count);  
  }

  //initialize with a data on the same device
  CCuMemory(CCuMemory<T>& o, cudaStream_t stream)  :  m_dev(NULL), m_host(NULL)
  {
    m_count = o.m_count;
    m_dev   = alloc(m_count); 
    from_dev(o.dev(),m_count,stream,0);
  }

  void resize(unsigned count)
  {
    release();
    m_count = count;
    m_dev   = alloc(m_count);  
  }

  void memset(int v)  
  {
    CUDA_CHECK(cudaMemset(m_dev,v,m_count*sizeof(T)));
  }

  T* dev()
  {    
    return  m_dev;
  }

  T* host()
  {
#ifdef _USE_VECTOR_
    return  &m_host.front();
#else
    assert(m_host);
    return  m_host;
#endif
  }

  //copy to member host memory
  T*  copy(cudaStream_t stream, unsigned count=0)
  {
#ifdef _USE_VECTOR_
    if(m_host.size()!=m_count)
      m_host.resize(m_count);
#else
    if(!m_host)
#ifdef _USE_PINNED_MEMORY_
      CUDA_CHECK(cudaMallocHost(&m_host,m_count*sizeof(T)));
#else
      m_host  = new T[m_count];
#endif
#endif
    if(count==0)
      to_host(host(),m_count,stream);
    else
      to_host(host(),count,stream);

#ifdef WIN32
    if(stream==0)
      cudaStreamSynchronize(stream);
#endif

    return  host();
  }
 
  void from_dev(const T* dev, unsigned count, cudaStream_t stream, const unsigned offset)
  {
    assert(m_dev);
    CUDA_CHECK(cudaMemcpyAsync(m_dev+offset,dev,count*sizeof(T),cudaMemcpyDeviceToDevice,stream));  
  }

  //read from host and write to device
  void to_dev(const T* host, unsigned count, cudaStream_t stream, const unsigned offset)
  {
    assert(host);
    assert(m_dev);
    CUDA_CHECK(cudaMemcpyAsync(m_dev+offset,host,count*sizeof(T),cudaMemcpyHostToDevice,stream));
  }


  //write back to host
  void to_host(T* host, unsigned count, cudaStream_t stream)
  {
    assert(host);
    assert(m_dev);
    CUDA_CHECK(cudaMemcpyAsync(host,m_dev,count*sizeof(T),cudaMemcpyDeviceToHost,stream));
  }

  void release(T* host, unsigned count, cudaStream_t stream)
  {
    read(host,count,stream);
    release();
  }

  void release()
  {
#ifndef _USE_VECTOR_
    if(m_host)
    {
#ifdef _USE_PINNED_MEMORY_
      cudaFreeHost(m_host);
#else
      delete[] m_host;
#endif
      m_host  = NULL;
    }
#endif

    if(m_dev)
    {
      cudaFree(m_dev);
      m_dev = NULL;  
    }
  }

  ~CCuMemory()
  { 
    release();
  }
};

#endif