#include "cuAcc_base.h"


pthread_mutex_t   CSystemInfo::m_mutex  = PTHREAD_MUTEX_INITIALIZER;
void CSystemInfo::mutex_lock()  
{
  pthread_mutex_lock(&m_mutex);
}

void CSystemInfo::mutex_unlock()
{
  pthread_mutex_unlock(&m_mutex);
}

void CSystemInfo::timer_check()
{
  push_back(timer_elapsed_time());
  timer_reset();
}

double CSystemInfo::timer_elapsed_time()
{
#ifdef WIN32
  double  elapsed =  difftime(clock(), m_start)/CLOCKS_PER_SEC;
#else
  struct timespec finish;
  clock_gettime(CLOCK_MONOTONIC, &finish);
  double elapsed = (finish.tv_sec - m_start.tv_sec)+(finish.tv_nsec - m_start.tv_nsec) / 1000000000.0;
#endif

  return  elapsed;
}

void CSystemInfo::timer_reset()
{
#ifdef WIN32
  m_start = clock();  
#else
  clock_gettime(CLOCK_MONOTONIC, &m_start);
#endif
}

void CSystemInfo::timer_print()
{
  m_printed  = true;

  double  tot = 0;

  log("%s : ",m_msg.c_str());

  for(unsigned i=0;i<size();++i)
  {
    log("[%d]%e ",i,at(i));
    tot +=  at(i);
  }

  log("==> %e\n",tot);
  fflush(stdout);
}

void CSystemInfo::proc_info()
{
#ifndef WIN32
#define MAX_BUFFER  (4096)
  char  cCmd[MAX_BUFFER];
  sprintf(cCmd,"ps -p %d -o cmd h",getpid());
  //ps -p 17826 -o cmd h
  //FILE* pFile = fopen("/proc/self/cmdline","rt");
  FILE* pFile = popen(cCmd,"r");
  if(!pFile)
  {
    printf("cannot open /proc/self/cmdline!\n");
  }
  else
  {     
    for(char cBuf[MAX_BUFFER];fgets(cBuf,sizeof(cBuf),pFile);)
    {
      printf("%s\n",cBuf); 
    }

    fclose(pFile);
  }
#endif
}

const char* CSystemInfo::get_host_info()
{
  static char host_info[512]  = {NULL};
 
  if(!host_info[0])
  {
    gethostname(host_info,sizeof(host_info));
    sprintf(host_info+strlen(host_info),":%d",getpid());
  }

  return  host_info;
}

void CSystemInfo::print_stack()
{
#ifndef WIN32
#include <execinfo.h>
#include <cxxabi.h>

  /** Print a demangled stack backtrace of the caller function to FILE* out. */
  FILE *out = stderr;
  const unsigned int max_frames = 63 ;
  fprintf(out, "stack trace:\n");

  // storage array for stack trace address data
  void* addrlist[max_frames+1];

  // retrieve current stack addresses
  int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

  if (addrlen == 0) {
    fprintf(out, "  <empty, possibly corrupt>\n");
    return;
  }

  // resolve addresses into strings containing "filename(function+address)",
  // this array must be free()-ed
  char** symbollist = backtrace_symbols(addrlist, addrlen);

  // allocate string which will be filled with the demangled function name
  size_t funcnamesize = 256;
  char* funcname = (char*)malloc(funcnamesize);

  // iterate over the returned symbol lines. skip the first, it is the
  // address of this function.
  for (int i = 1; i < addrlen; i++)
  {
    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

    // find parentheses and +address offset surrounding the mangled name:
    // ./module(function+0x15c) [0x8048a6d]
    for (char *p = symbollist[i]; *p; ++p)
    {
      if (*p == '(')
        begin_name = p;
      else if (*p == '+')
        begin_offset = p;
      else if (*p == ')' && begin_offset) {
        end_offset = p;
        break;
      }
    }

    if (begin_name && begin_offset && end_offset
      && begin_name < begin_offset)
    {
      *begin_name++ = '\0';
      *begin_offset++ = '\0';
      *end_offset = '\0';

      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():

      int status;
      char* ret = abi::__cxa_demangle(begin_name,
        funcname, &funcnamesize, &status);
      if (status == 0) {
        funcname = ret; // use possibly realloc()-ed string
        fprintf(out, "  %s : %s+%s\n",
          symbollist[i], funcname, begin_offset);
      }
      else {
        // demangling failed. Output function name as a C function with
        // no arguments.
        fprintf(out, "  %s : %s()+%s\n",
          symbollist[i], begin_name, begin_offset);
      }
    }
    else
    {
      // couldn't parse the line? print the whole line.
      fprintf(out, "  %s\n", symbollist[i]);
    }
  }

  free(funcname);
  free(symbollist);
#endif
}

int CSystemInfo::log(const char * format, ... )
{
  char buffer[2560];
  va_list args;
  va_start (args, format);
  vsprintf (buffer,format, args);  
  va_end (args);

  fprintf(stdout,"%s",buffer);
  return fflush(stdout);
}  

// cuBLAS API errors
const char* CCuAcc::cublas_get_error_enum(cublasStatus_t error)
{
  switch (error){
  case CUBLAS_STATUS_SUCCESS:           return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:   return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:         
    {
      CCuMemory<double>::showMemInfo();
      return "CUBLAS_STATUS_ALLOC_FAILED";
    }
  case CUBLAS_STATUS_INVALID_VALUE:     return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:     return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:     return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:  return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}


const char* CCuAcc::cusparse_get_error_enum(cusparseStatus_t error)
{
  switch (error){
  case CUSPARSE_STATUS_SUCCESS:           return "CUSPARSE_STATUS_SUCCESS";
  case CUSPARSE_STATUS_NOT_INITIALIZED:   return "CUSPARSE_STATUS_NOT_INITIALIZED";
  case CUSPARSE_STATUS_ALLOC_FAILED:    
    {
      CCuMemory<double>::showMemInfo();
      return "CUSPARSE_STATUS_ALLOC_FAILED";
    }
  case CUSPARSE_STATUS_INVALID_VALUE:     return "CUSPARSE_STATUS_INVALID_VALUE";
  case CUSPARSE_STATUS_ARCH_MISMATCH:     return "CUSPARSE_STATUS_ARCH_MISMATCH";
  case CUSPARSE_STATUS_MAPPING_ERROR:     return "CUSPARSE_STATUS_MAPPING_ERROR";
  case CUSPARSE_STATUS_EXECUTION_FAILED:  return "CUSPARSE_STATUS_EXECUTION_FAILED";
  case CUSPARSE_STATUS_INTERNAL_ERROR:    return "CUSPARSE_STATUS_INTERNAL_ERROR";
  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
  case CUSPARSE_STATUS_ZERO_PIVOT:    return "CUSPARSE_STATUS_ZERO_PIVOT";
  }

  return "<unknown>";
}

int CCuAcc::get_num_device()
{
  int     num_device;
  CUDA_CHECK(cudaGetDeviceCount(&num_device));  

  return  num_device;
}
int CCuAcc::get_most_idle_device()
{
  //return  0;
  int     num_device  = get_num_device();
  int     most_idle_gpu   = -1;
  double  most_idle_level = 0;

  for (int i = 0;i< num_device;i++) 
  {
    CUDA_CHECK(cudaSetDevice(i));
    size_t mem_tot  = 0;
    size_t mem_free = 0;    
    CUDA_CHECK(cudaMemGetInfo(&mem_free, & mem_tot));

    double  cur_idle_level  = (double)mem_free/mem_tot;
    if(most_idle_level<cur_idle_level)
    {
      most_idle_level   = cur_idle_level;
      most_idle_gpu     = i;
    }    
  }

  assert(most_idle_gpu>=0);

  return  most_idle_gpu;
}

