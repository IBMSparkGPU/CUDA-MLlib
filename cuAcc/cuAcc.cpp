#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <assert.h> 
#include <map>
#include <string>

#include "cuAcc_base.h"
#include "cuAcc_function.h"
#include "cuAcc_updater.h"
#include "cuAcc_cluster.h"
#include "LibSVMParser.h"

//////#include <device_functions.h>
//////
//////
//////
//////__device__ double atomicAdd(double* address, double val)
//////{
//////  unsigned long long int* address_as_ull = (unsigned long long int*)address;
//////  unsigned long long int old = *address_as_ull, assumed;
//////  do {
//////    assumed = old;
//////    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
//////  } while (assumed != old);
//////  return __longlong_as_double(old);
//////}
//////#define WARP_SIZE (32)
//////template<typename T>
//////__global__ void kernel_trans_csrmv(int numRow, int* ptr, int* idx, T* val, T* y, T* x){
//////
//////  // global thread index
//////  int thread_id = THREAD_BLOCK_SIZE * blockIdx.x + threadIdx.x;
//////
//////  // thread index within the warp
//////  int thread_lane = threadIdx.x & (WARP_SIZE-1);
//////  // global warp index
//////  int warp_id = thread_id / WARP_SIZE;
//////  // total number of active warps
//////  int num_warps = (THREAD_BLOCK_SIZE / WARP_SIZE) * gridDim.x;
//////  for(unsigned row=warp_id; row < numRow; row+=num_warps){
//////    int row_start = ptr[row];
//////    int row_end = ptr[row+1];
//////    for (unsigned i=row_start+thread_lane; i < row_end;i+=WARP_SIZE)
//////      atomicAdd(x+idx[i], val[i] * y[row]);
//////  }
//////}


////template<typename T>
//// __global__ void spmv_csr_vector_kernel (const int num_rows, const int *ptr, const int *indices, const T *data, const T *x , T *y) {
////
////	__shared__ T vals[THREAD_BLOCK_SIZE];
////
////	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;	// global thread index
////
////	int warp_id = thread_id / WARP_SIZE;							// global warp index
////
////	int lane = thread_id & (WARP_SIZE - 1);						// thread index within the warp
////
////	
////
////	// one warp per row
////
////	int row = warp_id;
////
////
////
////	if (row < num_rows) {
////
////		int row_start = ptr[row];
////
////		int row_end = ptr[row+1];
////
////		
////
////		// compute running sum per thread
////
////		vals [threadIdx.x] = 0;
////
////		for (int jj = row_start + lane; jj < row_end; jj += WARP_SIZE)
////
////			vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];
////
////		
////
////		// parallel reduction in shared memory
////
////		if ( lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
////
////		if ( lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8];
////
////		if ( lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4];
////
////		if ( lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2];
////
////		if ( lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];
////
////
////
////		// first thread writes the result
////		if (lane == 0)
////			y[row] += vals[threadIdx.x];
////	}
////}
//// 
//
//int test(int argc, char* argv[])
//{
//  /////////////////////////////////////////////////
//  //parse the input
//  std::map<std::string,std::string>    Parm;
//
//  printf("cmd: %s ",argv[0]);
//
//  char* pLast = NULL;
//  for(int iIdx=1;iIdx<argc;++iIdx)
//  {
//    // Skip any '+'.
//    // This is here because ddd has arguments like -r, so we need to pass to ddd +-r
//    char *arg = (argv[iIdx][0]=='+') ? argv[iIdx]+1 : argv[iIdx];
//    if(arg[0]=='-')           
//    {
//      Parm[arg+1] = "{empty}";
//      pLast       = arg+1;
//    }
//    else if(pLast)
//    {
//      Parm[pLast] =  arg;
//    }
//
//    printf("%s ",argv[iIdx]);
//  }
//
//  printf("\n");
//
//  if(!Parm["libsvm"].empty())
//  {
//    //read the libsvm file
//    CLibSVMParser<double> P;
//    P.read_libsvm(Parm["libsvm"].c_str());
//
//    printf("nx=%d\n",P.get_nx());
//    printf("ny=%d\n",P.get_ny());
//
//    //cusparse initialization
//    cudaStream_t      stream; 
//    cusparseHandle_t  cusparse_handle; 
//    cusparseMatDescr_t    data_descr;
//    cudaStreamCreate(&stream);    
//
//    cusparseCreate(&cusparse_handle);    
//    cusparseSetStream(cusparse_handle,stream);
//
//    cusparseCreateMatDescr(&data_descr);
//    cusparseSetMatType(data_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(data_descr,CUSPARSE_INDEX_BASE_ZERO);      
//
//    double*   p_dev_csr_data;
//    int*      p_dev_csr_ridx;
//    int*      p_dev_csr_cidx;
//
//    double*   p_dev_csc_data;
//    int*      p_dev_csc_ridx;
//    int*      p_dev_csc_cidx;
//
//    cudaMalloc( (void**)&p_dev_csr_data, P.get_nnz()*sizeof(double) );
//    cudaMalloc( (void**)&p_dev_csr_ridx, (P.get_ny()+1)*sizeof(int) );
//    cudaMalloc( (void**)&p_dev_csr_cidx, P.get_nnz()*sizeof(int) );
//
//    cudaMemcpyAsync(p_dev_csr_data,P.get_csr_data(),P.get_nnz()*sizeof(double),cudaMemcpyHostToDevice,stream);
//    cudaMemcpyAsync(p_dev_csr_ridx,P.get_csr_ridx(),(P.get_ny()+1)*sizeof(int),cudaMemcpyHostToDevice,stream);
//    cudaMemcpyAsync(p_dev_csr_cidx,P.get_csr_cidx(),P.get_nnz()*sizeof(int),cudaMemcpyHostToDevice,stream);
//
//    CSystemInfo t;
//
//    cudaMalloc( (void**)&p_dev_csc_data, P.get_nnz()*sizeof(double) );
//    cudaMalloc( (void**)&p_dev_csc_cidx, (P.get_nx()+1)*sizeof(int) );
//    cudaMalloc( (void**)&p_dev_csc_ridx, P.get_nnz()*sizeof(int) );
//
//    if(CUSPARSE_STATUS_SUCCESS!=cusparseDcsr2csc(cusparse_handle,P.get_ny(),P.get_nx(),P.get_nnz(),
//      p_dev_csr_data,p_dev_csr_ridx,p_dev_csr_cidx,
//      p_dev_csc_data,p_dev_csc_ridx,p_dev_csc_cidx,
//      CUSPARSE_ACTION_NUMERIC,CUSPARSE_INDEX_BASE_ZERO))
//    {
//      assert(false);
//    }
//
//    t.timer_check();
//
//
//    double  alpha = 1.0;
//    double  beta  = 0.0;
//
//    double* x;
//    double* y;
//
//    cudaMalloc( (void**)&x, P.get_nx()*sizeof(double) );
//    cudaMalloc( (void**)&y, P.get_ny()*sizeof(double) ); 
//
//    cudaMemset(x,1, P.get_nx()*sizeof(double));
//
//    for(int i=0;i<1000;++i)
//    {
//      //////////////////////////////////////////////////////////////////
//      //y = Ax
//      if(CUSPARSE_STATUS_SUCCESS!=cusparseDcsrmv(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,P.get_ny(),P.get_nx(),P.get_nnz(),&alpha,
//        data_descr,p_dev_csr_data,p_dev_csr_ridx,p_dev_csr_cidx,
//        x,
//        &beta,y))
//      {
//        assert(false);
//      }
//
//      CUDA_CHECK(cudaStreamSynchronize(stream));
//    }
//
//
//    t.timer_check();
//
//    for(int i=0;i<1000;++i)
//    {
//      /////////////////////////////////////////////////////////////////
//      //y = tran(A)x
//      if(CUSPARSE_STATUS_SUCCESS!=cusparseDcsrmv(cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,P.get_nx(),P.get_ny(),P.get_nnz(),&alpha,
//        data_descr,p_dev_csc_data,p_dev_csc_cidx,p_dev_csc_ridx,
//        x,
//        &beta,y))
//      {
//        assert(false);
//      }
//
//      CUDA_CHECK(cudaStreamSynchronize(stream));
//    }
//
//    t.timer_print();
//  }
//
//
//  return  0;
//}

int main(int argc, char* argv[])
{
  //return test(argc,argv);


  std::map<std::string,std::string>    Parm;

  printf("cmd: %s ",argv[0]);

  char* pLast = NULL;
  for(int iIdx=1;iIdx<argc;++iIdx)
  {
    // Skip any '+'.
    // This is here because ddd has arguments like -r, so we need to pass to ddd +-r
    char *arg = (argv[iIdx][0]=='+') ? argv[iIdx]+1 : argv[iIdx];
    if(arg[0]=='-')           
    {
      Parm[arg+1] = "{empty}";
      pLast       = arg+1;
    }
    else if(pLast)
    {
      Parm[pLast] =  arg;
    }

    printf("%s ",argv[iIdx]);
  }

  printf("\n");

  printf("%s\n",CSystemInfo::get_host_info());

  double    lambda          = 0.01;
  unsigned  max_iter        = max(10,atoi(Parm["miter"].c_str()));
  double    step_size       = 1.0;
  double    convergence_tol  = 0.000001;
  unsigned  num_partition   = max(1,atoi(Parm["nump"].c_str()));
  bool      intercept       = true;


  CLibSVMParser<double> P;
  if(!Parm["libsvm"].empty())
  {

    P.read_libsvm(Parm["libsvm"].c_str());
  }
  else if(!Parm["rdd"].empty())
  {
    P.read_rdd(Parm["rdd"].c_str());
  }
  else
  {
    assert(false);
  }

  CSystemInfo tm("main");

  printf("nx=%d\n",P.get_nx());
  printf("ny=%d\n",P.get_ny());

  std::vector<double> x(P.get_nx()+intercept,0);


  cudaDeviceReset();


  CCuAccCluster SLGC(CMachineLearning::LOGISTIC_REGRESSION,num_partition,P.get_ny(),P.get_nx(),
    NULL,P.get_y(),      
    P.get_csr_ridx(),P.get_csr_cidx(),P.get_nnz(),
    intercept
    );

  double weight_sum;
  double weight_nnz;

  SLGC.weighten(P.get_w(),&weight_sum,&weight_nnz);

  std::vector<double> mean(P.get_nx(),0);
  std::vector<double> std(P.get_nx(),0);
  std::vector<double> label_info(3,0);


  SLGC.summarize(weight_sum,&mean.front(),&std.front(),&label_info.front());


  if(intercept)
    x.back() = log(label_info[1]/label_info[0]);

  double  n = weight_sum;

  double unbiased_factor = P.get_ny() / (P.get_ny() - 1.0);

  for(unsigned i=0;i<mean.size();++i)
  {
    double x = std[i]-mean[i]*mean[i]*unbiased_factor;

    assert(x>=0);
    std[i]  = sqrt(x);
  }


  double label_mean = label_info[1] / n;
  double label_std = sqrt((label_info[2] / (n) - label_mean * label_mean) * unbiased_factor);

  double label_std2 = sqrt(

    double(P.get_ny())/(P.get_ny()-1)*
    (

    label_info[2] / (n) - label_mean * label_mean

    )

    );


  SLGC.standardize(&mean.front(),&std.front(),label_mean,label_std);

  SLGC.solve_sgd(weight_sum,&CL2Updater(),max_iter,lambda,step_size,convergence_tol,&x.front()); 

  CCuAccCluster::get_set_cluster_state(&SLGC,CCuAccCluster::DELETED);


  return 0;
}

