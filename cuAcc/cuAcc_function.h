#ifndef __CUACC_H__
#define __CUACC_H__

#include "cuAcc_base.h"

class CCuMatrix  {
public:
  unsigned  m_ncol;
  unsigned  m_nrow;   //sample size
  
  bool      m_data_one_only;
  bool      m_label_weighted;

  static std::vector<CCuMemory<double> > m_cache_dev_one_per_device; 

  CCuMemory<double>*  m_p_dev_one;  //let's keep in the memory (don't delete)
  CCuMemory<double>   m_dev_y;    //output vector  
  CCuMemory<double>   m_dev_x_std_inv;  
  CCuMemory<double>   m_dev_x_mean;
  CCuMemory<double>   m_dev_w;    //weight vector
  double  m_label_mean;
  double  m_label_std;

  CCuMatrix(const int device, const unsigned ncol, const unsigned nrow,const double* y, cudaStream_t stream);

  virtual ~CCuMatrix() 
  {
  }

  virtual void    summarize(double weight_sum, CCuMemory<double>* p_dev_tmp_data_sum, CCuMemory<double>* p_dev_tmp_data_sq_sum, CCuMemory<double>* p_dev_tmp_label_info, CCuAcc* p_acc) = 0;
  virtual void    gemv(unsigned trans, const double* alpha, CCuMemory<double>* x, const double* beta, CCuMemory<double>* y, CCuAcc* p_acc) = 0;  
  virtual void    standardize_orig(CCuAcc* p_acc) = 0;
  virtual void    standardize_trans(CCuAcc* p_acc) = 0;
  virtual void    transform(CCuAcc* p_acc) = 0;

  void            weighten(double* weight, CCuMemory<double>* p_dev_tmp_weight_sum, CCuAcc* p_acc);
};
 
//////////////////////////////////////////////
//dense matrix
class CCuDenseMatrix : public CCuMatrix{
public:
  CCuMemory<double>    m_dev_data;  //matrix storage (on host and device partitioned way)

  CCuDenseMatrix(const int device,const unsigned ny, const unsigned nx, const double* data, const double* y, cudaStream_t stream)
    :CCuMatrix(device,nx,ny,y,stream)
  {
    m_dev_data.resize(ny*nx);
    m_dev_data.to_dev(data,ny*nx,stream,0);
  }

  virtual ~CCuDenseMatrix(){}

  virtual void    debug(){}
  virtual void    summarize(double weight_sum, CCuMemory<double>* p_dev_tmp_data_sum, CCuMemory<double>* p_dev_tmp_data_sq_sum, CCuMemory<double>* p_dev_tmp_label_info, CCuAcc* p_acc){}
  virtual void    gemv(unsigned trans, const double* alpha, CCuMemory<double>* x, const double* beta, CCuMemory<double>* y, CCuAcc* p_acc){}
  virtual void    standardize_orig(CCuAcc* p_acc){}
  virtual void    standardize_trans(CCuAcc* p_acc){}
  virtual void    transform(CCuAcc* p_acc){}

  static void write(const char* filename, const unsigned ny, const unsigned nx, const double* data, const double* y, const double* x);
  static void read(const char* filename, unsigned& ny, unsigned& nx, double*& data, double*& y, double*& x);  
};

//////////////////////////////////////////////
//sparse matrix
class CCuSparseMatrix : public CCuMatrix{
public:
  unsigned            m_nnz;
  
  cusparseMatDescr_t  m_data_descr;
  cusparseHybMat_t    m_hybA;
  cusparseHybMat_t    m_hybT;

  CCuMemory<double>   m_dev_csr_data; //matrix storage in CRS foramt for cusparse, only on device
  CCuMemory<int>      m_dev_csr_ridx;
  CCuMemory<int>      m_dev_csr_cidx;

  CCuMemory<double>   m_dev_csc_data; //matrix storage in CSC foramt for cusparse, only on device
  CCuMemory<int>      m_dev_csc_ridx;
  CCuMemory<int>      m_dev_csc_cidx;
  
  CCuSparseMatrix(const int device,const unsigned ny, const unsigned nx, const double* csr_data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, cudaStream_t stream, cusparseHandle_t cusparse_handle);

  virtual ~CCuSparseMatrix()
  {   
    cusparseDestroyMatDescr(m_data_descr);
    cusparseDestroyHybMat(m_hybA);
    cusparseDestroyHybMat(m_hybT);    
  }

  virtual void    summarize(double weight_sum, 
    CCuMemory<double>* p_dev_tmp_data_sum, CCuMemory<double>* p_dev_tmp_data_sq_sum, 
    CCuMemory<double>* p_dev_tmp_label_info, CCuAcc* p_acc); 
  virtual void    gemv(unsigned trans, const double* alpha, CCuMemory<double>* x, const double* beta, CCuMemory<double>* y, CCuAcc* p_acc);
  virtual void    standardize_orig(CCuAcc* p_acc);
  virtual void    standardize_trans(CCuAcc* p_acc);
  virtual void    transform(CCuAcc* p_acc);


  static void write(const char* filename, const unsigned ny, const unsigned nx, const double* csr_data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const double* weight);
  static void read(const char* filename,  unsigned& ny,  unsigned& nx,  double*& csr_data,  double*& y, int*& csr_ridx,  int*& csr_cidx,  unsigned& nnz,  double*& weight);
};

class CMachineLearning : public CCuAcc{
public:

  enum ALGO {
    LINEAR_REGRESSION,
    LOGISTIC_REGRESSION,
    FACTORIZATION_MACHINE
  };

  bool  m_intercept;  //intercept term 
  
  CCuMatrix*           m_p_matrix; 
  CCuMemory<double>    m_dev_buf_xy1;  //buffer at the size of max(x,y)
  CCuMemory<double>    m_dev_buf_x2;   //buffer at the size of x
  CCuMemory<double>    m_dev_buf_y1;   //buffer at the size of y

  CCuMemory<double>*   m_p_dev_tmp_data_sum;
  CCuMemory<double>*   m_p_dev_tmp_data_sq_sum;
  CCuMemory<double>*   m_p_dev_tmp_label_info; 


  CMachineLearning(int device, const bool intercept)
    : CCuAcc(device), m_intercept(intercept){}

  virtual ~CMachineLearning()
  {  
    delete  m_p_matrix;
  }

 
  virtual void    standardize(double* data_mean, double* data_std, double label_mean, double label_std) = 0;
  virtual void    evaluate(const double* x, const double weight_sum) = 0;  
  virtual double  get_fx()  { return  *(m_dev_buf_y1.host());}
  virtual double* get_g()   { return  m_dev_buf_x2.host();}
  virtual double  aug(double* label, double * fx,unsigned n) = 0;
  virtual void    predict(const double* w, const double intercept, double* fx, double* label) = 0;
  virtual void    sq_err(const double* w, const double intercept, double* fx, double* label) = 0; 

  void initialize(CCuMatrix* p_matrix)
  {
    m_p_matrix      = p_matrix;

    m_dev_buf_xy1.resize(max(p_matrix->m_ncol,p_matrix->m_nrow)+1);
    m_dev_buf_x2.resize(p_matrix->m_ncol+2);
    m_dev_buf_y1.resize(p_matrix->m_nrow+1);

    rename_memory(&m_dev_buf_x2,&m_dev_buf_xy1,&m_dev_buf_y1);
  }
  
  void rename_memory(CCuMemory<double>* p_dev_tmp_data_sum,CCuMemory<double>* p_dev_tmp_data_sq_sum,CCuMemory<double>* p_dev_tmp_label_info)
  {
    m_p_dev_tmp_data_sum     = p_dev_tmp_data_sum;
    m_p_dev_tmp_data_sq_sum  = p_dev_tmp_data_sq_sum;
    m_p_dev_tmp_label_info   = p_dev_tmp_label_info;     
  }

  virtual void    debug()
  {
    m_dev_buf_xy1.copy(0);
    m_dev_buf_x2.copy(0);
    m_dev_buf_y1.copy(0); 

    m_p_matrix->m_dev_x_std_inv.copy(0);
    m_p_matrix->m_dev_x_mean.copy(0); 
    m_p_matrix->m_dev_y.copy(0);

    if(m_p_matrix->m_dev_w.m_count)
      m_p_matrix->m_dev_w.copy(0);
  }
};

class CLinearRegression : public CMachineLearning{
public: 
  //rename device memory
  CCuMemory<double>*  m_p_dev_tmp_x;
  CCuMemory<double>*  m_p_dev_tmp_g;
  CCuMemory<double>*  m_p_dev_tmp_fx;
  CCuMemory<double>*  m_p_dev_tmp_m;

  CLinearRegression(const int device,  const unsigned ny, const unsigned nx, const double* data, const double* y, const bool intercept)
    :CMachineLearning(device, intercept)
  {
    initialize(new CCuDenseMatrix(device,ny,nx,data,y,m_stream));
    rename_memory(&m_dev_buf_x2,&m_dev_buf_x2,&m_dev_buf_y1,&m_dev_buf_y1);
  }

  CLinearRegression(const int device,  const unsigned ny, const unsigned nx, const double* data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const bool intercept)
    :CMachineLearning(device, intercept)
  {
    initialize(new CCuSparseMatrix(device,ny,nx,data,y,csr_ridx,csr_cidx,nnz,m_stream,m_cusparse_handle));
    rename_memory(&m_dev_buf_x2,&m_dev_buf_x2,&m_dev_buf_y1,&m_dev_buf_y1);
  }  

  virtual ~CLinearRegression(){}

  virtual void    evaluate(const double* x, const double weight_sum);
  virtual void    standardize(double* data_mean, double* data_std, double label_mean, double label_std); 
  virtual void    sq_err(const double* w, const double intercept, double* fx, double* label);
  virtual void    predict(const double* w, const double intercept, double* fx, double* label){assert(false);}
  virtual double  aug(double* label, double * fx, unsigned n){assert(false);return -1;}

  void  rename_memory(CCuMemory<double>* p_dev_tmp_x, CCuMemory<double>* p_dev_tmp_g, CCuMemory<double>* p_dev_tmp_m, CCuMemory<double>* p_dev_tmp_fx)
  {
    //rename device memory
    m_p_dev_tmp_x   = p_dev_tmp_x;           
    m_p_dev_tmp_g   = p_dev_tmp_g;       
    m_p_dev_tmp_m   = p_dev_tmp_m; //margin
    m_p_dev_tmp_fx  = p_dev_tmp_fx;  
  }
}; 

class CLogisticRegression : public CMachineLearning{
public: 
  CCuMemory<double>*   m_p_dev_tmp_x;
  CCuMemory<double>*   m_p_dev_tmp_g;   
  CCuMemory<double>*   m_p_dev_tmp_m;   
  CCuMemory<double>*   m_p_dev_tmp_fx;  
  CCuMemory<double>*   m_p_dev_tmp_t;   

  CLogisticRegression(const int device,  const unsigned ny, const unsigned nx, const double* data, const double* y, const bool intercept)
    :CMachineLearning(device ,intercept)
  {
    initialize(new CCuDenseMatrix(device,ny,nx,data,y,m_stream));
    rename_memory(&m_dev_buf_x2,&m_dev_buf_x2,&m_dev_buf_y1,&m_dev_buf_y1,&m_dev_buf_xy1);
  }

  CLogisticRegression(const int device,  const unsigned ny, const unsigned nx, const double* data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const bool intercept)
    :CMachineLearning(device, intercept)
  {
    initialize(new CCuSparseMatrix(device,ny,nx,data,y,csr_ridx,csr_cidx,nnz,m_stream,m_cusparse_handle));
    rename_memory(&m_dev_buf_x2,&m_dev_buf_x2,&m_dev_buf_y1,&m_dev_buf_y1,&m_dev_buf_xy1);
  }

  virtual ~CLogisticRegression(){}

  virtual void    evaluate(const double* x, const double weight_sum);
  virtual void    standardize(double* data_mean, double* data_std, double label_mean, double label_std);  
  virtual void    sq_err(const double* w, const double intercept, double* fx, double* label){assert(false);}
  virtual void    predict(const double* w, const double intercept, double* fx, double* label);
  virtual double  aug(double* label, double * fx, unsigned n);
  
  void  rename_memory(CCuMemory<double>* p_dev_tmp_x, CCuMemory<double>* p_dev_tmp_g, CCuMemory<double>* p_dev_tmp_m, CCuMemory<double>* p_dev_tmp_fx, CCuMemory<double>* p_dev_tmp_t)
  {
    //rename device memory
    m_p_dev_tmp_x   = p_dev_tmp_x;           
    m_p_dev_tmp_g   = p_dev_tmp_g;   
    m_p_dev_tmp_m   = p_dev_tmp_m; //margin
    m_p_dev_tmp_fx  = p_dev_tmp_fx;   
    m_p_dev_tmp_t   = p_dev_tmp_t; //multiplier
  }
}; 

//class CFactorizationMachine : public CCuDenseMatrix {
//public:
//  CFactorizationMachine(const int device,const unsigned ny, const unsigned nx, const double* data, const double* y)
//    :CCuDenseMatrix(device,ny,nx,data,y){}
//
//  virtual ~CFactorizationMachine(){}
//
//  virtual void    standardize(double* std){ CCuDenseMatrix::standardize(std);}
//  virtual void    summarize(){  CCuDenseMatrix::summarize();}
//  virtual void    evaluate(const double* x) ;
//};

#endif