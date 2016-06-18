#ifndef __CUACCCLUSTER_H__
#define __CUACCCLUSTER_H__


#include "cuAcc_function.h"
#include "cuAcc_updater.h"


class CCuAccCluster
{
public:
  enum STATE{
    INVALID,
    DELETED,
    IDLE,
    ACTVE
  };
    
  unsigned  m_nx;
  unsigned  m_ny;
  double*   m_cgrad;

  STATE     m_state;
  std::vector<CMachineLearning*> m_ml_array;

  static int   m_num_deleted;
  static std::vector<CCuAccCluster*>  m_cluster_list;
  static void   set_cluster_state(CCuAccCluster* p_cluster, STATE next);
  static STATE  get_set_cluster_state(CCuAccCluster* p_cluster, CCuAccCluster::STATE next);

  //sparse
  CCuAccCluster(CMachineLearning::ALGO algo, const unsigned num_partition, const unsigned ny, const unsigned nx,
    const double* data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const bool intercept);

  //dense
  CCuAccCluster(CMachineLearning::ALGO algo, const unsigned num_partition, const unsigned ny, const unsigned nx,
    const double* data, const double* y, const bool intercept); 

  virtual ~CCuAccCluster(void)
  {
    release();
  }

  void    release()
  {
    for(unsigned i=0,s=m_ml_array.size();i<s;++i)
      delete m_ml_array[i];

    m_ml_array.clear();

    if(m_cgrad)
    {
      delete[]  m_cgrad;
      m_cgrad = NULL;
    }
  }

  double  evaluate(unsigned size, double* x, double* cgrad, double weight_sum);
  void    synchronize();
  void    synchronize(int idx);
  void    summarize(double weight_sum, double* data_sum, double* data_sq_sum, double* label_info);
  void    standardize(double* data_mean, double* data_std, double label_mean, double label_std);
  void    weighten(double* weight, double* weight_sum, double* weight_nnz);
  double  aug(double* weight, double intercept);
  double  rmse(double* weight, double intercept);


  void    solve_sgd(double weight_sum, CUpdater* updater, unsigned int max_iteration, double lambda, double step_size, double convergence_tol, double* x);

  void    solve_lbfgs(CUpdater* updater, unsigned int max_iteration, double lambda, double step_size, double convergence_tol, double* x);

};


#endif//__CUACCCLUSTER_H__
