#include "cuAcc_base.h"
#include "cuAcc_cluster.h"
#include "cuAcc_function.h"
#include <cblas.h>

std::vector<CCuAccCluster*> CCuAccCluster::m_cluster_list;
int CCuAccCluster::m_num_deleted  = 0;
CCuAccCluster::STATE  CCuAccCluster::get_set_cluster_state(CCuAccCluster* p_cluster, CCuAccCluster::STATE next)
{
  STATE cur = INVALID;

  CSystemInfo::mutex_lock();  

  if(!m_cluster_list.empty())
  {    
    cur = p_cluster->m_state;
    p_cluster->m_state  = next;

    if(next==DELETED&&cur!=DELETED)
    {     
      p_cluster->release();      
      m_num_deleted++;

      CSystemInfo::log("%s] clear cluster list (%d/%d)\n",CSystemInfo::get_host_info(),m_num_deleted,m_cluster_list.size());

      if(m_num_deleted==m_cluster_list.size())
      {        
        for(unsigned i=0,s=m_cluster_list.size();i<s;++i)
          delete m_cluster_list[i];

        m_cluster_list.clear();
        m_num_deleted = 0;        
      }      
    }
  }

  CSystemInfo::mutex_unlock();

  return  cur;
} 

void CCuAccCluster::set_cluster_state(CCuAccCluster* p_cluster, CCuAccCluster::STATE next)
{
  p_cluster->m_state  = next;
}


CCuAccCluster::CCuAccCluster(CMachineLearning::ALGO algo, const unsigned num_partition, const unsigned ny, const unsigned nx, 
  const double* data, const double* y, const int* csr_ridx, const int* csr_cidx, const unsigned nnz, const bool intercept)
  :m_nx(nx), m_ny(ny), m_cgrad(NULL), m_state(IDLE)
{
  CSystemInfo::mutex_lock();
  m_cluster_list.push_back(this);
  CSystemInfo::mutex_unlock();

  m_ml_array.resize(num_partition);

  std::vector<const int*>     csr_ridx_array(num_partition+1,csr_ridx);
  std::vector<unsigned>       ny_array(num_partition,0);
  std::vector<const double*>  y_array(num_partition,y);

  //make earlier partitions larger
  unsigned  partition_size  = ny/num_partition+1;

  for(unsigned i=1;i<num_partition;++i)
  {
    ny_array[i-1]     =   partition_size;    
    y_array[i]        =   y? (y_array[i-1]+partition_size):NULL;
    csr_ridx_array[i] =   csr_ridx_array[i-1]+partition_size;    
  }

  ny_array.back() = ny-(partition_size*(num_partition-1));
  csr_ridx_array.back()     = csr_ridx+ny;

  for(unsigned i=0;i<num_partition;++i)
  {
    const double* _data     =   data? (data+(*csr_ridx_array[i])):NULL;
    const int*    _csr_cidx =   csr_cidx+(*csr_ridx_array[i]);
    const unsigned  _nnz    =   (*csr_ridx_array[i+1])-(*csr_ridx_array[i]);

    assert(nx);
    assert(ny_array[i]);

    CSystemInfo::mutex_lock();

    int device  = CCuAcc::get_most_idle_device();
    CUDA_CHECK(cudaSetDevice(device));   //this needed for the CUDA calls in constructors

    switch(algo){
    case CMachineLearning::LINEAR_REGRESSION:
      m_ml_array[i]  =   new CLinearRegression(device,ny_array[i],nx,_data,y_array[i], csr_ridx_array[i],_csr_cidx,_nnz,intercept);  
      break;
    case CMachineLearning::LOGISTIC_REGRESSION:
      m_ml_array[i]  =   new CLogisticRegression(device,ny_array[i],nx,_data,y_array[i], csr_ridx_array[i],_csr_cidx,_nnz,intercept);  
      break;
    default:
      CSystemInfo::log("unsupported algorithm %d\n",algo);
      assert(false);
    }

    CSystemInfo::mutex_unlock();

    CSystemInfo::log("device=%d  nnz=%d ny_array=%d\n",device,nnz,ny_array[i]);
  }   
}

CCuAccCluster::CCuAccCluster(CMachineLearning::ALGO algo, const unsigned num_partition, const unsigned ny, const unsigned nx, const double* data, const double* y, const bool intercept)
  :m_nx(nx), m_ny(ny), m_cgrad(NULL), m_state(IDLE)
{
  CSystemInfo::mutex_lock();
  m_cluster_list.push_back(this);
  CSystemInfo::mutex_unlock();

  m_ml_array.resize(num_partition);

  std::vector<unsigned>       ny_array(num_partition,0);
  std::vector<const double*>  y_array(num_partition,y);

  unsigned  partition_size  = ny/num_partition+1;

  CSystemInfo::log("partition_size = %d\n",partition_size);

  for(unsigned i=1;i<num_partition;++i)
  {
    ny_array[i-1] = partition_size;
    y_array[i]        = y_array[i-1]+partition_size;
  }

  ny_array.back() = ny-(partition_size*(num_partition-1));

  unsigned  offset  = 0;
  for(unsigned i=0;i<num_partition;++i)
  {
    const double*   _data = data+offset;

    offset  +=  nx*ny_array[i];

    CSystemInfo::mutex_lock();

    int device  = CCuAcc::get_most_idle_device();
    CUDA_CHECK(cudaSetDevice(device));   //this needed for the CUDA calls in constructors

    switch(algo){
      //case CMachineLearning::LEAST_SQUARE:
      //  m_ml_array[i]  =   new CLeastSquareRegression(device,ny_array[i],nx,_data,y_array[i]);
      //  break;
    case CMachineLearning::LOGISTIC_REGRESSION:
      m_ml_array[i]  =   new CLogisticRegression(device,ny_array[i],nx,_data,y_array[i],intercept);
      break;
    default:
      assert(false);
    }

    CSystemInfo::mutex_unlock();
  }   
}

void CCuAccCluster::synchronize(int idx)
{
  CUDA_CHECK(cudaStreamSynchronize(m_ml_array[idx]->m_stream));
}

void CCuAccCluster::synchronize()
{
  const unsigned  num_partition = m_ml_array.size();  
  for(unsigned i=0;i<num_partition;++i)
    synchronize(i);
}

void CCuAccCluster::summarize(double weight_sum, double* data_sum, double* data_sq_sum, double* label_info)
{

  const unsigned  num_partition = m_ml_array.size();  

  synchronize();

  for(unsigned i=0;i<num_partition;++i)
  {
    CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device));

    m_ml_array[i]->m_p_matrix->summarize(weight_sum,
      m_ml_array[i]->m_p_dev_tmp_data_sum,
      m_ml_array[i]->m_p_dev_tmp_data_sq_sum,
      m_ml_array[i]->m_p_dev_tmp_label_info,
      m_ml_array[i]);
  }

  //honor existing values
  double  label_sum;
  for(unsigned i=0;i<num_partition;++i)
  {     
    synchronize(i);

    cblas_daxpy(m_nx,1,m_ml_array[i]->m_p_dev_tmp_data_sum->host(),1,data_sum,1);
    cblas_daxpy(m_nx,1,m_ml_array[i]->m_p_dev_tmp_data_sq_sum->host(),1,data_sq_sum,1);


    label_sum = *m_ml_array[i]->m_p_dev_tmp_label_info->host();

    //squared sum of label
    label_info[2]  +=  *(m_ml_array[i]->m_p_dev_tmp_label_info->host()+1);
    //this is the number of ones for binary classification, otherwise, just sum (to get mean)
    label_info[1]  +=  label_sum;
    //this is the number of zeros for binary classification
    label_info[0]  +=  m_ml_array[i]->m_p_matrix->m_dev_y.m_count-label_sum;
  }
}


void CCuAccCluster::weighten(double* weight, double* weight_sum, double* weight_nnz)
{
  const unsigned  num_partition = m_ml_array.size();  

  if(weight)
  {
    std::vector<double*>  w_array(num_partition,NULL);

    unsigned offset = 0;
    for(unsigned i=0;i<num_partition;++i)
    {
      w_array[i]  = weight+offset;
      offset  +=  m_ml_array[i]->m_p_matrix->m_nrow;
    }

    synchronize();

    for(unsigned i=0;i<num_partition;++i)
    {
      CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device));
      m_ml_array[i]->m_p_matrix->weighten(w_array[i],&m_ml_array[i]->m_dev_buf_xy1,m_ml_array[i]);
    }

    *weight_nnz = 0;
    for(unsigned i=0;i<m_ny;++i)
      *weight_nnz +=  weight[i]!=0;
   
    for(unsigned i=0;i<num_partition;++i)
    {
      synchronize(i);
      *weight_sum  +=  *w_array[i];
    }
  }
  else
  {    
    for(unsigned i=0;i<num_partition;++i)
      *weight_sum +=  m_ml_array[i]->m_p_matrix->m_nrow;

    *weight_nnz = *weight_sum;
  }
}

double CCuAccCluster::aug(double* weight, double intercept)
{
  const unsigned  num_partition = m_ml_array.size();  

  double* fx    = new double[m_ny];
  double* label = new double[m_ny]; 

  unsigned  offset  = 0;

  synchronize();

  for(unsigned i=0;i<num_partition;++i)
  {
    CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device));
    m_ml_array[i]->predict(weight,intercept,fx+offset,label+offset);

    offset  +=  m_ml_array[i]->m_p_matrix->m_nrow;
  }

  synchronize();

  double  ret = m_ml_array.front()->aug(label,fx,m_ny);

  delete[]  fx;
  delete[]  label;

  //printf("aug =%.16f\n",ret);

  return  ret;
}

double CCuAccCluster::rmse(double* weight, double intercept)
{
  const unsigned  num_partition = m_ml_array.size();  

  unsigned  max_ny  = m_ml_array[0]->m_p_matrix->m_nrow;
  for(unsigned i=1;i<num_partition;++i)
    max_ny  = max(max_ny,m_ml_array[i]->m_p_matrix->m_nrow);

  double* fx    = new double[max_ny+num_partition]; 
   
  synchronize();

  for(unsigned i=0;i<num_partition;++i)
  {
    CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device));
    m_ml_array[i]->sq_err(weight,intercept,fx+i,NULL);
  }

  synchronize();

  double  sum = fx[0];
  for(unsigned i=1;i<num_partition;++i)
    sum +=  fx[i];   

  delete[]  fx;

  //printf("rmse =%.16f\n",ret);

  return  sqrt(sum/m_ny);
}

void CCuAccCluster::standardize(double* data_mean, double* data_std, double label_mean, double label_std)
{
  const unsigned  num_partition = m_ml_array.size();  

  synchronize();

  for(unsigned i=0;i<num_partition;++i)
  {
    CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device));
    m_ml_array[i]->standardize(data_mean,data_std,label_mean,label_std);
  }
}

double CCuAccCluster::evaluate(unsigned size, double* x, double* cgrad, double weight_sum)
{
  assert(x);
  assert(cgrad);

  const unsigned  num_partition = m_ml_array.size();    

  synchronize();

  for(unsigned i=0;i<num_partition;++i)
  {
    CUDA_CHECK(cudaSetDevice(m_ml_array[i]->m_device)); 
    m_ml_array[i]->evaluate(x,weight_sum); 
  } 

  //honor existing cgrad
  double  cfx  = 0;
  for(unsigned i=0;i<num_partition;++i)
  {
    synchronize(i);

    cblas_daxpy(size,1,m_ml_array[i]->get_g(),1,cgrad,1);
    cfx  +=  m_ml_array[i]->get_fx();
  }

  //CSystemInfo::log("cfx=%f cgrad[0]=%f size=%d\n",cfx,cgrad[0],size);
  return  cfx;
}

void CCuAccCluster::solve_sgd(double weight_sum, CUpdater* updater, unsigned int max_iteration, double lambda, double step_size, double convergence_tol, double* x)
{
  CSystemInfo tm(__FUNCTION__);

  if(!m_cgrad) 
    m_cgrad = new double[m_nx+m_ml_array.front()->m_intercept];

  updater->initialize(m_nx,x,lambda);

  for(unsigned i=1;i<=max_iteration;++i)
  {      
    memset(m_cgrad,0,(m_nx+m_ml_array.front()->m_intercept)*sizeof(double));

    {
      double  nzx  = 0;
      for(unsigned i=0;i<m_nx;++i)
        if(x[i]==0) nzx++;

      printf("# of zero weights =%d out of %d\n",(int)nzx,m_nx);
    }
    double  fx = evaluate(m_nx+m_ml_array.front()->m_intercept,x,m_cgrad,weight_sum); 

    std::vector<double> convergence_info(2,1);

    double  regVal  = updater->update(x,m_cgrad,m_ny,step_size,i,lambda,&convergence_info.front());

    bool  converged = convergence_info[0] < convergence_tol * max(convergence_info[1], 1.0);

    if(m_ml_array.front()->m_intercept)
      printf("%s i=%d fx=%e converged=%d loss=%e cgrad(0)=%e x(0)=%e regVal=%e intercept=%e\n",
      "GPU", i, fx, converged,fx/weight_sum, m_cgrad[0], x[0], regVal,m_cgrad[m_nx]);
    else
      printf("%s i=%d fx=%e converged=%d loss=%e cgrad(0)=%e x(0)=%e regVal=%e\n",
      "GPU", i, fx, converged,fx/weight_sum, m_cgrad[0], x[0], regVal);

//    double test_X[] = {
//-0.006838634518180606,-0.04457094511877472,0.0,0.07773612956391085,0.020533723356241724,0.0,0.0,0.022182386253426847,0.09301926482022636,0.015388873442756919,-0.0055706686315458625,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2144022892015011,0.0,0.07630180403429483,-0.041239465967306106,0.0,0.0,0.054897525742876915,0.0,0.0,0.3234479467804168,-0.09985496661607271,0.0,-0.05912154214988861,0.0,0.0,0.0,0.15645030060625612,0.27932668102340386,0.0,0.0,0.0,0.0,0.0,0.015353470498619775,0.03626368315145928,-4.705486802913351E-4,-0.004311330923063878,0.0,0.15268269392225883,0.0,0.0052617150790345985,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01576293865701875,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.11983673271431113,0.0,0.0,0.0,0.0,-0.1211005227086926,0.1211005227087023,-0.0646176630433938,0.06461766304339668,-0.02892466364285392,0.0,-0.002802605387570024,0.026113589272167094,0.02955739713306733,0.00209330814976037,0.0,-0.03139374015546418,-0.03535609886258558,0.0,0.0,0.0,0.0,0.5200839381282015,0.0,0.0,-0.04422023506672689,0.0,0.0,0.0,0.049975552753353254,0.15381322000562847,0.0,0.0,0.0,0.0,0.0,-0.11695772237442635,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.1230812559815254,0.0,0.0,0.0,0.0,0.38527907409563156,0.0
//    };
//
//    this->rmse(test_X,0.19676101393228124);

    if(converged) break;
  }       
}

#ifdef _ENABLE_LBFGS_

#include "lbfgs\lbfgs.h"

static lbfgsfloatval_t evaluate(
  void *instance,
  const lbfgsfloatval_t *x,
  lbfgsfloatval_t *g,
  const int n,
  const lbfgsfloatval_t step
  )
{
  int i;
  lbfgsfloatval_t fx = 0.0;

  CCuMatrix*  func  = (CCuMatrix*)instance;

  func->evaluate(x);
  CUDA_CHECK(cudaStreamSynchronize(func->m_stream));

  memcpy(g,func->get_g(),n*sizeof(double));

  return  func->get_fx();

  //for (i = 0;i < n;i += 2) {
  //    lbfgsfloatval_t t1 = 1.0 - x[i];
  //    lbfgsfloatval_t t2 = 10.0 * (x[i+1] - x[i] * x[i]);
  //    g[i+1] = 20.0 * t2;
  //    g[i] = -2.0 * (x[i] * g[i+1] + t1);
  //    fx += t1 * t1 + t2 * t2;
  //}
  //return fx;
}

static int progress(
  void *instance,
  const lbfgsfloatval_t *x,
  const lbfgsfloatval_t *g,
  const lbfgsfloatval_t fx,
  const lbfgsfloatval_t xnorm,
  const lbfgsfloatval_t gnorm,
  const lbfgsfloatval_t step,
  int n,
  int k,
  int ls
  )
{
  printf("Iteration %d:\n", k);
  printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
  printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
  printf("\n");
  return 0;
}

void  CCuAccCluster::solve_lbfgs(CUpdater* updater, unsigned int max_iteration, double lambda, double step_size, double convergence_tol, double* x)
{
  assert(false);

  CSystemInfo tm(__FUNCTION__);

  lbfgs_parameter_t param;

  lbfgs_parameter_init(&param);
  param.max_iterations  = 20;

  if(!m_cgrad) 
    m_cgrad = new double[m_nx];





  double* new_x = new double[m_nx*m_ml_array.size()];


  for(unsigned i=1;i<=max_iteration;++i)
  {      
    if(i==1)
    {
      double  fx_check  = 0;
      for(unsigned j=0,s=m_ml_array.size();j<s;++j)
      {
        double* cur_x = new_x;

        memcpy(cur_x,x,m_nx*sizeof(double));

        double  fx  = -1;
        int ret = lbfgs(m_nx, new_x+j*m_nx, &fx, evaluate, NULL, m_ml_array[j], &param);

        fx_check  +=  fx;
      }

      for(unsigned k=0;k<m_nx;++k)
      {
        x[k] =0;
        for(unsigned j=0,s=m_ml_array.size();j<s;++j)
        {
          x[k]  = (new_x+j*m_nx)[k];
        }

        x[k]  /=  m_ml_array.size();
      }

      updater->initialize(m_nx,x,lambda);
      //memset(m_cgrad,0,m_nx*sizeof(double));

      //double  fx = evaluate(x,m_cgrad);

      //std::vector<double> convergence_info(2,1);

      //double  regVal  = updater->update(x,m_cgrad,m_ny,step_size,i,lambda,&convergence_info.front());

      //bool  converged = convergence_info[0] < convergence_tol * max(convergence_info[1], 1.0);

      //printf("%s i=%d fx=%e converged=%d cgrad(0)=%e x(0)=%e regVal=%e\n",
      //  "GPU", i, fx, converged, m_cgrad[0], x[0], regVal);

      //if(converged) break;
    }
    else
    {
      memset(m_cgrad,0,m_nx*sizeof(double));

      double  fx = evaluate(x,m_cgrad);

      std::vector<double> convergence_info(2,1);

      double  regVal  = updater->update(x,m_cgrad,m_ny,step_size,i,lambda,&convergence_info.front());

      bool  converged = convergence_info[0] < convergence_tol * max(convergence_info[1], 1.0);

      printf("%s i=%d fx=%e converged=%d cgrad(0)=%e x(0)=%e regVal=%e\n",
        "GPU", i, fx, converged, m_cgrad[0], x[0], regVal);

      if(converged) break;
    }
  }  

}

#endif