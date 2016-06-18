#ifndef __CUACC_UPDATER_H__
#define __CUACC_UPDATER_H__

#include "cuAcc_base.h"

class CAdaDelta {
public:
  CCuMemory<double>   m_dev_avg_squared_g;
  CCuMemory<double>   m_dev_avg_squared_delta_x;
   
  //delta history
  void initialize(unsigned nx)
  {
    m_dev_avg_squared_g.resize(nx);
    m_dev_avg_squared_delta_x.resize(nx);

    m_dev_avg_squared_g.memset(0);
    m_dev_avg_squared_delta_x.memset(0); 
  }

  virtual ~CAdaDelta(){}
};

class CUpdater : public CCuAcc {
public:
  enum ALGO {
    L1_UPDATER,
    L1_ADADELTA_UPDATER,
    L2_UPDATER,
    L2_ADADELTA_UPDATER    
  };

  CCuMemory<double>     m_dev_prev_x;
  unsigned              m_nx;

  virtual double  initialize(unsigned nx,double* x,double lambda) = 0;
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info) = 0;

  void convergence_check(double* dev_x, double* convergence_info);
  void convergence_check(CCuMemory<double>& dev_x, double* convergence_info);

  static void  write(const char* filename, const unsigned nx, const double* previousWeights, const double* x, const double* cgrad, long mini_batch_size, double step_size, int iter, double lambda);
  static void  read(const char* filename, unsigned& nx, double*&previousWeights, double*&x, double*&cgrad, long& mini_batch_size, double& step_size, int& iter, double& lambda);

  CUpdater() : CCuAcc(0){}

  virtual ~CUpdater(){}

};

class CSimpleUpdater  : public CUpdater {
public:
  CSimpleUpdater() : CUpdater(){}

  virtual ~CSimpleUpdater(){}

  virtual double  initialize(unsigned nx,double* x,double lambda);
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info);
};

class CL1Updater  : public CUpdater {
public:
  CL1Updater() : CUpdater(){}

  virtual ~CL1Updater(){}

  virtual double  initialize(unsigned nx,double* x,double lambda);
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info);

  double  compute_convergence_regularization(CCuMemory<double>& dev_x, int iter, double lambda, double*& convergence_info);
};

class CL1AdaDeltaUpdater : public CL1Updater, public CAdaDelta {
public:
  CL1AdaDeltaUpdater() : CL1Updater(){}

  virtual ~CL1AdaDeltaUpdater(){}

  virtual double  initialize(unsigned nx,double* x,double lambda);
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info);
};

class CL2Updater : public CUpdater {
public:
  CL2Updater() : CUpdater(){}

  virtual ~CL2Updater(){}

  virtual double  initialize(unsigned nx,double* x,double lambda);
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info);
  
  double  compute_convergence_regularization(CCuMemory<double>& dev_weight, int iter, double lambda, double*& convergence_info);
};

class CL2AdaDeltaUpdater : public CL2Updater, public CAdaDelta {
public:
  CL2AdaDeltaUpdater() : CL2Updater(){}

  virtual ~CL2AdaDeltaUpdater(){}

  virtual double  initialize(unsigned nx,double* x,double lambda);
  virtual double  update(double* x, double* cgrad, long mini_batch_size, double step_size, int iter, double lambda, double* convergence_info);
};

#endif