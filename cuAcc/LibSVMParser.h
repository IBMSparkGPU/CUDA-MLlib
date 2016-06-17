#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>

 

template<class T>
class CLibSVMParser
{
public:
  CLibSVMParser(){}

  std::vector<int>  m_csr_ridx;
  std::vector<int>  m_csr_cidx;
  std::vector<T>    m_csr_data;
  std::vector<T>    m_y;
  std::vector<T>    m_w;

  std::vector<T>    m_data;

  unsigned m_max_row;
  unsigned m_min_row;
  std::vector<unsigned> m_covered_by_X;

  int m_nx;
  
  unsigned get_nnz()         { return  m_csr_ridx.back(); }
  unsigned get_ny()  { return  m_csr_ridx.size()-1;}
  unsigned get_nx() { return  m_nx;}

  int*  get_csr_ridx()  { return  &m_csr_ridx.front();}
  int*  get_csr_cidx()  { return  &m_csr_cidx.front();}
  T*    get_csr_data()  { return  &m_csr_data.front();}
  T*    get_y() { return  &m_y.front();}
  T*    get_w() { return  m_w.empty()? NULL:&m_w.front();}
  T*    get_data()
  {
    if(m_data.empty())
    {
      m_data.resize(get_ny()*get_nx(),0);

      int row = 0;
      int col = 0;
      T   val;

      for(unsigned i=0,s=get_nnz();i<s;++i)
      {
        col = m_csr_cidx[i];
        val = m_csr_data[i];

        m_data[row*get_nx()+col]  = val;

        if(i+1==m_csr_ridx[row+1])  row++;
      }
    }

    return  &m_data.front();
  }

  void  read_rdd(const char* filename)
  {
    std::ifstream ifs;

    ifs.open(filename,std::ifstream::in);

    m_nx = 1;
    int nnz = 0;
    m_csr_ridx.push_back(nnz);

    std::string line;
    while(std::getline(ifs, line)) 
    { 
      char* pTok=strtok((char*)line.c_str(),",\n\t");      
      m_y.push_back(atof(pTok+1));

      pTok=strtok(NULL,",\n\t");   
      m_w.push_back(atof(pTok));

      pTok=strtok(NULL,",\n\t");   
      m_nx  = atoi(pTok+1);

      while(true)
      {
        pTok=strtok(NULL,",\n\t");   

        T data  = 1;
        int csr_cidx;
        if(pTok[0]=='[')
        {
          csr_cidx  = atoi(pTok+1);
        }
        else
          csr_cidx  = atoi(pTok);

        m_csr_cidx.push_back(csr_cidx);
        m_csr_data.push_back(data);

        nnz++;

        if(pTok[strlen(pTok)-1]==']')
          break;

      }

        m_csr_ridx.push_back(nnz);
    }
  }

  void  read_libsvm(const char* filename)
  {
    std::ifstream ifs;

    ifs.open(filename,std::ifstream::in);

    m_covered_by_X.resize(8,0);

    m_max_row  = 0;
    m_min_row  = -1;
    m_nx = 1;
    int nnz = 0;
    m_csr_ridx.push_back(nnz);

    std::string line;
    while(std::getline(ifs, line)) 
    { 
      int cur_nnz = nnz;
      //printf("%s\n",line.c_str());
      char* pTok=strtok((char*)line.c_str()," \n\t");      
      m_y.push_back(atof(pTok));

      for(pTok=strtok(NULL," :\t");pTok;pTok=strtok(NULL," :\n\t"))
      {
        int csr_cidx  = atoi(pTok)-1;

        m_nx = max(m_nx,csr_cidx);

        pTok=strtok(NULL," ");

        T data  = atof(pTok);

        m_csr_cidx.push_back(csr_cidx);
        m_csr_data.push_back(data);

        nnz++;
      }

      m_csr_ridx.push_back(nnz);

      int cur_col_size = nnz-cur_nnz;

      m_max_row = max(m_max_row,cur_col_size);  

      if(cur_col_size<=64)
      {
        for(unsigned j = cur_col_size/8;j<m_covered_by_X.size();++j)
          m_covered_by_X[j]++;
      }
    }

    //from zero to one-based
    m_nx++;

    ifs.close();

    
    //printf("max_nnz_row = %d/%d (%f)\n",m_max_row,m_nx,double(m_max_row)/m_nx);
    //printf("avg_nnz_row = %f\n",double(nnz)/get_ny())
    //for(int i=0;i<m_covered_by_X.size();++i);

    //printf("m_covered_by_%d=%d (%f)\n", (i+1)*8, m_covered_by_X[i],double(m_covered_by_X[i])/get_ny());
  }
};

