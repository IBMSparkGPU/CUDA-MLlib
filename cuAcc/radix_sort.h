#ifndef __RADIX_SORT_H__
#define __RADIX_SORT_H__

#include <assert.h>
#include <algorithm>


#define _ENABLE_PREFETCH_
#ifdef _ENABLE_PREFETCH_
#ifdef _MSC_VER
#include <xmmintrin.h>
#define PREFETCH(addr)   _mm_prefetch((char*)(addr),_MM_HINT_T0)
#else 
#define PREFETCH(addr)  __builtin_prefetch((addr),1,1)
#endif
#else
#define PREFETCH(addr)
#endif


#define BUCKET_BIT    (8)
#define BUCKET_SIZE   (1<<BUCKET_BIT) 
#define BUCKET_MASK   (BUCKET_SIZE-1)

template<typename KT>
unsigned get_key_udp(const KT* key, const int shift)
{
  if(sizeof(KT)==4)
  {
    unsigned long _key = *(unsigned long*)key;
    return (_key>>shift)&BUCKET_MASK;  
  }
  else
  {
    unsigned long long _key = *(unsigned long long*)key;
    return (_key>>shift)&BUCKET_MASK;
  }
}

template<typename KT, typename DT>
void  sort5(KT* key, DT* data, const size_t size, const int e0=0, const int e1=1, const int e2=2, const int e3=3, const int e4=4)
{
  if (key[e1] < key[e0]) 
  {
    std::swap(key[e1],key[e0]);
    std::swap(data[e1],data[e0]);
  }

  if(size==2) return;

  KT  k;
  DT  d;

  k = key[e2];
  d = data[e2];
  if (k < key[e1]) 
  {
    key[e2]   = key[e1]; 
    data[e2]  = data[e1]; 
    if (k < key[e0])
    { 
      key[e1]   = key[e0];
      key[e0]   = k; 
      data[e1]  = data[e0];
      data[e0]  = d; 
    }
    else
    {
      key[e1]   = k;
      data[e1]  = d;
    }
  }

  if(size==3) return;

  k = key[e3];
  d = data[e3];
  if (k < key[e2])
  { 
    key[e3]   = key[e2];
    data[e3]  = data[e2];
    if (k < key[e1]) 
    {
      key[e2]   = key[e1];
      data[e2]  = data[e1];
      if (k < key[e0]) 
      { 
        key[e1]   = key[e0]; 
        key[e0]   = k; 
        data[e1]  = data[e0]; 
        data[e0]  = d; 
      }
      else
      {
        key[e1]   = k;
        data[e1]  = d;
      }
    }
    else
    {
      key[e2]   = k;
      data[e2]  = d;
    }
  }

  if(size==4) return;

  k = key[e4];
  d = data[e4];
  if (k < key[e3]) 
  {
    key[e4]   = key[e3];
    data[e4]  = data[e3];
    if (k < key[e2]) 
    {
      key[e3]   = key[e2];     
      data[e3]  = data[e2];
      if (k < key[e1]) 
      {
        key[e2]   = key[e1];
        data[e2]  = data[e1];
        if (k < key[e0]) 
        { 
          key[e1]   = key[e0]; 
          key[e0]   = k; 
          data[e1]  = data[e0]; 
          data[e0]  = d; 
        }
        else
        {
          key[e1]   = k;
          data[e1]  = d;
        }
      }
      else
      {
        key[e2]   = k;
        data[e2]  = d;
      }
    }
    else
    {
      key[e3]   = k;
      data[e3]  = d;
    }
  }
}
 
template<typename KT, typename DT>
void  insertion_sort(KT* key, DT* data, const size_t size)
{
  KT  k;
  DT  d;

  unsigned src,dst;

  for(src = 1; src < size; src++)
  {
    k = key[src];
    d = data[src];

    for(dst=src; dst-- > 0 && k < key[dst]; );
    /* when we get out of the loop, 
    ** table[dst] is the element BEFORE our insertion point,
    ** so re-increment it to point at the insertion point */
    if (++dst == src) continue;

    memmove(key+dst+1,key+dst, (src-dst) * sizeof(KT));
    memmove(data+dst+1,data+dst, (src-dst) * sizeof(DT));
    
    key[dst]  = k;
    data[dst] = d;
  }
}

template<typename KT, typename DT>
void radix_sort(KT* key, DT* data, const size_t size, int shift=(sizeof(KT)-1)*BUCKET_BIT)
{
  size_t  head[BUCKET_SIZE];
  size_t  first[BUCKET_SIZE+1];
  size_t  count[BUCKET_SIZE] = {0,}; 
  size_t* last = first+1;
  size_t  i,id;
  KT      k;
  DT      d;

  for (i=0;i<size;++i) 
    ++count[get_key_udp<KT>(key+i,shift)];

  head[0]  = 0;
  memcpy(head+1,count,sizeof(count)-sizeof(size_t));  //head -> has the start address

  for (i=1;i<BUCKET_SIZE;++i)   //build histogram for a given bucket
  {
    head[i] +=  head[i-1];      //last -> has the last address  

    if(head[i]==size)
      break;    

    PREFETCH(key+head[i]);
  }

  const unsigned last_bucket  = i;
  memcpy(first,head,sizeof(size_t)*last_bucket); 

  size_t  begin,end; 
  for (i=0;i<last_bucket-1;++i) 
  {
    for(begin=head[i],end=last[i];begin!=end;) 
    {
      assert(begin<end);

      k = key[begin];    
      d = data[begin];

      id= get_key_udp<KT>(&k,shift);
      assert(id<BUCKET_SIZE);

      if(i==id)
        begin++;
      else
      {
        assert(id>i);
        do
        {
          //keep moving until you hit the right bucket
          std::swap(d,data[head[id]]);
          std::swap(k,key[head[id]]);

          head[id]++;

          PREFETCH(key+head[id]);
          PREFETCH(data+head[id]);

          id= get_key_udp<KT>(&k,shift);
        }
        while(i!=id);

        //done with this key, and proceed for the next
        //cannot be id as it might break previously
        key[begin]  = k;
        data[begin] = d;
        begin++;
      }        
    }
  }

  if(shift>=BUCKET_BIT)
  {
    shift -=  BUCKET_BIT;

    for (i=0;i<last_bucket;++i) 
    {
      size_t cur_size = count[i];

      if(cur_size<=1)  
        continue;
      else if(cur_size<=5) 
        sort5<KT,DT>(key+first[i],data+first[i],cur_size);
      else if(cur_size<=64)
        insertion_sort<KT,DT>(key+first[i],data+first[i],cur_size);
      else
        radix_sort<KT,DT>(key+first[i],data+first[i],cur_size,shift);        
    }
  }
}

#endif