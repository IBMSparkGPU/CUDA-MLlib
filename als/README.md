# CuMF: CUDA-Acclerated ALS on mulitple GPUs. 

This folder contains:

(1) the CUDA kernel code implementing ALS (alternating least square), and 

(2) the JNI code to link to and accelerate the ALS.scala program in Spark MLlib.

## What is matrix factorization?

Matrix factorization (MF) factors a sparse rating matrix R (m by n, with N_z non-zero elements) into a m-by-f and a f-by-n matrices, as shown below.

![alt text](https://github.com/wei-tan/CUDA-MLlib/raw/master/als/images/mf.png "matrix factorization")
 
Matrix factorization (MF) is at the core of many popular algorithms, e.g., [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), word embedding, and topic model. GPU (graphics processing units) with massive cores and high intra-chip memory bandwidth sheds light on accelerating MF much further when appropriately exploiting its architectural characteristics.

## What is cuMF?

**CuMF** is a CUDA-based matrix factorization library that optimizes alternate least square (ALS) method to solve very large-scale MF. CuMF uses a set of techniques to maximize the performance on single and multiple GPUs. These techniques include smart access of sparse data leveraging GPU memory hierarchy, using data parallelism in conjunction with model parallelism, minimizing the communication overhead among GPUs, and a novel topology-aware parallel reduction scheme.

With only a single machine with four Nvidia GPU cards, cuMF can be 6-10 times as fast, and 33-100 times as cost-efficient, compared with the state-of-art distributed CPU solutions. Moreover, cuMF can solve the largest matrix factorization problem ever reported yet in current literature. CuMF can be used standalone, or to accelerate the [ALS implementation in Spark MLlib](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala).

CuMF achieves excellent scalability and performance by innovatively applying the following techniques on GPUs:  

(1) On a single GPU, MF deals with sparse matrices, which makes it difficult to utilize GPU's compute power. We optimize memory access in ALS by various techniques including reducing discontiguous memory access, retaining hotspot variables in faster memory, and aggressively using registers. By this means cuMF gets closer to the roofline performance of a single GPU. 

(2) On multiple GPUs, we add data parallelism to ALS's inherent model parallelism. Data parallelism needs a faster reduction operation among GPUs, leading to (3).

(3) We also develop an innovative topology-aware, parallel reduction method to fully leverage the bandwidth between GPUs. By this means cuMF ensures that multiple GPUs are efficiently utilized simultaneously.


## Build
There are scripts to build the program locally, run in local mode, and run in distributed mode.  
You should modify the first line or two to point to your own instalation of Java, Spark, Scala and data files.

To build, first set $CUDA_ROOT to your cuda installation (e.g., /usr/local/cuda) and $JAVA_HOME to your JDK (not JRE, we need JDK to build the jni code).

Then run:

	build.sh


## Run

To run, first set $SPARK_HOME and $JAVA_HOME. 

To submit to a local Spark installation:
Run runLocal.sh, specifying the mode (gpu or cpu), #cores, the lambda, and the rank. Prepare a data file and put its name after "--kryo" in the runLocal.sh script.

Note: rank value has to be a multiply of 10, e.g., 10, 50, 100, 200). For example:

	./runLocal.sh gpu 12 0.058 100

## Known Issues
We are trying to improve the usability, stability and performance. Here are some known issues:

(1) Out-of-memory error from GPUs, when there are many CPU threads accessing a small number of GPUs on any node. We tested Netflix data on one node, with 12 CPU cores used by the executor, and 2 Nvidia K40 GPU cards. If you have more GPU cards, you may be able to accomodate more CPU cores/threads. Otherwise you need to lessen the #cores assigned to Spark executor.

(2) CPU-GPU hybrid execution. We want to push as much workload to GPU as possible. If GPUs cannot accomodate all CPU threads, we want to retain the execution on CPUs.

## References

More details can be found at:

1) CuMF: Large-Scale Matrix Factorization on Just One Machine with GPUs. Nvidia GTC 2016 talk. [ppt](http://www.slideshare.net/tanwei/s6211-cumf-largescale-matrix-factorization-on-just-one-machine-with-gpus)[video](http://on-demand.gputechconf.com/gtc/2016/video/S6211.html)

2) Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. Wei Tan, Liangliang Cao, Liana Fong. [HPDC 2016](http://arxiv.org/abs/1603.03820), Kyoto, Japan