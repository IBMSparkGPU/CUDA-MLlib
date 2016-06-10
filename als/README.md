# CuMF: CUDA-Acclerated ALS on mulitple GPUs. 

This folder contains:

(1) the CUDA kernel code implementing ALS (alternating least square), and 

(2) the JNI code to link to and accelerate the ALS.scala program in Spark MLlib.

## What is matrix factorization?

Matrix factorization (MF) factors a sparse rating matrix R (m by n, with N_z non-zero elements) into a m-by-f and a f-by-n matrices, as shown below.

<img src=https://github.com/wei-tan/CUDA-MLlib/raw/master/als/images/mf.png width=444 height=223 />
 
Matrix factorization (MF) is at the core of many popular algorithms, e.g., [collaborative filtering](https://en.wikipedia.org/wiki/Collaborative_filtering), word embedding, and topic model. GPU (graphics processing units) with massive cores and high intra-chip memory bandwidth sheds light on accelerating MF much further when appropriately exploiting its architectural characteristics.

## What is cuMF?

**CuMF** is a CUDA-based matrix factorization library that optimizes alternate least square (ALS) method to solve very large-scale MF. CuMF uses a set of techniques to maximize the performance on single and multiple GPUs. These techniques include smart access of sparse data leveraging GPU memory hierarchy, using data parallelism in conjunction with model parallelism, minimizing the communication overhead among GPUs, and a novel topology-aware parallel reduction scheme.

With only a single machine with four Nvidia GPU cards, cuMF can be 6-10 times as fast, and 33-100 times as cost-efficient, compared with the state-of-art distributed CPU solutions. Moreover, cuMF can solve the largest matrix factorization problem ever reported yet in current literature. 

CuMF achieves excellent scalability and performance by innovatively applying the following techniques on GPUs:  

(1) On a single GPU, MF deals with sparse matrices, which makes it difficult to utilize GPU's compute power. We optimize memory access in ALS by various techniques including reducing discontiguous memory access, retaining hotspot variables in faster memory, and aggressively using registers. By this means cuMF gets closer to the roofline performance of a single GPU. 

(2) On multiple GPUs, we add data parallelism to ALS's inherent model parallelism. Data parallelism needs a faster reduction operation among GPUs, leading to (3).

(3) We also develop an innovative topology-aware, parallel reduction method to fully leverage the bandwidth between GPUs. By this means cuMF ensures that multiple GPUs are efficiently utilized simultaneously.

## Use cuMF to accelerate Spark ALS

CuMF can be used standalone, or to accelerate the [ALS implementation in Spark MLlib](https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala).

We modified Spark's ml/recommendation/als.scala ([code](https://github.com/wei-tan/SparkGPU/blob/MLlib/mllib/src/main/scala/org/apache/spark/ml/recommendation/ALS.scala)) to detect GPU and offload the ALS forming and solving to GPUs, while retain shuffling on Spark RDD. 

<img src=https://github.com/wei-tan/CUDA-MLlib/raw/master/als/images/spark-gpu.png width=380 height=240 />

This approach has several advantages. First, existing Spark applications relying on mllib/ALS need no change. Second, we leverage the best of Spark (to scale-out to multiple nodes) and GPU (to scale-up in one node).

## Build
There are scripts to build the program locally, run in local mode, and run in distributed mode.  
You should modify the first line or two to point to your own instalation of Java, Spark, Scala and data files.

To build, first set $CUDA_HOME to your CUDA installation (e.g., /usr/local/cuda) and $JAVA_HOME to your JDK (not a JRE, we need the JDK to build the jni code).

Then run:
	build.sh help to see the options you have available, for example
	build.sh spark 2.5 -> only build Spark with profiles hardcoded here (Hadoop 2.5)
        build.sh spark 2.6 dist -> build Spark and create the distribution package (Hadoop 2.6)
        build.sh spark 2.7 dist tgz -> build Spark, create the distribution package, tarball and zip it (Hadoop 2.7)

## Run

To run, first set your $SPARK_HOME and $JAVA_HOME and then execute runALS.sh.

runALS.sh accepts several parameters, you'll want to specify the execution mode (with GPUs or CPUs only), the master URL (e.g. spark://foo.com:7077 or local[12]), the rank, how many iterations to run with, the lambda, and finally the data set to use.

For example:

	runLocal.sh gpu local[12] 100 12 0.058 $SPARK_HOME/netflix.data

Note: rank value must be a multiple of 10.
For picking a lambda, this is best achieved with trial and error; starting with something roughly 1 is useful, in our Netflix case we know the ideal lambda value is around 0.05.

We look for the following system property:

	spark.mllib.ALS.useGPU
	
and you can set this by adding the following to your spark-submit command line (assuming the shared library is at lib/ibm under your $SPARK_HOME folder, we allow you to choose any location):

	--conf spark.mllib.ALS.useGPU=$SPARK_HOME/lib/ibm/libGPUALS.so

## Known Issues
We are trying to improve the usability, stability and performance. Here are some known issues:

(1) Out-of-memory error from GPUs, when there are many CPU threads accessing a small number of GPUs on any node. We tested Netflix data on one node, with 12 CPU cores used by the executor, and 2 Nvidia K40 GPU cards. If you have more GPU cards, you may be able to accomodate more CPU cores/threads. Otherwise you need to lessen the #cores assigned to Spark executor.

(2) CPU-GPU hybrid execution. We want to push as much workload to GPU as possible. If GPUs cannot accomodate all CPU threads, we want to retain the execution on CPUs. 

(3) Currently the user is responsible for partitioning the data correctly (try not to give the GPU too much or too little work, there's usually a sweet spot for GPU offloading) and handling any errors.

## References

More details can be found at:

1) CuMF: Large-Scale Matrix Factorization on Just One Machine with GPUs. Nvidia GTC 2016 talk. [ppt](http://www.slideshare.net/tanwei/s6211-cumf-largescale-matrix-factorization-on-just-one-machine-with-gpus), [video](http://on-demand.gputechconf.com/gtc/2016/video/S6211.html)

2) Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs. Wei Tan, Liangliang Cao, Liana Fong. [HPDC 2016](http://arxiv.org/abs/1603.03820), Kyoto, Japan
