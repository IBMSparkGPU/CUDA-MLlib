# CuMF: CUDA-Acclerated ALS on mulitple GPUs. 

This folder contains:

(1) the CUDA kernel code implementing ALS (alternating least square), and 

(2) the JNI code to link to and accelerate the ALS.scala program in Spark MLlib.

## Technical details
By optimizing memory access and parallelism, cuMF is much faster and cost-efficient compared with state-of-art CPU based solutions. 

More details can be found at:

1) This Nvidia GTC 2016 talk
ppt:

<http://www.slideshare.net/tanwei/s6211-cumf-largescale-matrix-factorization-on-just-one-machine-with-gpus>

video:

<http://on-demand.gputechconf.com/gtc/2016/video/S6211.html>

2) This HPDC 2016 paper: 

"Faster and Cheaper: Parallelizing Large-Scale Matrix Factorization on GPUs"
<http://arxiv.org/abs/1603.03820>

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