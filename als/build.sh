#!/bin/bash

if [ "$1" = "help" ]; then
  echo ""
  echo "Use this script to build the JNI .cpp code and CUDA code for ALS, optionally Spark too"
  echo ""
  echo "build.sh spark 2.5 -> only build Spark with profiles hardcoded here (Hadoop 2.5)"
  echo ""
  echo "build.sh spark 2.6 dist -> build Spark and create the distribution package (Hadoop 2.6)"
  echo ""
  echo "build.sh spark 2.7 dist tgz -> build Spark, create the distribution package, tarball and zip it (Hadoop 2.7)"
  echo ""
fi

if [ -z ${JAVA_HOME} ]; then
  echo "Please set your JAVA_HOME to point to your Java installation"
  exit 1
fi

if [ -z ${CUDA_HOME} ]; then
  echo "Please set your CUDA_HOME to point to your CUDA installation e.g. /usr/local/cuda"
  exit 1
fi

echo "Using JAVA_HOME: $JAVA_HOME"
echo "Using CUDA_HOME: $CUDA_HOME"
echo "Compiling the CUDA and native code"

$CUDA_HOME/bin/nvcc -shared  -D_USE_GPU_ -I/usr/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux ../utilities.cu src/cuda/als.cu src/CuMFJNIInterface.cpp -o libGPUALS.so -Xcompiler "-fPIC" -m64  -use_fast_math -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -O3 -Xptxas -dlcm=ca -L{$CUDA_ROOT}/lib64 -lcublas -lcusparse    

if [ "$1" = "spark" ]; then
  SPARK_HOME=../../SparkGPU/
  echo "Building Spark from $SPARK_HOME, will include profiles for Yarn, Hadoop, Hive, Hive-Thriftserver by default, edit this to override (e.g. for SparkR, Kinesis)"
  cd $SPARK_HOME
  # Prevents OoM issues on IBM Power LE and JDK 8
  export MAVEN_OPTS="-Xmx4g"
  PROFILES="-Pyarn -Phadoop-$2 -Phive -Phive-thriftserver"
  # -T 1C means: run with multiple threads, one per core, this is OK for Spark
  build/mvn -T 1C $PROFILES -DskipTests package
  # Should we create the distribution package?
  if [ "$3" = "dist" ]; then
    # Should we tarball and zip it?
    if [ "$4" = "tgz" ]; then
      dev/make-distribution.sh $PROFILES --tgz
    else 
      dev/make-distribution.sh $PROFILES
    fi
  fi
fi
