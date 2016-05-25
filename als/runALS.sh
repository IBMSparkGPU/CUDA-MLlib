#!/bin/bash

# Runs the MovieLensALS application with netflix data, this script
# allows us to easily modify GPU usage

# Check input arguments, we want five and cpu or gpu specified
# the input format is the same as it is for the MovieLens example
# except with the addition of the cpu/gpu parameter and the file name to use
# arg 1 = execution method e.g. cpu or gpu
# arg 2 = master e.g. local[*] or spark://foo.com:7077
# arg 3 = rank e.g. 100
# arg 4 = numIterations e.g. 10
# arg 5 = lambda e.g. 0.058
# arg 6 = data to use e.g. $SPARK_HOME/netflix.data

if [[ ( "$1" != "cpu" && "$1" != "gpu" ) || "$#" -ne 6 ]]; then
  echo "Usage: <cpu|gpu> <master> <rank> <numIterations> <lambda> <fileToUse>" >&2
  echo "e.g. $SPARK_HOME/../CUDA-MLlib/als/runLocal.sh gpu local[12] 100 10 0.058 $SPARK_HOME/netflix.data"
  exit 1
fi

gpu=""
if [ "$1" = "gpu" ]; then
  gpu="--conf spark.mllib.ALS.useGPU=$SPARK_HOME/../CUDA-MLlib/als/libGPUALS.so"
  echo "Using gpu library: $gpu"
fi

# For the netflix sample data we know ideal parameters are
# rank 100, numIterations 10, lambda 0.058
$SPARK_HOME/bin/spark-submit --class org.apache.spark.examples.mllib.MovieLensALS $gpu \
  --master $2 \
  --jars $SPARK_HOME/examples/target/scala-2.11/jars/spark-examples*.jar \
         $SPARK_HOME/examples/target/scala-2.11/jars/scopt*.jar \
    --rank $3 --numIterations $4 --lambda $5 --kryo $6
