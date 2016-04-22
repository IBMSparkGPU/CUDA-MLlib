#wtan: Runs the MovieLensALS application with netflix data in local mode
#please read the TODOs and change the environment configurations.

#!/bin/bash
#check input arguments
if [[ ( "$1" != "cpu" && "$1" != "gpu" ) || "$#" -ne 4 ]]; then
  echo "Usage: cpu|gpu #cores lambda rank" >&2
  echo "e.g., ./runLocal.sh gpu 12 0.058 100"
  exit 1
fi

#TODO modify spark.ALS.useGPU= to point to your libGPUALS.so
gpu=""
if [ "$1" = "gpu" ]; then
	gpu="--conf spark.mllib.ALS.useGPU=/u/weitan/gpfs/github/CUDA-MLlib/als/libGPUALS.so"
	echo "using gpu libaray: "
	echo $gpu
fi

#TODO modify to point to your java and spark folders
JAVA_HOME=/u/weitan/gpfs/spark/jdk1.8.0_72
SPARK_HOME=/u/weitan/gpfs/github/SparkGPU/dist

# TODO: modify "/u/weitan/gpfs/spark/data/netflix.data" to point to your own data file
# --jars has to be before the application jar
$SPARK_HOME/bin/spark-submit --driver-memory 128g --class org.apache.spark.examples.mllib.MovieLensALS $gpu --master local[$2] --jars $SPARK_HOME/examples/jars/scopt_*.jar $SPARK_HOME/examples/jars/spark-examples*.jar  --rank $4 --numIterations 10 --lambda $3 --kryo /u/weitan/gpfs/spark/data/netflix.data


