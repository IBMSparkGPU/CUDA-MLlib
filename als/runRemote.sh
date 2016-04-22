#submit to a remote cluster
#!/bin/bash
#check input arguments
if [[ ( "$1" != "cpu" && "$1" != "gpu" ) || "$#" -ne 4 ]]; then
  echo "Usage: cpu|gpu #cores lambda rank" >&2
  echo "e.g., ./runRemote.sh gpu 12 0.058 100"
  exit 1
fi

#modify spark.ALS.useGPU= to point to your libJNIInterface.so
gpu=""
if [ "$1" = "gpu" ]; then
	gpu="--conf spark.mllib.ALS.useGPU=/u/weitan/gpfs/spark/git/CUDA-MLlib/als/libGPUALS.so"
	echo "using gpu libaray: "
	echo $gpu
fi

#modify to point to your java folders
JAVA_HOME=/u/weitan/gpfs/spark/jdk1.8.0_72
#SPARK_HOME=/u/weitan/gpfs/spark/git/Spark-MLlib
SPARK_HOME=/u/bherta/Spark-MLlib/dist
$SPARK_HOME/bin/spark-submit --driver-memory 128g --executor-memory 128G --class org.apache.spark.examples.mllib.MovieLensALS $gpu --master spark://dccxc009:7077 --executor-cores $2 $SPARK_HOME/lib/spark-examples-*.jar --rank $4 --numIterations 10 --lambda $3 --kryo /u/weitan/gpfs/spark/data/netflix.data

#example: ./runRemote.sh gpu 12 0.058 100
#$SPARK_HOME/bin/spark-submit --driver-memory 128g --executor-memory 128G --class org.apache.spark.examples.mllib.MovieLensALS $gpu --master spark://dccxc009:7077 --executor-cores $2 $SPARK_HOME/examples/target/scala-*/spark-examples-*.jar --rank $4 --numIterations 10 --lambda $3 --kryo /u/weitan/gpfs/spark/data/netflix.data
