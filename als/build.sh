#build the jni cpp code and cuda code for ALS
#!/bin/bash
#unamestr=`uname -m`
if [ -z ${JAVA_HOME} ]; then
	echo "Please set JAVA_HOME!"
	exit 1
else
	echo "use existing JAVA_HOME " $JAVA_HOME
fi

if [ -z ${CUDA_ROOT} ]; then
	echo "Please set CUDA_ROOT to the cuda installation, say /usr/local/cuda !"
	exit 1
else
	echo "use existing CUDA_ROOT " $CUDA_ROOT
fi

echo "compile the cuda & native code"
$CUDA_ROOT/bin/nvcc -shared  -D_USE_GPU_ -I/usr/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux ../utilities.cu src/cuda/als.cu src/CuMFJNIInterface.cpp -o libGPUALS.so -Xcompiler "-fPIC" -m64  -use_fast_math -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35 -O3 -Xptxas -dlcm=ca -L{$CUDA_ROOT}/lib64 -lcublas -lcusparse    

#echo "build spark"
#SPARK_HOME=../../Spark-MLlib/
#cd $SPARK_HOME
#build/mvn -Pyarn -Phadoop-2.4 -Dhadoop.version=2.4.0 -DskipTests clean package

#echo "build spark distribution"
#cd $SPARK_HOME
#./dev/make-distribution.sh -Pnetlib-lgpl -Pyarn -Phadoop-2.7 -Dhadoop.version=2.7.2 