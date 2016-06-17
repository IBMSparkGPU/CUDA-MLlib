cd ../../../../../

$SCALA_HOME/bin/scalac org/apache/spark/ml/optim/NativeCuAcc.scala


javah -o org/apache/spark/ml/optim/NativeCuAcc.h -cp $SCALA_CP:. org.apache.spark.ml.optim.NativeCuAcc

cd -

rm -f *.so

 
echo "**** nvcc compile"
nvcc -shared -I/usr/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -I/opt/share/OpenBLAS-0.2.14/include *.cpp *.cu -o libCuAcc_nvcc.so -Xcompiler "-fPIC -g" -g -lcublas -lcusparse -m64 --use_fast_math --ptxas-options=-v -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35  /opt/share/OpenBLAS-0.2.14/lib/libopenblas.a -lpthread -lrt

ln -sf libCuAcc_nvcc.so libCuAcc.so

nvcc  -I/usr/include -I$JAVA_HOME/include -I$JAVA_HOME/include/linux -I/opt/share/OpenBLAS-0.2.14/include *.cpp *.cu -o cuAcc -Xcompiler "-fPIC" -lcublas -lcusparse -m64  --use_fast_math --ptxas-options=-v -rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35  /opt/share/OpenBLAS-0.2.14/lib/libopenblas.a -lpthread -lrt

