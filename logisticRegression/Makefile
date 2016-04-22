# Source
CUFILES		:= predict.cu ../utilities.cu
JNIFILES	:= spark/LogisticRegressionNative.cpp
MFILES		:= predict_main.c

CUDA		:= /usr/local/cuda

NVCC			:= nvcc
CC			:= gcc
CXX			:= g++

CUOBJS      := predict.o utilities.o
JNIOBJS		:= LogisticRegressionNative.o

CFLAGS		:= -O3 -g 

EXECUTABLE		:= gpu_predict

SHARED_LIBRARY  := libLogisticRegressionNative.so

INCLUDES        := -I.. -I./. -I$(CUDA)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux
#Libs
LIB     	:= -L $(CUDA)/lib64 -L /usr/lib64
LINK    	:= -lcublas -lcusparse -lcudart -lm -lcuda

OPT             := -g --ptxas-options=-v -Xptxas -dlcm=cg -use_fast_math -O4 -m64 -lineinfo -maxrregcount=32

# CUDA code generation flags
GENCODE_SM21    := -gencode arch=compute_20,code=sm_21
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35

GENCODE_FLAGS   := $(GENCODE_SM21) $(GENCODE_SM35)

#jni:
#    javah -cp spark-mllib_2.10-2.0.0-SNAPSHOT.jar:scala-library.jar org.apache.spark.mllib.classification.LogisticRegressionNative

all: $(EXECUTABLE)

lib: $(JNIFILES) $(CUFILES)
	$(NVCC) -c $(CUFILES) $(INCLUDES) $(GENCODE_FLAGS) $(OPT) -Xcompiler "-fPIC -c -O3"
	$(CXX) -c $(JNIFILES) $(INCLUDES) $(CFLAGS) -fPIC
	$(CXX) -shared -o $(SHARED_LIBRARY) $(JNIOBJS) $(CUOBJS) $(LIB) $(LINK)

$(EXECUTABLE): lib
	$(CXX) -o $(EXECUTABLE) $(MFILES) $(SHARED_LIBRARY) $(INCLUDES) $(CFLAGS) $(LIB) $(LINK)

clean: 
	rm -f $(EXECUTABLE) *.o *.so
	
