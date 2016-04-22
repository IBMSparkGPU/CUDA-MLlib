CUDA		:= /usr/local/cuda

CUDASRC := $(wildcard src/*.cu)
CUDAOBJ := $(CUDASRC:%.cu=%.o)

CPPSRC := $(wildcard src/*.C)
CPPOBJ := $(CPPSRC:%.C=%.o)

JNISRC := $(wildcard spark/*.c)
JNIOBJ := $(JNISRC:%.c=%.o)

UTILSRC := ../utilities.cu
UTILOBJ := utilities.o


# Allow various cuda versions
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM21    := -gencode arch=compute_20,code=sm_21
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
CUDAFLAGS       := $(GENCODE_SM21) $(GENCODE_SM35)


INCLUDES        := -I.. -I./. -I$(CUDA)/include -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux 

# It is important that CUDAFLAGS end with -Xcompiler and CXXFLAGS start with -fPIC
# because nvcc wants the pair -Xcompiler -fPIC, while gcc wants only -fPIC
CUDAFLAGS +=   -Xcompiler
CXXFLAGS  :=   -fPIC -g -shared -m64 -O3 $(INCLUDES)


#extend gcc loader flags to load cuda libraries
LDFLAGS  += -L$(CUDA)/lib64 -lcudart -lcublas -lpthread

SHARED_LIBRARY  := libGPULBFGS.so


default: lbfgs

spark/%.o: spark/%.c
	gcc $(CXXFLAGS) -c $< -o $@

src/%.o: src/%.C
	g++ $(CXXFLAGS) -c $< -o $@

src/%.o: src/%.cu
	nvcc  $(CUDAFLAGS) $(CXXFLAGS) -c $< -o $@

$(UTILOBJ): $(UTILSRC)
	nvcc  $(CUDAFLAGS) $(CXXFLAGS) -c $< -o $@

$(SHARED_LIBRARY): $(COBJ) $(CUDAOBJ) $(CPPOBJ) $(JNIOBJ) $(UTILOBJ)
	g++ -shared -o $(SHARED_LIBRARY) $(COBJ) $(CUDAOBJ) $(CPPOBJ) $(JNIOBJ) $(UTILOBJ) $(LDFLAGS)

lbfgs: $(SHARED_LIBRARY)
	g++  -g $(SHARED_LIBRARY) -o src/lbfgs -lm

clean:
	rm -f src/lbfgs src/*.o $(SHARED_LIBRARY) $(COBJ) $(CPPOBJ) $(CUDAOBJ) $(LVOVOBJ) $(JNIOBJ) $(UTILOBJ)

all:
	make clean
	make 
	./run.sh
