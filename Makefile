#CODE VARIABLES
DEBUG  =$(if $(debug),-D__DEBUG,)
MSTIME  =$(if $(mstime),-D__MSTIME,)
NDEBUG  =$(if $(ndebug),-DNDEBUG,)
CPU_ONLY =$(if $(cpuonly),-DCPU_ONLY,)
CPU_ONLY_FLAGS = $(if $(cpuonly),,-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand)

CXX=g++
CFLAGS=-c -Wall -std=c++11 -g3 -Ofast -msse2  -I. -I/home/blcv/CODE/Face_LIB/dlib-18.12/ -I /home/blcv/LIB/yaml-cpp-0.5.1/include/ \
      -I/home/blcv/LIB/caffe_master/include/ -I/home/blcv/LIB/caffe_master/src/ -I/usr/local/cuda/include  \
      -I/home/blcv/LIB/ $(DEBUG) $(MSTIME) $(NDEBUG) $(CPU_ONLY)

LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_calib3d -lopencv_ml -ltbb \
	-lboost_filesystem -lboost_system -lboost_program_options -lboost_serialization -lboost_iostreams \
	-L/home/blcv/CODE/Face_LIB/dlib-18.12/examples/build/dlib_build/ -ldlib -lpthread  -L/home/blcv/LIB/yaml-cpp-0.5.1/ -lyaml-cpp \
	-L/home/blcv/LIB/caffe_master/build/lib/ -lcaffe \
	-L/home/blcv/LIB/liblinear/blas/ -lblas -lglog -lcurl -lcurlcpp $(CPU_ONLY_FLAGS)

SOURCES=${wildcard *.cpp Frontalization/*.cpp Utils/*.cpp Net/*.cpp Verification/*.cpp inotify-cxx/*.cpp}
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=deepface

all:  $(SOURCES) $(EXECUTABLE)


clean:
	rm -f *.o
	rm -f $(EXECUTABLE)
	find ./ -name \*.o  -delete

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CXX) $(CFLAGS) $< -o $@