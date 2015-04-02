#CODE VARIABLES
DEBUG  =$(if $(debug),-D__DEBUG,)

CXX=g++
CFLAGS=-c -Wall -std=c++0x -g3 -Ofast -msse2  -I. -I/home/blcv/CODE/Face_LIB/dlib-18.12/ -I /home/blcv/LIB/yaml-cpp-0.5.1/include/ \
      -I/home/blcv/LIB/caffe_bn/include/ -I/home/blcv/LIB/caffe_bn/src/ -I/usr/local/cuda/include -I/media/blcv/44488cdd-c584-4aab-9706-6929f09b9871/LIB/ $(DEBUG) 
LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_nonfree -lopencv_flann -lopencv_objdetect \
    -lopencv_video -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand -ltbb -lboost_filesystem -lboost_system -lboost_program_options -lopencv_ml \
		-lboost_serialization -lboost_iostreams -L/home/blcv/CODE/Face_LIB/dlib-18.12/examples/build/dlib_build/ -ldlib -lpthread  -L /home/blcv/LIB/yaml-cpp-0.5.1/ -lyaml-cpp \
		 -L/home/blcv/LIB/caffe_bn/build/lib/ -lcaffe -lproto -L/media/blcv/44488cdd-c584-4aab-9706-6929f09b9871/LIB/liblinear-1.96/ -llinear \
		 -L/media/blcv/44488cdd-c584-4aab-9706-6929f09b9871/LIB/liblinear-1.96/blas/ -lblas
SOURCES=${wildcard *.cpp Frontalization/*.cpp Utils/*.cpp Net/*.cpp Verification/*.cpp}
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