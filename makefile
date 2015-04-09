#CODE VARIABLES
DEBUG  =$(if $(debug),-D__DEBUG,)

CXX=g++
CFLAGS=-c -Wall -std=c++0x -g3 -Ofast -msse2  -I. -I/home/melgor/CODE/Face_Detection/dlib/ \
      -I/home/melgor/CODE/caffe_bn/include/ -I/home/melgor/CODE/caffe_bn/src/ -I/usr/local/cuda/include -I/home/melgor/LIB/ $(DEBUG) 
LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_nonfree -lopencv_flann -lopencv_objdetect \
    -lopencv_video -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -lcudart -lcublas -lcurand -ltbb -lboost_filesystem -lboost_system -lboost_program_options -lopencv_ml \
		-lboost_serialization -lboost_iostreams -L/home/melgor/CODE/Face_Detection/dlib/examples/build/dlib_build/ -ldlib -lpthread   -lyaml-cpp \
		 -L/home/melgor/CODE/caffe_bn/build/lib/ -lcaffe  -L/home/melgor/LIB/liblinear-1.96/ -llinear \
		 -L/home/melgor/LIB/liblinear-1.96/blas/ -lblas
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