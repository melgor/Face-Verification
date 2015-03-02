#CODE VARIABLES
DEBUG  =$(if $(debug),-D__DEBUG,)

CXX=g++
CFLAGS=-c -Wall -std=c++0x -g3 -Ofast -msse2  -I. -I/home/blcv/CODE/Face_LIB/dlib-18.12/ -I /home/blcv/LIB/yaml-cpp-0.5.1/include/ $(DEBUG) 
LDFLAGS=-L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_nonfree \
	 -lopencv_flann -lopencv_objdetect -ltbb -lboost_filesystem -lboost_system -lboost_program_options -lopencv_ml -lboost_serialization \
	 -lboost_iostreams -L/home/blcv/CODE/Face_LIB/dlib-18.12/examples/build/dlib_build/ -ldlib -lpthread  -L /home/blcv/LIB/yaml-cpp-0.5.1/ -lyaml-cpp
SOURCES=${wildcard *.cpp frontalization/*.cpp utils/*.cpp}
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