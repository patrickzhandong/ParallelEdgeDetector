EXECUTABLE := license

CU_FILES   := cudaprocess.cu

CU_DEPS    :=

CC_FILES   := preprocess.cpp sobel.cpp segmentation.cpp binarize.cpp connected_component.cpp location.cpp findRegion.cpp main.cpp

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -std=c++11
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib/ -lcudart -lpng
else
# Building on Linux
LDFLAGS=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart -lpng
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_35


OBJS= $(OBJDIR)/cudaprocess.o $(OBJDIR)/sobel.o $(OBJDIR)/findRegion.o $(OBJDIR)/preprocess.o $(OBJDIR)/segmentation.o $(OBJDIR)/binarize.o $(OBJDIR)/connected_component.o $(OBJDIR)/location.o $(OBJDIR)/main.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.pgm *.png *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
