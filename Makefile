# Compilers and commands
CC=		g++
NVCC=		nvcc -Xptxas -v
LINK=		nvcc
DEL_FILE= 	rm -f
MPI_INCLUDES=	/usr/include/openmpi
MPI_LIBS=	/usr/lib/openmpi

#Flags
CFLAGS		= -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lm -lefence
NVCCFLAGS	= -arch=sm_35 --use_fast_math -Wno-deprecated-gpu-targets --relocatable-device-code true #-g -G

INCPATH		= /usr/include/

####### Files
SOURCES		= exp_gpu.cu exp_gpu_double.cu
OBJECTS		= main.o

TARGET 		= run

	
run: main.o exp_gpu.o exp_gpu_double.o
	$(NVCC) -L$(MPI_LIBS) -lmpi main.o exp_gpu.o exp_gpu_double.o $(NVCCFLAGS) -o  $(TARGET) -I$(INCPATH)

exp_gpu.o: exp_gpu.cu exp_gpu.h
	$(NVCC)  -I$(MPI_INCLUDES) exp_gpu.cu -c $(NVCCFLAGS) -I$(INCPATH)

exp_gpu_double.o: exp_gpu_double.cu exp_gpu_double.h
	$(NVCC)  -I$(MPI_INCLUDES) exp_gpu_double.cu -c $(NVCCFLAGS) -I$(INCPATH)

main.o: main.cpp exp_gpu.h exp_gpu_double.h
	$(CC) -I$(MPI_INCLUDES) -c main.cpp -lm -o main.o $(CFLAGS)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET) *.o

test: run
	./run -n 1000 -m 1000 -tr 3

mpitest: run
	mpirun -n 3 ./run -n 1000 -m 1000 -pt
