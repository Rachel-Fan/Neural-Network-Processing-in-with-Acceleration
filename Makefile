CC=g++
NVCC=nvcc
CFLAGS=-I./include

# Link with CUDA libraries
LDFLAGS=-L/usr/local/cuda/lib64 -lcudart

# Define objects
OBJS=src/main.o src/helpers.o src/matrixVectorMul.o

# Define binary
BIN=build/app

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

src/matrixVectorMul.o: src/matrixVectorMul.cu
	$(NVCC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(BIN)

.PHONY: all clean
