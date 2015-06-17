.PHONY: all
all: failingCU

.PHONY: clean
clean:
	rm *.o failingCU failingCL

failingCU: failing_main.cu ../test_utils.o
	nvcc failing_main.cu ../test_utils.o -o failingCU
