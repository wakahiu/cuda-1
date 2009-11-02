EXE := test
OBJ := test.o
SRC := test.cu

GCCFLAGS := #--compiler-bindir /opt/gcc34/

all: $(EXE)

devemu: $(SRC)
	nvcc -deviceemu $(GCCFLAGS) $(SRC) -o $(EXE) 

$(EXE): $(SRC)
	nvcc $(GCCFLAGS) $(SRC) -o $(EXE) 

clean: 
	rm -rf *.o *.out $(EXE)