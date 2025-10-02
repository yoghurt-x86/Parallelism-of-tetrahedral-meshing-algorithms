CXX        = hipcc -O3 
#-arch=sm_60

SRC = wa1-task3-hipified.cu
EXECUTABLE = wa1-task3

default: compile run

.cu.o: $(SRC)
	$(CXX) -c $@ $<


compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 32

clean:
	rm -f $(EXECUTABLE)

