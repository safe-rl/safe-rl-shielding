ifndef CUDD_PATH
CUDD_PATH = /usr/local
endif

EXECUTABLE = shield_synthesizer
SOURCES    = $(wildcard *.cpp)
OBJECTS    = $(patsubst %,%,${SOURCES:.cpp=.o})
CXX        = g++
FLAGS      = -Wall -c -std=c++11 -O3 -march=native
LIBS       = -L$(CUDD_PATH)/lib -l cudd -l obj
INCLUDE    = -I$(CUDD_PATH)/include

all: $(EXECUTABLE) 

%.o: %.cpp
	$(CXX) $(FLAGS) -o $@ $< $(INCLUDE)

#link Objects
$(EXECUTABLE) : $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS)
	chmod u+x $(EXECUTABLE)

clean:
	rm -f *.o
	rm $(EXECUTABLE)
