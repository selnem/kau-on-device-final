CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra
LIBS = -ldxrt -lfftw3 -lm

recognize: main.cpp features.cpp features.hpp
	$(CXX) $(CXXFLAGS) -o recognize main.cpp features.cpp $(LIBS)

clean:
	rm -f recognize
