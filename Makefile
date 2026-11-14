#
# @fileoverview Copyright (c) 2019-2021, by Stefano Gualandi, UniPv,
#               via Ferrata, 1, Pavia, Italy, 27100
#
# @author stefano.gualandi@gmail.com (Stefano Gualandi)
#
#

# Directory for my files
MYHOME          = ${PWD}
BIN             = ${MYHOME}/bin
INCLUDE         = ${MYHOME}/include
LIB             = ${MYHOME}/lib
SRC             = ${MYHOME}/src

OPTFLAG = -O2 -ffast-math -march=native -DNDEBUG -Wall -std=c++11 -Wall
LDFLAGS = -O2 -DNDEBUG -lm -pthread -std=c++11

COMPILER    = g++ ${OPTFLAG}
LINKER      = g++ ${LDFLAGS}

# Directory for output files
OUT_DIR=bin lib

# Build and Test C++ code
laptop: ${OUT_DIR} ${SRC}/TestCLI.cpp
	${COMPILER} -c -g -pg ${SRC}/TestCLI.cpp -o ${LIB}/TestCLI.o -I${INCLUDE}
	${LINKER} -o ${BIN}/test ${LIB}/TestCLI.o
	./bin/test
	
# Correctness tests (compare against a reference solver, check plans)
test: ${OUT_DIR} tests/test_ot1d.cpp ${INCLUDE}/OT1D.hpp
	g++ -O2 -ffast-math -march=native -DNDEBUG -Wall -std=c++11 -pthread tests/test_ot1d.cpp -o ${BIN}/test_ot1d -I${INCLUDE}
	./bin/test_ot1d

# Same tests with AddressSanitizer and UndefinedBehaviorSanitizer
sanitize: ${OUT_DIR} tests/test_ot1d.cpp ${INCLUDE}/OT1D.hpp
	g++ -O1 -g -Wall -Wextra -std=c++11 -pthread -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_ot1d.cpp -o ${BIN}/test_ot1d_asan -I${INCLUDE}
	./bin/test_ot1d_asan

# Build and Test Python wrapper
buildpython:
	cp ${INCLUDE}/OT1D.hpp ${INCLUDE}/parasort.h ${INCLUDE}/pdqsort.h ./python/
	cp README.md ./python/
	cd python && python3 setup.py build_ext --inplace && python3 basic_test.py && python3 test_wrapper.py

# Run longer test:
longtest:
	cd python && python3 OT1D_test.py
	
# Create subdirectory for output files (bin,lib)
MKDIR_P = mkdir -p

lib:
	${MKDIR_P} lib

bin:
	${MKDIR_P} bin

# Be careful!
clean::
	rm -f *.o
	rm -f ${LIB}/*.o
	rm -f *~
	rm -f ${SRC}/*~ ${INCLUDE}/*~
	rm -rf build/*
	rm python/OT1D.cpp python/OT1D.cp*.so python/OT1D.cp*.pyd
