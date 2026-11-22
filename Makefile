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
	${MAKE} test-simd

# Same tests with the SIMD sorting enabled (x86-simd-sort, AVX2 and AVX-512).
# Only ot1d_avx2.cpp and ot1d_avx512.cpp are compiled with SIMD flags, and they
# are linked last, AVX2 before AVX-512; the tests of a level are skipped at
# runtime if the CPU does not support it.
AVX512_FLAGS = -mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd
AVX2_FLAGS   = -mavx2
SIMD_DEFS    = -DOT1D_USE_AVX2 -DOT1D_USE_AVX512

test-simd: ${OUT_DIR} tests/test_ot1d.cpp ${INCLUDE}/OT1D.hpp ${INCLUDE}/ot1d_simdsort.inc ${INCLUDE}/ot1d_avx2.cpp ${INCLUDE}/ot1d_avx512.cpp
	g++ -O2 -DNDEBUG -Wall -std=c++17 ${AVX2_FLAGS} -c ${INCLUDE}/ot1d_avx2.cpp -o ${LIB}/ot1d_avx2.o -I${INCLUDE}
	g++ -O2 -DNDEBUG -Wall -std=c++17 ${AVX512_FLAGS} -c ${INCLUDE}/ot1d_avx512.cpp -o ${LIB}/ot1d_avx512.o -I${INCLUDE}
	g++ -O2 -ffast-math -DNDEBUG -Wall -std=c++11 -pthread ${SIMD_DEFS} -c tests/test_ot1d.cpp -o ${LIB}/test_ot1d.o -I${INCLUDE}
	g++ -pthread ${LIB}/test_ot1d.o ${LIB}/ot1d_avx2.o ${LIB}/ot1d_avx512.o -o ${BIN}/test_ot1d_simd
	./bin/test_ot1d_simd

# Same tests with AddressSanitizer and UndefinedBehaviorSanitizer
sanitize: ${OUT_DIR} tests/test_ot1d.cpp ${INCLUDE}/OT1D.hpp
	g++ -O1 -g -Wall -Wextra -std=c++11 -pthread -fsanitize=address,undefined -fno-omit-frame-pointer tests/test_ot1d.cpp -o ${BIN}/test_ot1d_asan -I${INCLUDE}
	./bin/test_ot1d_asan
	g++ -O1 -g -Wall -std=c++17 ${AVX2_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer -c ${INCLUDE}/ot1d_avx2.cpp -o ${LIB}/ot1d_avx2_asan.o -I${INCLUDE}
	g++ -O1 -g -Wall -std=c++17 ${AVX512_FLAGS} -fsanitize=address,undefined -fno-omit-frame-pointer -c ${INCLUDE}/ot1d_avx512.cpp -o ${LIB}/ot1d_avx512_asan.o -I${INCLUDE}
	g++ -O1 -g -Wall -Wextra -std=c++11 -pthread ${SIMD_DEFS} -fsanitize=address,undefined -fno-omit-frame-pointer -c tests/test_ot1d.cpp -o ${LIB}/test_ot1d_asan.o -I${INCLUDE}
	g++ -pthread -fsanitize=address,undefined ${LIB}/test_ot1d_asan.o ${LIB}/ot1d_avx2_asan.o ${LIB}/ot1d_avx512_asan.o -o ${BIN}/test_ot1d_simd_asan
	./bin/test_ot1d_simd_asan
	g++ -O1 -g -Wall -std=c++11 -pthread -fsanitize=thread tests/test_ot1d.cpp -o ${BIN}/test_ot1d_tsan -I${INCLUDE}
	./bin/test_ot1d_tsan

# Build and Test Python wrapper
buildpython:
	cp ${INCLUDE}/OT1D.hpp ${INCLUDE}/pdqsort.h ${INCLUDE}/ot1d_simdsort.inc ${INCLUDE}/ot1d_avx2.cpp ${INCLUDE}/ot1d_avx512.cpp ./python/
	rm -rf ./python/x86-simd-sort && cp -r ${INCLUDE}/x86-simd-sort ./python/
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
	rm -f python/ot1d.cpp python/ot1d.cp*.so python/ot1d.cp*.pyd
