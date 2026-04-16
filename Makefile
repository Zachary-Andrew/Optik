# Makefile for TP+OptOA
#
# Targets:
#   all       — build all four executables (default)
#   test      — build and run the unit tests
#   bench     — run the full benchmark (30 trials, default sizes)
#   quick     — run a quick benchmark (5 trials, small sizes, for CI)
#   clean     — remove build artefacts
#
# Variables you can override on the command line:
#   CXX       — compiler (default: g++)
#   BUILD     — output directory (default: build)
#   TRIALS    — number of benchmark trials (default: 30)
#   SMALL     — small dataset k-mer count (default: 500000)
#   LARGE     — large dataset k-mer count (default: 5000000)

CXX    ?= g++
BUILD  ?= build
TRIALS ?= 30
K      ?= 31
SMALL  ?= 500000
LARGE  ?= 5000000

CXXFLAGS := -std=c++17 -O3 -march=native -funroll-loops -ffast-math \
            -Wall -Wextra -Wpedantic -Wshadow -Iinclude

SRCS_BUILD  := src/build.cpp
SRCS_QUERY  := src/query.cpp
SRCS_TEST   := src/test.cpp
SRCS_BENCH  := src/benchmark.cpp

BINS := $(BUILD)/tpoptoa_build \
        $(BUILD)/tpoptoa_query \
        $(BUILD)/tpoptoa_test  \
        $(BUILD)/tpoptoa_benchmark

.PHONY: all test bench quick clean

all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/tpoptoa_build: $(SRCS_BUILD) $(wildcard include/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tpoptoa_query: $(SRCS_QUERY) $(wildcard include/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tpoptoa_test: $(SRCS_TEST) $(wildcard include/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tpoptoa_benchmark: $(SRCS_BENCH) $(wildcard include/*.hpp) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

# INPUT is required for test and bench targets.
# Set it on the command line: make test INPUT=genome.fa
#                             make bench INPUT=genome.fa
INPUT ?=

test: $(BUILD)/tpoptoa_test
	@if [ -z "$(INPUT)" ]; then 	    echo "Usage: make test INPUT=genome.fa"; exit 1; fi
	$(BUILD)/tpoptoa_test -i $(INPUT) -k $(K)

bench: $(BUILD)/tpoptoa_benchmark
	@if [ -z "$(INPUT)" ]; then 	    echo "Usage: make bench INPUT=genome.fa"; exit 1; fi
	$(BUILD)/tpoptoa_benchmark -i $(INPUT) -n $(TRIALS) -k $(K) \
	    -o benchmark.tsv 2>&1 | tee benchmark.log

quick: $(BUILD)/tpoptoa_benchmark
	@if [ -z "$(INPUT)" ]; then 	    echo "Usage: make quick INPUT=genome.fa"; exit 1; fi
	$(BUILD)/tpoptoa_benchmark -i $(INPUT) -n 5 -k $(K) \
	    -o benchmark_quick.tsv

clean:
	rm -rf $(BUILD) benchmark.tsv benchmark_quick.tsv benchmark.log
