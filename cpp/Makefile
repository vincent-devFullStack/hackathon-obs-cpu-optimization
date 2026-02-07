CXX ?= g++
PROFILE ?= portable
CXXFLAGS_BASE := -std=c++17 -Wall -Wextra -pedantic
ifeq ($(PROFILE),native)
OPT_FLAGS := -O3 -march=native -mtune=native
else ifeq ($(PROFILE),portable)
OPT_FLAGS := -O3
else
$(error unsupported PROFILE='$(PROFILE)' (expected portable or native))
endif
CXXFLAGS ?= $(OPT_FLAGS) $(CXXFLAGS_BASE)
LDFLAGS ?=
empty :=
space := $(empty) $(empty)
comma := ,
BUILD_FLAGS_STR := $(subst $(space),$(comma),$(CXXFLAGS))
DEFINES_COMMON = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CXX)\" -DBUILD_PROFILE=\"$(PROFILE)\"
DEFINES_NAIVE = $(DEFINES_COMMON) -DIMPL_ID=\"cpp-naive\"
DEFINES_SIMD = $(DEFINES_COMMON) -DIMPL_ID=\"cpp-simd-$(PROFILE)\"

TARGET_NAIVE = benchmark_cpp
TARGET_SIMD = benchmark_cpp_simd

all: $(TARGET_NAIVE) $(TARGET_SIMD)

$(TARGET_NAIVE): benchmark_cpp.cpp cosine.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_NAIVE) benchmark_cpp.cpp cosine.cpp -lm $(LDFLAGS) -o $(TARGET_NAIVE)

$(TARGET_SIMD): benchmark_cpp_simd.cpp cosine.cpp cosine_simd.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES_SIMD) benchmark_cpp_simd.cpp cosine.cpp cosine_simd.cpp -lm $(LDFLAGS) -o $(TARGET_SIMD)

clean:
	rm -f $(TARGET_NAIVE) $(TARGET_SIMD)

print-flags:
	@echo PROFILE=$(PROFILE)
	@echo CXXFLAGS=$(CXXFLAGS)

.PHONY: all clean print-flags
