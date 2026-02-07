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
DEFINES = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CXX)\" -DBUILD_PROFILE=\"$(PROFILE)\"

TARGET = benchmark_cpp
SOURCES = benchmark_cpp.cpp cosine.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(SOURCES) -lm $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

print-flags:
	@echo PROFILE=$(PROFILE)
	@echo CXXFLAGS=$(CXXFLAGS)

.PHONY: all clean print-flags
