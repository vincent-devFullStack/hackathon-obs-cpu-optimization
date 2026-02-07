CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
LDFLAGS ?=
empty :=
space := $(empty) $(empty)
comma := ,
BUILD_FLAGS_STR := $(subst $(space),$(comma),$(CXXFLAGS))
DEFINES = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CXX)\"

TARGET = benchmark_cpp
SOURCES = benchmark_cpp.cpp cosine.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(DEFINES) $(SOURCES) -lm $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
