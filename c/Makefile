CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic
LDFLAGS ?=
empty :=
space := $(empty) $(empty)
comma := ,
BUILD_FLAGS_STR := $(subst $(space),$(comma),$(CFLAGS))
DEFINES = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CC)\"

TARGET = benchmark_c
SOURCES = benchmark_c.c cosine.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(DEFINES) $(SOURCES) -lm $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean
