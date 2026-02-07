CC ?= cc
PROFILE ?= portable
CFLAGS_BASE := -std=c11 -Wall -Wextra -pedantic
ifeq ($(PROFILE),native)
OPT_FLAGS := -O3 -march=native -mtune=native
else ifeq ($(PROFILE),portable)
OPT_FLAGS := -O3
else
$(error unsupported PROFILE='$(PROFILE)' (expected portable or native))
endif
CFLAGS ?= $(OPT_FLAGS) $(CFLAGS_BASE)
LDFLAGS ?=
empty :=
space := $(empty) $(empty)
comma := ,
BUILD_FLAGS_STR := $(subst $(space),$(comma),$(CFLAGS))
DEFINES = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CC)\" -DBUILD_PROFILE=\"$(PROFILE)\"

TARGET = benchmark_c
SOURCES = benchmark_c.c cosine.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(DEFINES) $(SOURCES) -lm $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

print-flags:
	@echo PROFILE=$(PROFILE)
	@echo CFLAGS=$(CFLAGS)

.PHONY: all clean print-flags
