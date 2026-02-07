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
DEFINES_COMMON = -DBUILD_FLAGS=\"$(BUILD_FLAGS_STR)\" -DBUILD_COMPILER=\"$(CC)\" -DBUILD_PROFILE=\"$(PROFILE)\"
DEFINES_SIMD = $(DEFINES_COMMON) -DIMPL_ID=\"c-simd-$(PROFILE)\"
DEFINES_NAIVE = $(DEFINES_COMMON) -DIMPL_ID=\"c-naive\"

TARGET_NAIVE = benchmark_c
TARGET_SIMD = benchmark_c_simd

all: $(TARGET_NAIVE) $(TARGET_SIMD)

$(TARGET_NAIVE): benchmark_c.c cosine.c
	$(CC) $(CFLAGS) $(DEFINES_NAIVE) benchmark_c.c cosine.c -lm $(LDFLAGS) -o $(TARGET_NAIVE)

$(TARGET_SIMD): benchmark_c_simd.c cosine.c cosine_simd.c
	$(CC) $(CFLAGS) $(DEFINES_SIMD) benchmark_c_simd.c cosine.c cosine_simd.c -lm $(LDFLAGS) -o $(TARGET_SIMD)

clean:
	rm -f $(TARGET_NAIVE) $(TARGET_SIMD)

print-flags:
	@echo PROFILE=$(PROFILE)
	@echo CFLAGS=$(CFLAGS)

.PHONY: all clean print-flags
