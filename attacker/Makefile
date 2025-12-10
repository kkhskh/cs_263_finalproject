# FLUSH+RELOAD Attacker Makefile

CC = gcc
CFLAGS_COMMON = -Wall -Wextra -march=native

# Release build (optimized for timing accuracy)
CFLAGS_RELEASE = $(CFLAGS_COMMON) -O2 -DNDEBUG

# Debug build (symbols, no optimization)
CFLAGS_DEBUG = $(CFLAGS_COMMON) -O0 -g -DDEBUG

# Profile build (optimized with debug symbols)
CFLAGS_PROFILE = $(CFLAGS_COMMON) -O2 -g -pg

LDFLAGS = -ldl

TARGET = flush_reload
SRC = flush_reload.c

.PHONY: all clean debug release profile help test

# Default target
all: release

release: CFLAGS = $(CFLAGS_RELEASE)
release: $(TARGET)

debug: CFLAGS = $(CFLAGS_DEBUG)
debug: $(TARGET)_debug

profile: CFLAGS = $(CFLAGS_PROFILE)
profile: $(TARGET)_profile

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@ (release)"

$(TARGET)_debug: $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@ (debug)"

$(TARGET)_profile: $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built $@ (profile)"

# Quick test
test: release
	@echo "=== Testing calibration mode ==="
	./$(TARGET) 1000 0 2
	@echo ""
	@echo "=== Testing CSV mode (10 samples) ==="
	./$(TARGET) 10 150 0
	@echo ""
	@echo "=== Testing stats mode ==="
	./$(TARGET) 1000 150 1

clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_profile *.o gmon.out

help:
	@echo "FLUSH+RELOAD Attacker Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build release version (default)"
	@echo "  release  - Build optimized release version"
	@echo "  debug    - Build debug version with symbols"
	@echo "  profile  - Build profiling version"
	@echo "  test     - Build and run quick tests"
	@echo "  clean    - Remove build artifacts"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Usage after building:"
	@echo "  ./flush_reload [iterations] [threshold] [mode] [target_lib] [offset]"
	@echo ""
	@echo "Modes:"
	@echo "  0 = CSV output (iter,cycles,hit)"
	@echo "  1 = Statistics only"
	@echo "  2 = Calibration (find threshold)"
	@echo "  3 = Realtime monitoring"
