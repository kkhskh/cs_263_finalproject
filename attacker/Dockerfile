<<<<<<< HEAD
# FLUSH+RELOAD Attacker Dockerfile
#
# Build: docker build -t llm-attacker .
# Run:   docker run --rm --privileged llm-attacker
#
# Environment variables:
#   ITERATIONS   - Number of iterations (default: 100000)
#   THRESHOLD    - Hit/miss threshold, 0=auto (default: 0)
#   MODE         - Output mode 0-3 (default: 0)
#   TARGET_LIB   - Library path to probe (default: "")
#   TARGET_OFFSET - Hex offset in library (default: 0)
#   PIN_CPU      - CPU to pin to (default: "")
#   OUTPUT_FILE  - Save output to file (default: "")

FROM ubuntu:24.04

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    linux-tools-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source and build
COPY flush_reload.c Makefile ./
RUN make release && make clean && make release

# Create output directory
RUN mkdir -p /app/output

# Environment variables with defaults
ENV ITERATIONS=100000
ENV THRESHOLD=0
ENV MODE=0
ENV TARGET_LIB=""
ENV TARGET_OFFSET=0
ENV PIN_CPU=""
ENV OUTPUT_FILE=""

# Entrypoint script that passes env vars to the binary
COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e

# Build command with arguments
CMD="./flush_reload $ITERATIONS $THRESHOLD $MODE"

# Add target lib if specified
if [ -n "$TARGET_LIB" ]; then
    CMD="$CMD $TARGET_LIB $TARGET_OFFSET"
fi

echo "[entrypoint] Running: $CMD" >&2

# Run with or without output file
if [ -n "$OUTPUT_FILE" ]; then
    echo "[entrypoint] Output to: /app/output/$OUTPUT_FILE" >&2
    $CMD > /app/output/$OUTPUT_FILE
else
    $CMD
fi
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
=======
# attacker/Dockerfile
# Build this on the x86 VM later.

FROM gcc:13

WORKDIR /attacker

COPY flush_reload.c Makefile ./

RUN make

CMD ["./flush_reload", "500000"]
>>>>>>> 8213a2da0756912468d3a900e918f4e3f949446b
