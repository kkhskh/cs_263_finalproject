# FLUSH+RELOAD Attacker Dockerfile
#
# Build: docker build -t llm-attacker-flushreload .
# Run:   docker run --rm --privileged llm-attacker-flushreload
#
# Environment variables:
#   ITERATIONS, THRESHOLD, MODE, TARGET_LIB, TARGET_OFFSET, PIN_CPU, OUTPUT_FILE

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential linux-tools-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY flush_reload.c Makefile ./
RUN make release && make clean && make release

RUN mkdir -p /app/output

ENV ITERATIONS=100000 THRESHOLD=0 MODE=0 TARGET_LIB="" TARGET_OFFSET=0 PIN_CPU="" OUTPUT_FILE=""

COPY <<'EOF' /app/entrypoint.sh
#!/bin/bash
set -e
CMD="./flush_reload $ITERATIONS $THRESHOLD $MODE"
if [ -n "$TARGET_LIB" ]; then
    CMD="$CMD $TARGET_LIB $TARGET_OFFSET"
fi
echo "[entrypoint] Running: $CMD" >&2
if [ -n "$OUTPUT_FILE" ]; then
    echo "[entrypoint] Output to: /app/output/$OUTPUT_FILE" >&2
    $CMD > /app/output/$OUTPUT_FILE
else
    $CMD
fi
EOF

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
