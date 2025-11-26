// attacker/flush_reload.c
// Skeleton of a FLUSH+RELOAD attacker that probes a single address
// and prints CSV lines: iter,cycles.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>  // rdtscp, clflush
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline uint64_t rdtsc(void) {
    unsigned int aux;
    return __rdtscp(&aux);
}

static inline void flush(void *p) {
    _mm_clflush(p);
}

int main(int argc, char **argv) {
    size_t iterations = 500000;
    if (argc >= 2) {
        iterations = strtoull(argv[1], NULL, 10);
    }

    // For now we just point at a local probe buffer.
    // Later: map a shared library page and set target to a specific offset.
    static unsigned char probe[4096];
    memset((void *)probe, 1, sizeof(probe));
    volatile unsigned char *target = &probe[0];

    fprintf(stderr, "# FLUSH+RELOAD skeleton, iterations=%zu\n", iterations);
    fprintf(stderr, "# CSV: iter,cycles\n");
    printf("iter,cycles\n");

    for (size_t i = 0; i < iterations; i++) {
        flush((void *)target);

        // small barrier so flush completes
        asm volatile("mfence" ::: "memory");

        uint64_t t0 = rdtsc();
        (void)*target;
        uint64_t t1 = rdtsc();
        uint64_t delta = t1 - t0;

        printf("%zu,%lu\n", i, (unsigned long)delta);
    }

    return 0;
}
