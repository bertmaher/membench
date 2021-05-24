#include <benchmark/benchmark.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <random>
#include <sys/mman.h>

volatile float dummy;

static void sequential(benchmark::State& state) {
  size_t size = state.range(0);
  std::vector<float> v(size, 1.0f);
  float sum = 0.0f;
  for (auto _ : state) {
    for (size_t i = 0; i < size; i++) {
      sum += v[i];
    }
  }
  dummy = sum;
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(sequential)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void sequential_vector(benchmark::State& state) {
  //std::vector<float> v(state.range(0), 1.0f);
  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum = {0.0f};
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; i += 8) {
      sum = _mm256_add_ps(sum, _mm256_load_ps(&v[i]));
    }
  }
  dummy = sum[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(sequential_vector)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void sequential_vector_unrolled(benchmark::State& state) {
  //std::vector<float> v(state.range(0), 1.0f);
  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum0 = {0.0f};
  __m256 sum1 = {0.0f};
  __m256 sum2 = {0.0f};
  __m256 sum3 = {0.0f};
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; i += 32) {
      sum0 = _mm256_add_ps(sum0, _mm256_load_ps(&v[i]));
      sum1 = _mm256_add_ps(sum1, _mm256_load_ps(&v[i + 8]));
      sum2 = _mm256_add_ps(sum2, _mm256_load_ps(&v[i + 16]));
      sum3 = _mm256_add_ps(sum3, _mm256_load_ps(&v[i + 24]));
    }
  }
  dummy = sum0[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(sequential_vector_unrolled)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void sequential_load(benchmark::State& state) {
  size_t size = state.range(0);
  std::vector<float> idx(size);
  for (size_t i = 0; i < size; i++) {
    idx[i] = i;
  }
  
  std::vector<float> v(size, 1.0f);
  float sum = 0.0f;
  for (auto _ : state) {
    for (size_t i = 0; i < v.size(); i++) {
      sum += v[idx[i]];
    }
  }
  dummy = sum;
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(sequential_load)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void sequential_load_vector(benchmark::State& state) {
  size_t size = state.range(0);
  std::vector<size_t> idx(size / 8);
  for (size_t i = 0; i < idx.size(); i++) {
    idx[i] = i * 8;
  }
  
  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum = {0.0f};

  for (auto _ : state) {
    for (size_t i = 0; i < idx.size(); i++) {
      sum = _mm256_add_ps(sum, _mm256_load_ps(&v[idx[i]]));
    }
  }
  dummy = sum[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(sequential_load_vector)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void random_load(benchmark::State& state) {
  size_t size = state.range(0);
  std::vector<float> idx(size);
  for (size_t i = 0; i < size; i++) {
    idx[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(idx.begin(), idx.end(), g);
  
  std::vector<float> v(size, 1.0f);
  float sum = 0.0f;
  for (auto _ : state) {
    for (size_t i = 0; i < v.size(); i++) {
      sum += v[idx[i]];
    }
  }
  dummy = sum;
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(random_load)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void random_load_vector(benchmark::State& state) {
  size_t size = state.range(0);
  std::vector<size_t> idx(size / 8);
  for (size_t i = 0; i < idx.size(); i++) {
    idx[i] = i * 8;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(idx.begin(), idx.end(), g);

  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum = {0.0f};

  for (auto _ : state) {
    for (size_t i = 0; i < idx.size(); i++) {
      sum = _mm256_add_ps(sum, _mm256_load_ps(&v[idx[i]]));
    }
  }
  dummy = sum[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(random_load_vector)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void random_compute_vector(benchmark::State& state) {
  size_t size = state.range(0);
  std::random_device rd;
  std::minstd_rand g(rd());

  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum = {0.0f};

  size_t mask = (size - 1) - 7;
  for (auto _ : state) {
    for (size_t i = 0; i < size / 8; i++) {
      sum = _mm256_add_ps(sum, _mm256_load_ps(&v[g()  & mask]));
    }
  }
  dummy = sum[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(random_compute_vector)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

static void random_lcg(benchmark::State& state) {
  std::random_device rd;
  std::minstd_rand g(rd());
  auto x = g();
  for (auto _ : state) {
     x = g();
  }
  dummy = x;
  state.counters["rand/s"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(random_lcg);

static void random_lcg_nr(benchmark::State& state) {
  uint32_t rstate = 0xdeafbeef;
  for (auto _ : state) {
    rstate = 1664525 * rstate + 1013904223;
  }
  dummy = rstate;
  state.counters["rand/s"] = benchmark::Counter(state.iterations(), benchmark::Counter::kIsRate);
}
BENCHMARK(random_lcg_nr);

static void random_compute_vector_lcg(benchmark::State& state) {
  size_t size = state.range(0);
  uint32_t rstate = 0xdeafbeef;

  void* aligned;
  posix_memalign(&aligned, sizeof(__m256), state.range(0) * sizeof(float));
  float* v = (float*)aligned;
  __m256 sum = {0.0f};

  size_t mask = (size - 1) - 7;
  for (auto _ : state) {
    for (size_t i = 0; i < size / 8; i++) {
      rstate = 1664525 * rstate + 1013904223;
      sum = _mm256_add_ps(sum, _mm256_load_ps(&v[rstate & mask]));
    }
  }
  dummy = sum[0];
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * state.range(0) * sizeof(float),
				   benchmark::Counter::kIsRate);
}
BENCHMARK(random_compute_vector_lcg)->RangeMultiplier(8)->Range(32 * 8, 1<<26);

#define STREAM_ARRAY_SIZE (16 << 20)
#define OFFSET 0
#define STREAM_TYPE double

static STREAM_TYPE a[STREAM_ARRAY_SIZE+OFFSET],
  b[STREAM_ARRAY_SIZE+OFFSET],
  c[STREAM_ARRAY_SIZE+OFFSET];

static void stream_copy(benchmark::State& state) {
  for (int j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
  
  for (auto _ : state) {
    for (int j = 0; j < STREAM_ARRAY_SIZE; j++) {
      c[j] = a[j];
    }
    benchmark::DoNotOptimize(c[0]);
  }
  
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * 2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
					      benchmark::Counter::kIsRate);
  
}
BENCHMARK(stream_copy);

// "Quick and dirty" random number generator from "Numerical Recipes":
// https://www.unf.edu/~cwinton/html/cop4300/s09/class.notes/LCGinfo.pdf
struct randqd1 {
  uint32_t operator()() {
    return state = 1664525 * state + 1013904223;
  }
  uint32_t state{0xdeadbeef};
};

template<int CHUNK_SIZE>
static void stream_copy_rand_chunks(benchmark::State& state) {
  randqd1 gen;

  constexpr auto NCHUNKS = STREAM_ARRAY_SIZE / CHUNK_SIZE;

  for (int j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
  
  for (auto _ : state) {
    for (int j = 0; j < NCHUNKS; j++) {
      auto rstate = gen();
      double* cchunk = &c[(rstate % NCHUNKS) * CHUNK_SIZE];
      double* achunk = &a[(rstate % NCHUNKS) * CHUNK_SIZE];
      for (int i = 0; i < CHUNK_SIZE; i++) {
	cchunk[i] = achunk[i];
      }
    }
  }
  
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * 2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
					      benchmark::Counter::kIsRate);
  
}

#define BENCH_RAND_CHUNKS(N) \
static void stream_copy_rand_chunks_##N(benchmark::State& state) { stream_copy_rand_chunks<N>(state); } BENCHMARK(stream_copy_rand_chunks_##N);

BENCH_RAND_CHUNKS(1);
BENCH_RAND_CHUNKS(8);
BENCH_RAND_CHUNKS(64);
BENCH_RAND_CHUNKS(512);
BENCH_RAND_CHUNKS(1024);
BENCH_RAND_CHUNKS(2048);
BENCH_RAND_CHUNKS(4096);
BENCH_RAND_CHUNKS(8192);
#undef BENCH_RAND_CHUNKS

template<int CHUNK_SIZE>
static void stream_huge_copy_rand_chunks(benchmark::State& state) {
  randqd1 gen;

  constexpr auto NCHUNKS = STREAM_ARRAY_SIZE / CHUNK_SIZE;

  STREAM_TYPE* a = (STREAM_TYPE*)mmap(NULL, sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
		  PROT_READ | PROT_WRITE,
		  MAP_ANON | MAP_PRIVATE,
		  -1, 0);
  STREAM_TYPE* c = (STREAM_TYPE*)mmap(NULL, sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
		  PROT_READ | PROT_WRITE,
		  MAP_ANON | MAP_PRIVATE,
		  -1, 0);
  for (int j = 0; j < STREAM_ARRAY_SIZE; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
  
  for (auto _ : state) {
    for (int j = 0; j < NCHUNKS; j++) {
      auto rstate = gen();
      double* cchunk = &c[(rstate % NCHUNKS) * CHUNK_SIZE];
      double* achunk = &a[(rstate % NCHUNKS) * CHUNK_SIZE];
      for (int i = 0; i < CHUNK_SIZE; i++) {
	cchunk[i] = achunk[i];
      }
    }
  }
  
  state.counters["GB/s"] = benchmark::Counter(state.iterations() * 2 * sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
					      benchmark::Counter::kIsRate);

  munmap(a, sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE);
  munmap(c, sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE);
}

#define BENCH_RAND_CHUNKS(N) \
static void stream_huge_copy_rand_chunks_##N(benchmark::State& state) { stream_huge_copy_rand_chunks<N>(state); } BENCHMARK(stream_huge_copy_rand_chunks_##N);

BENCH_RAND_CHUNKS(1);
BENCH_RAND_CHUNKS(8);
BENCH_RAND_CHUNKS(64);
BENCH_RAND_CHUNKS(512);
BENCH_RAND_CHUNKS(1024);
BENCH_RAND_CHUNKS(2048);
BENCH_RAND_CHUNKS(4096);
BENCH_RAND_CHUNKS(8192);

BENCHMARK_MAIN();
