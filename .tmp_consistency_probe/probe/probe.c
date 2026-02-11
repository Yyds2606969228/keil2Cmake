#include "probe.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

static const float const_W[12] = { 0.81447929f, 0.90600044f, 0.93032634f, 0.05676498f, 0.88803339f, 0.22401592f, 0.12948546f, 0.97931123f, 0.70134342f, 0.99576390f, 0.87575012f, 0.59275734f };
static const float const_B[3] = { 0.63732642f, 0.89626187f, 0.69001114f };

static const int k2c_input_shape[2] = { 1, 4 };
static const int k2c_output_shape[2] = { 1, 3 };
static const k2c_io_desc_t k2c_inputs[] = {
  { "input", k2c_input_shape, 2, sizeof(float) },
};
static const k2c_io_desc_t k2c_outputs[] = {
  { "output", k2c_output_shape, 2, sizeof(float) },
};

const k2c_io_desc_t* k2c_get_input_desc(size_t* n) {
  if (n) *n = 1;
  return k2c_inputs;
}

const k2c_io_desc_t* k2c_get_output_desc(size_t* n) {
  if (n) *n = 1;
  return k2c_outputs;
}

int k2c_prepare(k2c_ctx_t* ctx, void* arena, size_t arena_bytes) {
  if (!ctx) return -1;
  ctx->arena = arena;
  ctx->arena_bytes = arena_bytes;
  if (K2C_ARENA_BYTES > 0) {
    if (!arena || arena_bytes < K2C_ARENA_BYTES) return -2;
    unsigned char* base = (unsigned char*)arena;
  }
  return 0;
}

int k2c_invoke(k2c_ctx_t* ctx, const void* input_ptr, void* output_ptr) {
  if (!ctx || !input_ptr || !output_ptr) return -1;
  const float* input = (const float*)input_ptr;
  float* output = (float*)output_ptr;
  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      float sum = 0.0f;
      for (size_t t = 0; t < 4; ++t) {
        sum += (float*)input[i * 4 + t] * (float*)const_W[t * 3 + j];
      }
      (float*)output[i * 3 + j] = sum;
    }
  }
  for (size_t i = 0; i < 1; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      (float*)output[i * 3 + j] += (float*)const_B[j];
    }
  }

  return 0;
}

void k2c_forward(const void* input, void* output) {
  static int k2c_prepared = 0;
  static k2c_ctx_t k2c_ctx;
  static float k2c_default_arena[K2C_ARENA_WORDS];
  if (!k2c_prepared) {
    (void)k2c_prepare(&k2c_ctx, k2c_default_arena, sizeof(k2c_default_arena));
    k2c_prepared = 1;
  }
  (void)k2c_invoke(&k2c_ctx, input, output);
}

static const k2c_model_t k2c_model = {
  .forward = k2c_forward,
  .prepare = k2c_prepare,
  .invoke = k2c_invoke,
  .input_size = 4,
  .output_size = 3,
  .arena_bytes = 0,
};

const k2c_model_t* k2c_get_model(void) {
  return &k2c_model;
}

const k2c_model_t* getModel(void) {
  return &k2c_model;
}