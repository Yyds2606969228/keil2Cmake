#include "group.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

static const float const_W[108] = { 0.52601814f, 0.35633081f, 0.58690387f, 0.35205299f, 0.68945092f, 0.47760758f, 0.28162712f, 0.31716579f, 0.13355698f, 0.46581393f, 0.67171150f, 0.03761704f, 0.92601556f, 0.64193690f, 0.86871094f, 0.26669309f, 0.33507261f, 0.02321799f, 0.60151345f, 0.84093183f, 0.65989023f, 0.96975642f, 0.48992568f, 0.35287994f, 0.43082410f, 0.35524324f, 0.19765112f, 0.72899425f, 0.00365644f, 0.91373307f, 0.11893459f, 0.57686424f, 0.62851179f, 0.69528127f, 0.58252269f, 0.52684724f, 0.29628140f, 0.82690322f, 0.06408279f, 0.21719585f, 0.95368040f, 0.84347636f, 0.49582416f, 0.27314112f, 0.73202389f, 0.40381336f, 0.55049884f, 0.09814419f, 0.33978987f, 0.40380967f, 0.91064000f, 0.03769377f, 0.79414243f, 0.06822248f, 0.51380628f, 0.55574328f, 0.82547575f, 0.35378328f, 0.16526800f, 0.15387118f, 0.45976937f, 0.83529347f, 0.03651457f, 0.60015357f, 0.91219944f, 0.14179505f, 0.78909701f, 0.12171857f, 0.81221896f, 0.32569602f, 0.72318953f, 0.98953032f, 0.54327744f, 0.02268874f, 0.95524812f, 0.67619520f, 0.25948393f, 0.77404881f, 0.41385457f, 0.16287450f, 0.33645475f, 0.25234559f, 0.08638212f, 0.08585867f, 0.23600589f, 0.59464222f, 0.33195972f, 0.70577443f, 0.16889769f, 0.40958500f, 0.12007143f, 0.69018900f, 0.71038628f, 0.39230779f, 0.23048761f, 0.58516550f, 0.76528251f, 0.41842791f, 0.96283358f, 0.23384131f, 0.21693087f, 0.58504325f, 0.46641695f, 0.83266658f, 0.83157313f, 0.40710899f, 0.07381804f, 0.64665931f };
static const float const_B[6] = { 0.16002914f, 0.66982055f, 0.43569899f, 0.21596844f, 0.12262303f, 0.30569479f };

static const int k2c_input_shape[4] = { 1, 4, 4, 4 };
static const int k2c_output_shape[4] = { 1, 6, 2, 2 };
static const k2c_io_desc_t k2c_inputs[] = {
  { "input", k2c_input_shape, 4, sizeof(float) },
};
static const k2c_io_desc_t k2c_outputs[] = {
  { "output", k2c_output_shape, 4, sizeof(float) },
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
  for (size_t ni = 0; ni < 1; ++ni) {
    for (size_t oc = 0; oc < 6; ++oc) {
      for (size_t oh = 0; oh < 2; ++oh) {
        for (size_t ow = 0; ow < 2; ++ow) {
          float sum = (float*)const_B[oc];
          size_t g = oc / 3;
          size_t ic_begin = g * 2;
          for (size_t ic_local = 0; ic_local < 2; ++ic_local) {
            size_t ic = ic_begin + ic_local;
            for (size_t kh = 0; kh < 3; ++kh) {
              for (size_t kw = 0; kw < 3; ++kw) {
                int in_h = (int)(oh * 1 + kh * 1) - 0;
                int in_w = (int)(ow * 1 + kw * 1) - 0;
                if (in_h >= 0 && in_h < (int)4 && in_w >= 0 && in_w < (int)4) {
                  size_t in_idx = ((ni * 4 + ic) * 4 + (size_t)in_h) * 4 + (size_t)in_w;
                  size_t w_idx = ((oc * 2 + ic_local) * 3 + kh) * 3 + kw;
                  sum += (float*)input[in_idx] * (float*)const_W[w_idx];
                }
              }
            }
          }
          (float*)output[((ni * 6 + oc) * 2 + oh) * 2 + ow] = sum;
        }
      }
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
  .input_size = 64,
  .output_size = 24,
  .arena_bytes = 0,
};

const k2c_model_t* k2c_get_model(void) {
  return &k2c_model;
}

const k2c_model_t* getModel(void) {
  return &k2c_model;
}