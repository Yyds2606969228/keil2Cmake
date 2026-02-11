#ifndef K2C_GROUP_H
#define K2C_GROUP_H

#include <stddef.h>

#define K2C_INPUT_SIZE 64
#define K2C_OUTPUT_SIZE 24
#define K2C_NUM_BUFFERS 0
#define K2C_NUM_WEIGHTS 0
#define K2C_ARENA_BYTES 0
#define K2C_ARENA_WORDS 1

#if K2C_NUM_BUFFERS > 0
#define K2C_BUFFER_SLOTS K2C_NUM_BUFFERS
#else
#define K2C_BUFFER_SLOTS 1
#endif
#if K2C_NUM_WEIGHTS > 0
#define K2C_WEIGHT_SLOTS K2C_NUM_WEIGHTS
#else
#define K2C_WEIGHT_SLOTS 1
#endif

typedef struct {
  const char* name;
  const int* shape;
  size_t rank;
  size_t elem_size;
} k2c_io_desc_t;

typedef struct {
  void* arena;
  size_t arena_bytes;
  void* buffers[K2C_BUFFER_SLOTS];
  void* weights[K2C_WEIGHT_SLOTS];
} k2c_ctx_t;

typedef struct {
  void (*forward)(const void* input, void* output);
  int (*prepare)(k2c_ctx_t* ctx, void* arena, size_t arena_bytes);
  int (*invoke)(k2c_ctx_t* ctx, const void* input, void* output);
  size_t input_size;
  size_t output_size;
  size_t arena_bytes;
} k2c_model_t;

const k2c_model_t* k2c_get_model(void);
int k2c_prepare(k2c_ctx_t* ctx, void* arena, size_t arena_bytes);
int k2c_invoke(k2c_ctx_t* ctx, const void* input, void* output);
const k2c_io_desc_t* k2c_get_input_desc(size_t* n);
const k2c_io_desc_t* k2c_get_output_desc(size_t* n);
const k2c_model_t* getModel(void);
void k2c_forward(const void* input, void* output);

#endif /* K2C_GROUP_H */