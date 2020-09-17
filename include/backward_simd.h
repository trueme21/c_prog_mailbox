#ifndef BACKWARD_SIMD_H
#define BACKWARD_SIMD_H


#include <iostream>
#include <random>
#include <chrono>
#include <unistd.h>

#include "avx2neon.h"

#include <omp.h>
#include <ctime>
#include <cassert>
#include <algorithm>
#include "stridy.h"

void embeddingbag_backward_blk16(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices);
void embeddingbag_backward_blk32(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices);
void embeddingbag_backward_blk64(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices);
void embeddingbag_backward_blk128(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices);


#endif
