#ifndef FORWARD_SIMD_H
#define FORWARD_SIMD_H

#include <iostream>
#include <random>
#include <chrono>
#include <unistd.h>

#include "avx2neon.h"
// #include <x86intrin.h>
#include <omp.h>
#include <ctime>
#include <cassert>
#include <algorithm>


void embeddingbag_forward_blk16(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* output, float* embed, long* index_iter, long* offsets);
void embeddingbag_forward_blk32(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* output, float* embed, long* index_iter, long* offsets);
void embeddingbag_forward_blk64(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* output, float* embed, long* index_iter, long* offsets);
void embeddingbag_forward_blk128(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* output, float* embed, long* index_iter, long* offsets);
// void embeddingbag_forward_generic(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* output, float* embed, long* index_iter, long* offsets);

#endif
