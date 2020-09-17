#ifndef NDP_SLS_H
#define NDP_SLS_H

#include <stdint.h>
#include <unordered_map>

void embedding_forward_simd(float* weights, long* indices, long* offsets, bool scale_grad_by_freq,
                        int mode, bool sparse, unsigned int embedding_dim, unsigned int offsets_size,
                        int weight_stride0, int weight_stride1, unsigned int indices_size, float* out);


// void embedding_backward_naive(float* grads, unsigned int* indices, unsigned int* offsets, unsigned int indices_size, unsigned int offsets_size, float* d_weights, unsigned int embedding_dim);
void embedding_backward_simd(float* grads, unsigned int* indices, unsigned int* offsets, unsigned int indices_size, unsigned int offsets_size, float* d_weights, unsigned int embedding_dim);

void grad_coalesce_sorting(float* grads, long* indices, long* offsets, 
                            unsigned int indices_size, unsigned int offsets_size, 
                            float* weights, unsigned int embedding_dim);

void grad_coalesce_hash(float* grads, long* indices, long* offsets, 
                        unsigned int indices_size, unsigned int offsets_size, 
                        float* weights, unsigned int embedding_dim);
#endif
