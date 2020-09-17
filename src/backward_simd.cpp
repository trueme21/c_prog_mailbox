#include "../include/backward_simd.h"
#include "../include/stridy.h"


void embeddingbag_backward_blk16(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices) {
    unsigned int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        unsigned int start_offset = offsets[offset_idx];
        unsigned int end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];

        __m256 vop0 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 0, dimension));
        __m256 vop8 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 8, dimension));
        
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 0, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 8, dimension)), _MM_HINT_T0);

        for (unsigned int j = 0; j < end_offset - start_offset ; j++) {
            unsigned int index_ij = offsets[offset_idx] + j;            
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 0, dimension), vop0);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 8, dimension), vop8);
        }
    }
}

void embeddingbag_backward_blk32(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices) {
    unsigned int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        unsigned int start_offset = offsets[offset_idx];
        unsigned int end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];

        __m256 vop0 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 0, dimension));
        __m256 vop8 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 8, dimension));
        __m256 vop16 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 16, dimension));
        __m256 vop24 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 24, dimension));

        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 0, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 8, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 16, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 24, dimension)), _MM_HINT_T0);
            
        for (unsigned int j = 0; j < end_offset - start_offset ; j++) {
            unsigned int index_ij = offsets[offset_idx] + j;
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 0, dimension), vop0);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 8, dimension), vop8);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 16, dimension), vop16);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 24, dimension), vop24);
        }
    }
}


void embeddingbag_backward_blk64(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices) {
    unsigned int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        unsigned int start_offset = offsets[offset_idx];
        unsigned int end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];
        
        __m256 vop0 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 0, dimension));
        __m256 vop8 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 8, dimension));
        __m256 vop16 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 16, dimension));
        __m256 vop24 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 24, dimension));
        __m256 vop32 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 32, dimension));
        __m256 vop40 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 40, dimension));
        __m256 vop48 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 48, dimension));
        __m256 vop56 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 56, dimension));

        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 0, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 8, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 16, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 24, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 32, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 40, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 48, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 56, dimension)), _MM_HINT_T0);
            
        for (unsigned int j = 0; j < end_offset - start_offset ; j++) {
            unsigned int index_ij = offsets[offset_idx] + j;
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 0, dimension), vop0);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 8, dimension), vop8);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 16, dimension), vop16);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 24, dimension), vop24);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 32, dimension), vop32);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 40, dimension), vop40);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 48, dimension), vop48);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 56, dimension), vop56);
        }
    }
}


void embeddingbag_backward_blk128(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension, float* grads, float* d_weights, unsigned int* offsets, unsigned int* indices) {
    unsigned int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        unsigned int start_offset = offsets[offset_idx];
        unsigned int end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];
        
        __m256 vop0 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 0, dimension));
        __m256 vop8 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 8, dimension));
        __m256 vop16 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 16, dimension));
        __m256 vop24 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 24, dimension));
        __m256 vop32 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 32, dimension));
        __m256 vop40 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 40, dimension));
        __m256 vop48 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 48, dimension));
        __m256 vop56 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 56, dimension));
        __m256 vop64 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 64, dimension));
        __m256 vop72 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 72, dimension));
        __m256 vop80 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 80, dimension));
        __m256 vop88 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 88, dimension));
        __m256 vop96 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 96, dimension));
        __m256 vop104 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 104, dimension));
        __m256 vop112 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 112, dimension));
        __m256 vop120 = _mm256_loadu_ps(get2df<float>(grads, offset_idx, 120, dimension));

        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 0, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 8, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 16, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 24, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 32, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 40, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 48, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 56, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 64, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 72, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 80, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 88, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 96, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 104, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 112, dimension)), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(get2df<float>(grads, offset_idx, 120, dimension)), _MM_HINT_T0);
        
        for (unsigned int j = 0; j < end_offset - start_offset ; j++) {
            unsigned int index_ij = offsets[offset_idx] + j;
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 0, dimension), vop0);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 8, dimension), vop8);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 16, dimension), vop16);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 24, dimension), vop24);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 32, dimension), vop32);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 40, dimension), vop40);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 48, dimension), vop48);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 56, dimension), vop56);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 64, dimension), vop64);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 72, dimension), vop72);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 80, dimension), vop80);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 88, dimension), vop88);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 96, dimension), vop96);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 104, dimension), vop104);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 112, dimension), vop112);
            _mm256_storeu_ps(get2df<float>(d_weights, index_ij, 120, dimension), vop120);
        }
    }
}

