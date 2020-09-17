#include "../include/forward_simd.h"
#include "../include/stridy.h"

void embeddingbag_forward_blk16(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension,
                                float* output, float* embed, long* indices, long* offsets) {
/*
 * input : weights => embed, indices, offsets, mode
 * output: (output, offset2bag, bag_size, max_indices)
 */
    int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        long start_offset = offsets[offset_idx];
        long end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];
        
        __m256 vop0 = _mm256_setzero_ps();
        __m256 vop8 = _mm256_setzero_ps();
        
        for (long j = start_offset ; j < end_offset ; ++j) {
            int index_ij = (int) indices[j];

            __m256 vop0_lo = _mm256_loadu_ps(embed + index_ij * dimension + 0);
            __m256 vop8_lo = _mm256_loadu_ps(embed + index_ij * dimension + 8);

            // #pragma omp atomic update
            {
            vop0 = _mm256_add_ps(vop0, vop0_lo);
            vop8 = _mm256_add_ps(vop8, vop8_lo);
            }
            // printf("i = %d, j= %d, *indices = %d, threadId = %d \n", i, j, index_ij, omp_get_thread_num());
        }
        _mm256_storeu_ps(output + offset_idx * dimension + 0, vop0);
        _mm256_storeu_ps(output + offset_idx * dimension + 8, vop8);
    }
    return;
}


void embeddingbag_forward_blk32(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension,
                                float* output, float* embed, long* indices, long* offsets) {
/*
 * input : weights => embed, indices, offsets, mode
 * output: (output, offset2bag, bag_size, max_indices)
 */
    int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        long start_offset = offsets[offset_idx];
        long end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];


        __m256 vop0 = _mm256_setzero_ps();
        __m256 vop8 = _mm256_setzero_ps();
        __m256 vop16 = _mm256_setzero_ps();
        __m256 vop24 = _mm256_setzero_ps();

        for (long j = start_offset ; j < end_offset ; ++j) {
            int index_ij = (int) indices[j];

            __m256 vop0_lo = _mm256_loadu_ps(embed + index_ij * dimension + 0);
            __m256 vop8_lo = _mm256_loadu_ps(embed + index_ij * dimension + 8);
            __m256 vop16_lo = _mm256_loadu_ps(embed + index_ij * dimension + 16);
            __m256 vop24_lo = _mm256_loadu_ps(embed + index_ij * dimension + 24);

            // #pragma omp atomic update
            {
            vop0 = _mm256_add_ps(vop0, vop0_lo);
            vop8 = _mm256_add_ps(vop8, vop8_lo);
            vop16 = _mm256_add_ps(vop16, vop16_lo);
            vop24 = _mm256_add_ps(vop24, vop24_lo);
            }
            // printf("i = %d, j= %d, *indices = %d, threadId = %d \n", i, j, index_ij, omp_get_thread_num());
        
        }
        _mm256_storeu_ps(output + offset_idx * dimension + 0, vop0);
        _mm256_storeu_ps(output + offset_idx * dimension + 8, vop8);
        _mm256_storeu_ps(output + offset_idx * dimension + 16, vop16);
        _mm256_storeu_ps(output + offset_idx * dimension + 24, vop24);
    }
    return;
}


void embeddingbag_forward_blk64(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension,
                                float* output, float* embed, long* indices, long* offsets) {
/*
 * input : weights => embed, indices, offsets, mode
 * output: (output, offset2bag, bag_size, max_indices)
 */
    int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        long start_offset = offsets[offset_idx];
        long end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];

        __m256 vop0 = _mm256_setzero_ps();
        __m256 vop8 = _mm256_setzero_ps();
        __m256 vop16 = _mm256_setzero_ps();
        __m256 vop24 = _mm256_setzero_ps();
        __m256 vop32 = _mm256_setzero_ps();
        __m256 vop40 = _mm256_setzero_ps();
        __m256 vop48 = _mm256_setzero_ps();
        __m256 vop56 = _mm256_setzero_ps();

        // #pragma unroll
        for (long j = start_offset ; j < end_offset ; ++j) {
            int index_ij = (int) indices[j];

            __m256 vop0_lo = _mm256_loadu_ps(embed + index_ij * dimension + 0);
            __m256 vop8_lo = _mm256_loadu_ps(embed + index_ij * dimension + 8);
            __m256 vop16_lo = _mm256_loadu_ps(embed + index_ij * dimension + 16);
            __m256 vop24_lo = _mm256_loadu_ps(embed + index_ij * dimension + 24);
            __m256 vop32_lo = _mm256_loadu_ps(embed + index_ij * dimension + 32);
            __m256 vop40_lo = _mm256_loadu_ps(embed + index_ij * dimension + 40);
            __m256 vop48_lo = _mm256_loadu_ps(embed + index_ij * dimension + 48);
            __m256 vop56_lo = _mm256_loadu_ps(embed + index_ij * dimension + 56);

            // Addition
            // #pragma omp atomic update
            {
            vop0 = _mm256_add_ps(vop0, vop0_lo);
            vop8 = _mm256_add_ps(vop8, vop8_lo);
            vop16 = _mm256_add_ps(vop16, vop16_lo);
            vop24 = _mm256_add_ps(vop24, vop24_lo);
            vop32 = _mm256_add_ps(vop32, vop32_lo);
            vop40 = _mm256_add_ps(vop40, vop40_lo);
            vop48 = _mm256_add_ps(vop48, vop48_lo);
            vop56 = _mm256_add_ps(vop56, vop56_lo);
            }
            // printf("i = %d, j= %d, *indices = %d, threadId = %d \n", i, j, index_ij, omp_get_thread_num());
        }
        _mm256_storeu_ps(output + offset_idx * dimension + 0, vop0);
        _mm256_storeu_ps(output + offset_idx * dimension + 8, vop8);
        _mm256_storeu_ps(output + offset_idx * dimension + 16, vop16);
        _mm256_storeu_ps(output + offset_idx * dimension + 24, vop24);
        _mm256_storeu_ps(output + offset_idx * dimension + 32, vop32);
        _mm256_storeu_ps(output + offset_idx * dimension + 40, vop40);
        _mm256_storeu_ps(output + offset_idx * dimension + 48, vop48);
        _mm256_storeu_ps(output + offset_idx * dimension + 56, vop56);
    }
    return;
}



void embeddingbag_forward_blk128(unsigned int offsets_size, unsigned int indices_size, unsigned int dimension,
                                float* output, float* embed, long* indices, long* offsets) {
/*
 * input : weights => embed, indices, offsets, mode
 * output: (output, offset2bag, bag_size, max_indices)
 */
    
    __m256 vop0 = _mm256_setzero_ps();
    __m256 vop8 = _mm256_setzero_ps();
    __m256 vop16 = _mm256_setzero_ps();
    __m256 vop24 = _mm256_setzero_ps();
    __m256 vop32 = _mm256_setzero_ps();
    __m256 vop40 = _mm256_setzero_ps();
    __m256 vop48 = _mm256_setzero_ps();
    __m256 vop56 = _mm256_setzero_ps();
    __m256 vop64 = _mm256_setzero_ps();
    __m256 vop72 = _mm256_setzero_ps();
    __m256 vop80 = _mm256_setzero_ps();
    __m256 vop88 = _mm256_setzero_ps();
    __m256 vop96 = _mm256_setzero_ps();
    __m256 vop104 = _mm256_setzero_ps();
    __m256 vop112 = _mm256_setzero_ps();
    __m256 vop120 = _mm256_setzero_ps();
    int offset_idx;
    #pragma omp parallel for schedule(dynamic)
    for (offset_idx = 0; offset_idx < offsets_size ; ++offset_idx) {
        long start_offset = offsets[offset_idx];
        long end_offset = (offset_idx == offsets_size-1) ? indices_size : offsets[offset_idx+1];

        for (long j = start_offset ; j < end_offset ; ++j) {
            int index_ij = (int) indices[j];

            __m256 vop0_lo = _mm256_loadu_ps(embed + index_ij * dimension + 0);
            __m256 vop8_lo = _mm256_loadu_ps(embed + index_ij * dimension + 8);
            __m256 vop16_lo = _mm256_loadu_ps(embed + index_ij * dimension + 16);
            __m256 vop24_lo = _mm256_loadu_ps(embed + index_ij * dimension + 24);
            __m256 vop32_lo = _mm256_loadu_ps(embed + index_ij * dimension + 32);
            __m256 vop40_lo = _mm256_loadu_ps(embed + index_ij * dimension + 40);
            __m256 vop48_lo = _mm256_loadu_ps(embed + index_ij * dimension + 48);
            __m256 vop56_lo = _mm256_loadu_ps(embed + index_ij * dimension + 56);
            __m256 vop64_lo = _mm256_loadu_ps(embed + index_ij * dimension + 64);
            __m256 vop72_lo = _mm256_loadu_ps(embed + index_ij * dimension + 72);
            __m256 vop80_lo = _mm256_loadu_ps(embed + index_ij * dimension + 80);
            __m256 vop88_lo = _mm256_loadu_ps(embed + index_ij * dimension + 88);
            __m256 vop96_lo = _mm256_loadu_ps(embed + index_ij * dimension + 96);
            __m256 vop104_lo = _mm256_loadu_ps(embed + index_ij * dimension + 104);
            __m256 vop112_lo = _mm256_loadu_ps(embed + index_ij * dimension + 112);
            __m256 vop120_lo = _mm256_loadu_ps(embed + index_ij * dimension + 120);

            // #pragma omp atomic update
            {
            vop0 = _mm256_add_ps(vop0, vop0_lo);
            vop8 = _mm256_add_ps(vop8, vop8_lo);
            vop16 = _mm256_add_ps(vop16, vop16_lo);
            vop24 = _mm256_add_ps(vop24, vop24_lo);
            vop32 = _mm256_add_ps(vop32, vop32_lo);
            vop40 = _mm256_add_ps(vop40, vop40_lo);
            vop48 = _mm256_add_ps(vop48, vop48_lo);
            vop56 = _mm256_add_ps(vop56, vop56_lo);
            vop64 = _mm256_add_ps(vop64, vop64_lo);
            vop72 = _mm256_add_ps(vop72, vop72_lo);
            vop80 = _mm256_add_ps(vop80, vop80_lo);
            vop88 = _mm256_add_ps(vop88, vop88_lo);
            vop96 = _mm256_add_ps(vop96, vop96_lo);
            vop104 = _mm256_add_ps(vop104, vop104_lo);
            vop112 = _mm256_add_ps(vop112, vop112_lo);
            vop120 = _mm256_add_ps(vop120, vop120_lo);
            }
            // printf("i = %d, j= %d, *indices = %d, threadId = %d \n", i, j, index_ij, omp_get_thread_num());
        }
        _mm256_storeu_ps(output + offset_idx * dimension + 0, vop0);
        _mm256_storeu_ps(output + offset_idx * dimension + 8, vop8);
        _mm256_storeu_ps(output + offset_idx * dimension + 16, vop16);
        _mm256_storeu_ps(output + offset_idx * dimension + 24, vop24);
        _mm256_storeu_ps(output + offset_idx * dimension + 32, vop32);
        _mm256_storeu_ps(output + offset_idx * dimension + 40, vop40);
        _mm256_storeu_ps(output + offset_idx * dimension + 48, vop48);
        _mm256_storeu_ps(output + offset_idx * dimension + 56, vop56);
        _mm256_storeu_ps(output + offset_idx * dimension + 64, vop64);
        _mm256_storeu_ps(output + offset_idx * dimension + 72, vop72);
        _mm256_storeu_ps(output + offset_idx * dimension + 80, vop80);
        _mm256_storeu_ps(output + offset_idx * dimension + 88, vop88);
        _mm256_storeu_ps(output + offset_idx * dimension + 96, vop96);
        _mm256_storeu_ps(output + offset_idx * dimension + 104, vop104);
        _mm256_storeu_ps(output + offset_idx * dimension + 112, vop112);
        _mm256_storeu_ps(output + offset_idx * dimension + 120, vop120);
    }
    return;
}


// inline void embeddingbag_forward_generic(int offset_idx, int start_offset, int end_offset, int dimension,
//                                 float* output, float* embed, int* indices) {
//     int i, j;
//     #pragma omp parallel for private(i) schedule(dynamic)
//     for(i = 0; i < batch_size ; ++i) {
//         __m256 vop0 = _mm256_setzero_ps();
//         __m256 vop8 = _mm256_setzero_ps();

//         // #pragma omp parallel for private(j)
//         for (int j = 0 ; j < pooling_size ; ++j) {
//             __m256 vop0_lo = _mm256_setzero_ps();
//             __m256 vop8_lo = _mm256_setzero_ps();

//             int index_ij = *get2df<int>(indices, i, j, pooling_size);

//             vop0_lo = _mm256_loadu_ps(embed + index_ij * dimension + 0);
//             _mm_prefetch(reinterpret_cast<const char*>(embed + index_ij * dimension + 8), _MM_HINT_T0);
//             vop8_lo = _mm256_loadu_ps(embed + index_ij * dimension + 8);

//             // Addition
//             // #pragma omp atomic update
//             {
//             vop0 = _mm256_add_ps(vop0, vop0_lo);
//             vop8 = _mm256_add_ps(vop8, vop8_lo);
//             }
//             // printf("i = %d, j= %d, *indices = %d, threadId = %d \n", i, j, index_ij, omp_get_thread_num());
//         }
        
//         _mm256_storeu_ps(get2df<float>(output, offset_idx, 0, dimension), vop0);
//         _mm256_storeu_ps(get2df<float>(output, offset_idx, 8, dimension), vop8);
//     }
// }
