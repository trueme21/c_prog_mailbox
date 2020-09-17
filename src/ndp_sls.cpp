#include <iostream>
#include <random>
#include <chrono>
#include <unistd.h>
//#include <x86intrin.h>
#include <omp.h>
#include <ctime>
#include <cassert>
#include <algorithm>
#include <string.h>
#include <stdint.h>
#include "../include/ndp_sls.h"
#include "../include/forward_simd.h"
#include "../include/backward_simd.h"


void embedding_forward_simd(float* weights, long* indices, long* offsets, bool scale_grad_by_freq,
                        int mode, bool sparse, unsigned int embedding_dim, unsigned int offsets_size,
                        int weight_stride0, int weight_stride1, unsigned int indices_size, float* out){//per_sample_weights
  
    assert(embedding_dim >= 0);
    if (embedding_dim <= 16)
        embeddingbag_forward_blk16(offsets_size, indices_size, embedding_dim, out, weights, indices, offsets);
    else if (embedding_dim <= 32)
        embeddingbag_forward_blk32(offsets_size, indices_size, embedding_dim, out, weights, indices, offsets);
    else if (embedding_dim <= 64)
        embeddingbag_forward_blk64(offsets_size, indices_size, embedding_dim, out, weights, indices, offsets);
    else if (embedding_dim <= 128)
        embeddingbag_forward_blk128(offsets_size, indices_size, embedding_dim, out, weights, indices, offsets);
    else {
        printf("Not Implemented yet\n");
    }
        // embeddingbag_forward_generic(start_offset, end_offset, dimension, output, embed, index_iter);
//    std::cout << std::endl;
//    std::cout << "[ndp_sls] out : " << std::endl;
//    for (int i = 0; i < offsets_size*embedding_dim ; ++i) {
//        std::cout << out[i] << " ";
//    }
//    std::cout << std::endl;
}

// // Declare embedding_size * dimension of d_weights on the benchmark script
// void embedding_backward_naive(float* grads, int* indices, int* offsets, int indices_size, int offsets_size, float* d_weights, int embedding_dim) {
//   // num_weights => number of rows in embedding weights
//   // d_weights = num_weights * embedding_dim
//   // grads = num_weights * embedding_dim
//   // size = num_weights

//   // TODO: d_weights -- Initialize to zero here?
//     int m;
//     #pragma omp parallel for private(m) schedule(dynamic)
//     for (m = 0; m < offsets_size; m++) {
//         int start_offset = offsets[m];
//         int end_offset = (m == offsets_size - 1) ? indices_size : offsets[m+1];
//         for (int j = start_offset ; j < end_offset ; ++j) {
//             int idx = indices[j];
//             for (int dim = 0; dim < embedding_dim; dim++) {
//                 #pragma omp atomic
//                 d_weights[idx * embedding_dim + dim] += grads[m * embedding_dim + dim];
//             }
//         }
//     }
// }

// void embedding_backward_naive(float* grads, unsigned int* indices, unsigned int* offsets, unsigned int indices_size, unsigned int offsets_size, float* d_weights, unsigned int embedding_dim) {
//     unsigned int m;
//     #pragma omp parallel for private(m) schedule(dynamic)
//     for (m = 0; m < offsets_size; m++) {
//         unsigned int start_offset = offsets[m];
//         unsigned int end_offset = (m == offsets_size - 1) ? indices_size : offsets[m+1];
//         for (unsigned int j = 0; j < end_offset - start_offset ; j++) {
//             for (unsigned int dim = 0; dim < embedding_dim; dim++) {
//                 *get2df<float>(d_weights, start_offset + j, dim, embedding_dim) = *get2df<float>(grads, m, dim, embedding_dim);
//                 // d_weights[(offsets[m] + j) * embedding_dim + dim] = grads[m * embedding_dim + dim];
//             }
//         }
//     }
// }


void embedding_backward_simd(float* grads, unsigned int* indices, unsigned int* offsets, unsigned int indices_size, unsigned int offsets_size, float* d_weights, unsigned int embedding_dim) {
  // TODO: d_weights -- Initialize to zero here?
    assert(embedding_dim >= 0);
    if (embedding_dim <= 16)
        embeddingbag_backward_blk16(offsets_size, indices_size, embedding_dim, grads, d_weights, offsets, indices);
    else if (embedding_dim <= 32)
        embeddingbag_backward_blk32(offsets_size, indices_size, embedding_dim, grads, d_weights, offsets, indices);
    else if (embedding_dim <= 64)
        embeddingbag_backward_blk64(offsets_size, indices_size, embedding_dim, grads, d_weights, offsets, indices);
    else if (embedding_dim <= 128)
        embeddingbag_backward_blk128(offsets_size, indices_size, embedding_dim, grads, d_weights, offsets, indices);
    else {
        printf("Not Implemented yet\n");
    }
}



// d_weights needs to be coalesced form
void grad_coalesce_sorting(float* grads, long* indices, long* offsets, 
                        unsigned int indices_size, unsigned int offsets_size, 
                        float* weights, unsigned int embedding_dim) {
    // sort and sweep
    long* arg_unique_indices = (long *) malloc(indices_size * sizeof(*indices));   // arg
    long* argsorted_indices = (long *) malloc(indices_size * sizeof(*indices));
    // FIXME: argsort: naive approach...
    std::iota(argsorted_indices, argsorted_indices + indices_size, 0);
    auto comparator = [&indices](long a, long b){ return indices[a] < indices[b]; };
    std::sort(argsorted_indices, argsorted_indices + indices_size, comparator);
    
    // sorted indices
    int xx = 0;
    arg_unique_indices[argsorted_indices[0]] = 0;
    for (int i = 1; i < indices_size; i++) {
        if (indices[argsorted_indices[i]] != indices[argsorted_indices[i-1]]) {
            xx++;
        }
        arg_unique_indices[argsorted_indices[i]] = xx;
    }

    long* unique_indices = (long *) malloc((xx + 1) * sizeof(*indices));
    for (int i = 0 ; i < indices_size; i++) {
        unique_indices[arg_unique_indices[i]] = indices[i];
    }

    // expand offset
    long* expanded_offset = (long *) malloc(indices_size * sizeof(*indices));
    #pragma omp parallel for
    for (int i = 0 ; i < offsets_size; i++) {
        long start_offset = offsets[i];
        long tmp = (i == offsets_size-1) ? indices_size : offsets[i + 1];
        for (int j = start_offset; j < tmp; j++) {
            expanded_offset[j] = i;  // ex) 0 0 1 1 1 1 2 ...
        }
    }
    // xx + 1 = unique_indices_size
    //float* d_coalesced = (float *) malloc ((xx + 1) * embedding_dim * sizeof(float));
    float* d_coalesced = (float *) calloc ((xx + 1) * embedding_dim, sizeof(float));

#ifdef DEBUG
    printf("arg_unique_indices : \n");
    printVector<unsigned int>(arg_unique_indices, indices_size);
    printVector<int>(expanded_offset, indices_size);
#endif

    // (1) Reduction omp, (2) Critical omp, (3) column-major loop performs worse
    for (int j = 0 ; j < indices_size; j++) {
        long offset_idx = expanded_offset[j];
        long coalesced_idx = arg_unique_indices[j];
        for (int dim = 0; dim < embedding_dim; dim++) {
            d_coalesced[coalesced_idx * embedding_dim + dim] += grads[offset_idx * embedding_dim + dim];  // TODO: Change to get2df
        }
    }

#ifdef DEBUG
    printf("d_coalesced\n");
    printVector<float>(d_coalesced, (xx+1) * embedding_dim);
#endif

    // DIRECT UPDATE ON EMBEDDING TABLE
    // Faster than collapse 2
    #pragma omp parallel for  
    for (int j = 0; j < (xx + 1); j++) {  // Unique indices
        for (int dim = 0; dim < embedding_dim; dim++) {
            weights[unique_indices[j] * embedding_dim + dim] -= d_coalesced[j * embedding_dim + dim];
        }
    }
    free(d_coalesced);
    free(expanded_offset);
    free(arg_unique_indices);
    free(unique_indices);
    free(argsorted_indices);
}

void grad_coalesce_hash(float* grads, long* indices, long* offsets, 
                        unsigned int indices_size, unsigned int offsets_size, 
                        float* weights, unsigned int embedding_dim) {
    /*
     * int* unique_indices, // with idxs, same with indices size
     * Example unique_indices
     * if indices = [0, 2, 4, 0, 4, 3],
     * then unique_indices = [0, 1, 2, 0, 2, 3]
     * unique_indices should be cumulative from 0 to n - 1 (n = size of unique_indices array)
     */
    
    // NOTE: unique_indices => args
    // Indices <- Unique_indices
    unsigned int* arg_unique_indices = (unsigned int *) malloc(indices_size * sizeof(unsigned int));  // arg
    std::unordered_map<long, int> s;

    int xx = 0;
    // Traverse the input array 
    for (unsigned int i = 0; i < indices_size; i++) { 
        // If not present, then put it in hashtable and increment result 
        if (s.find(indices[i]) == s.end()) {  // avg find: O(1)
            s.insert(std::make_pair(indices[i], xx));
            xx++;
        }
        arg_unique_indices[i] = s[indices[i]];
    }
    // Here, xx is equivalent to the number of unique indices; not xx + 1
    unsigned int* unique_indices = (unsigned int*) malloc((xx) * sizeof(unsigned int));
    for (unsigned int i = 0 ; i < indices_size; i++) {
        unique_indices[arg_unique_indices[i]] = (unsigned int) indices[i];
    }

    // expand offset
    unsigned int* expanded_offset = (unsigned int *) malloc(indices_size * sizeof(unsigned int));
    #pragma omp parallel for
    for (unsigned int i = 0 ; i < offsets_size; i++) {
        unsigned int start_offset = (unsigned int) offsets[i];
        unsigned int tmp = (i == offsets_size-1) ? indices_size : offsets[i + 1];
        for (unsigned int j = start_offset; j < tmp; j++) {
            expanded_offset[j] = i;  // ex) 0 0 1 1 1 1 2 ...
        }
    }

    float* d_coalesced = (float *) calloc ((xx) * embedding_dim, sizeof(float));

    // (1) Reduction omp, (2) Critical omp, (3) column-major loop performs worse
    for (unsigned int j = 0; j < indices_size; j++) {
        unsigned int offset_idx = expanded_offset[j];
        unsigned int coalesced_idx = arg_unique_indices[j];
        for (unsigned int dim = 0; dim < embedding_dim; dim++) {
            d_coalesced[coalesced_idx * embedding_dim + dim] += grads[offset_idx * embedding_dim + dim];  
        }
    }

    // UPDATE ON EMBEDDING TABLE
    #pragma omp parallel for
    for (int j = 0; j < (xx); j++) {  // Unique indices
        unsigned int w_idx = unique_indices[j];
        for (unsigned int dim = 0; dim < embedding_dim; dim++) {
            weights[w_idx * embedding_dim + dim] -= d_coalesced[j * embedding_dim + dim];
        }
    }
	free(d_coalesced);
    free(expanded_offset);
    free(arg_unique_indices);
    free(unique_indices);
}
