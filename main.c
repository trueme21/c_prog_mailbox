#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/time.h>

#define NDP_COMPILE

#ifdef NDP_COMPILE
    #include "include/ndp_sls.h"
#endif

#define MAX_PRINT 32

// Refer to excel file
#define CMD_START_ADDR 0xA0020000
//#define ACK_START_ADDR 0xA0020000  // After the NDP execution, send finished signal to the host
#define QUERIES_START_ADDR 0x800000000
#define OUTPUT_START_ADDR 0x8000A3000
#define GRAD_START_ADDR 0x8000E4000
#define EMB_START_ADDR 0x800125000

#define CMD_SIZE 64  // TODO
//#define ACK_SIZE 4  // TODO
#define QUERIES_SIZE 663552  // TODO
#define OUTPUT_SIZE 262144
#define GRAD_SIZE 262144
// EMB_SIZE is defined at initialization

int main()
{
	struct timeval start_point , end_point;
	double operating_time;	
    size_t pagesize = sysconf(_SC_PAGE_SIZE);
    int cmd_fd = open("/dev/mem", O_RDWR | O_SYNC);
    //int ack_fd = open("/dev/mem", O_RDWR | O_SYNC);
    //int queries_fd = open("/dev/mem", O_RDWR | O_SYNC);
    int queries_fd = open("/dev/mem", O_RDWR);
    int output_fd = open("/dev/mem", O_RDWR | O_SYNC);
    //int output_fd = open("/dev/mem", O_RDWR);
    int grad_fd = open("/dev/mem", O_RDWR | O_SYNC);
    //int emb_fd = open("/dev/mem", O_RDWR | O_SYNC);
    int emb_fd = open("/dev/mem", O_RDWR);

    // Allocate CMD space
    // Truncate offset to a multiple of the page size, or mmap will fail.
    off_t cmd_page_base = (CMD_START_ADDR / pagesize) * pagesize;
    printf("cmd_page_base = %08x\n", cmd_page_base);
    
    off_t cmd_page_offset = CMD_START_ADDR - cmd_page_base;
    unsigned int *cmd_mem = (unsigned int*)mmap(NULL, cmd_page_offset + CMD_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, cmd_fd, cmd_page_base);
    unsigned int cmd_mem_t;
    if (cmd_mem == MAP_FAILED) {
        perror("Can't map CMD memory");
        return -2;
    }

    /*
    off_t ack_page_base = (ACK_START_ADDR / pagesize) * pagesize;
    off_t ack_page_offset = ACK_START_ADDR - ack_page_base;
    unsigned int *ack_mem = (unsigned int*)mmap(NULL, ack_page_offset + CMD_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ack_fd, ack_page_base);
    if (ack_mem == MAP_FAILED) {
        perror("Can't map ACK memory");
        return -2;
    }
    */

    // Allocate QUERIES space
    // Offset + Indices
    off_t queries_page_base = (QUERIES_START_ADDR / pagesize) * pagesize;
    off_t queries_page_offset = QUERIES_START_ADDR - queries_page_base;
    long *queries_mem = (long*)mmap(NULL, queries_page_offset + QUERIES_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, queries_fd, queries_page_base);
    if (queries_mem == MAP_FAILED) {
        perror("Can't map QUERIES memory");
        return -2;
    }

    // Allocate Output space 
    off_t output_page_base = (OUTPUT_START_ADDR / pagesize) * pagesize;
    off_t output_page_offset = OUTPUT_START_ADDR - output_page_base;
    float *output_mem = (float*)mmap(NULL, output_page_offset + OUTPUT_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, output_fd, output_page_base);
    if (output_mem == MAP_FAILED) {
        perror("Can't map OUTPUT memory");
        return -2;
    }

    // Allocate Grad space
    off_t grad_page_base = (GRAD_START_ADDR / pagesize) * pagesize;
    off_t grad_page_offset = GRAD_START_ADDR - grad_page_base;
    float *grad_mem = (float*)mmap(NULL, grad_page_offset + GRAD_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, grad_fd, grad_page_base);
    if (grad_mem == MAP_FAILED) {
        perror("Can't map GRAD memory");
        return -2;
    }
    float* emb_mem = NULL;
    
    // CMD Decoder: command
    unsigned int queries_offset = queries_page_offset / sizeof(long);
    unsigned int cmd_offset = cmd_page_offset / sizeof(int) + 2;  // Align
    unsigned int ack_offset = cmd_page_offset / sizeof(int);
    unsigned int output_offset = output_page_offset / sizeof(float);
    unsigned int grad_offset = grad_page_offset / sizeof(int);
    unsigned int embed_offset;
    unsigned int embedding_num;
    unsigned int dimension;
    unsigned int emb_size;
    off_t emb_page_offset;

	// cleanup
	cmd_mem[ack_offset] = 0;

    printf("Allocation Completed. Executing First Loop...\n");
    for(;;) {
        sleep(1);

        // Allocate & Initialize Embedding Table
        /* 
         * Opcode (3bit) + Embedding Num (21bit) + Dimension (8bit)
         */
        cmd_mem_t = cmd_mem[cmd_offset];

        // printf("cmd_offset = %d\n", cmd_offset);
        // printf("cmd_mem[cmd_offset] = %08x\n", cmd_mem_t);
        if((cmd_mem_t & 0x00000007) == 0x00000001){
            printf("EMB Initialization Start...\n");
            embedding_num = (cmd_mem_t >> 3) & 0x001FFFFF;
            dimension = (cmd_mem_t >> 24) & 0x000000FF;
            emb_size = (unsigned int) (embedding_num * dimension);

            // Allocate Embedding table space
            off_t emb_page_base = (EMB_START_ADDR / pagesize) * pagesize;
            emb_page_offset = EMB_START_ADDR - emb_page_base;
            embed_offset = emb_page_offset / sizeof(float);

            emb_mem = (float*)mmap(NULL, emb_page_offset + sizeof(float) * emb_size, PROT_READ | PROT_WRITE, MAP_SHARED, emb_fd, emb_page_base);
            if (emb_mem == MAP_FAILED) {
                perror("Can't map EMB memory");
                return -2;
            }
            // Initialization emb table with random
            //int rnd=open("/dev/urandom", O_RDONLY);
            //read(2 * ((float)rnd / (float)(RAND_MAX)) - 1, emb_mem, sizeof(float)*emb_size); // -1 ~ 1
            //close(rnd);
            for (int jj = 0 ; jj < emb_size ; ++jj) {
                emb_mem[embed_offset + jj] = (float)rand()/(float)(RAND_MAX) ;  // plug in embedding size
            }
            printf("RAND Generated...\n");
            //cmd_mem[ack_offset] = 1;
            break; // Move to next forever loop
        }

        // Kill switch

        if((cmd_mem_t & 0x00000007) == 0x00000004){
            // free CMD and Output space
            printf("TERMINATION!\n");
            munmap(queries_mem, (queries_page_offset+QUERIES_SIZE));
            munmap(cmd_mem, (cmd_page_offset+CMD_SIZE));
            //munmap(ack_mem, (ack_page_offset+CMD_SIZE));
            munmap(output_mem, (output_page_offset+OUTPUT_SIZE));
            munmap(grad_mem, (grad_page_offset+GRAD_SIZE));
            close(queries_fd);
            close(cmd_fd);
            //close(ack_fd);
            close(output_fd);
            close(grad_fd);
            return -1;
        }
    }
    printf("Emb table Allocation Completed. Executing Second Loop...\n");
    for(;;) {
        sleep(1);
        cmd_mem_t = cmd_mem[cmd_offset];

        if((cmd_mem_t & 0x00000007) == 0x00000002){      
            /* 
             * Opcode (3bit) + Offset Size (11bit) + Indices Size (17bit)
             */ 
            printf("Inference Start...\n");
            unsigned int offset_size = (cmd_mem_t >> 3) & 0x000007FF;
            unsigned int indices_size = (cmd_mem_t >> 14) & 0x0001FFFF;
            // Don't have to parse dimension... already got from opcode==1
			gettimeofday(&start_point, NULL);
#ifdef NDP_COMPILE
            embedding_forward_simd(emb_mem + embed_offset, 
                                queries_mem + queries_offset + offset_size,  // indices, long
                                queries_mem + queries_offset,  // offset start address
                                0, 0, 0, dimension, offset_size,
                                0, 0, indices_size, 
                                output_mem + output_offset);
#endif
			gettimeofday(&end_point, NULL);
			operating_time = (double)(end_point.tv_sec)+(end_point.tv_usec)/1000000.0-(double)(start_point.tv_sec)-(double)(start_point.tv_usec)/1000000.0;
            printf("mmap write time : %f\n", operating_time);

            cmd_mem[ack_offset] = 1;
            //cmd_mem[ack_offset] = 0;
            printf("Inference Finished...\n");
        }

        if((cmd_mem_t & 0x00000007) == 0x00000003){
            /* 
             * Opcode (3bit) + Offset Size (11bit) + Indices Size (17bit)
             */ 
            printf("Backward Start...\n");
            unsigned int offset_size = (cmd_mem_t >> 3) & 0x000007FF;
            unsigned int indices_size = (cmd_mem_t >> 14) & 0x0001FFFF;
#ifdef NDP_COMPILE
			grad_coalesce_hash(grad_mem + grad_offset,
                                queries_mem + queries_offset + offset_size, // indices
                                queries_mem + queries_offset, // offset
                                indices_size, offset_size, 
                                emb_mem + embed_offset, 
                                dimension);
#endif
            cmd_mem[ack_offset] = 1;
            //cmd_mem[ack_offset] = 0;
            printf("Backward Finished...\n");
        }

        // Deallocate Embedding Table and Kill switch
        if((cmd_mem_t & 0x00000007) == 0x00000004){
            // free CMD and Output space
            printf("TERMINATION!\n");
            munmap(queries_mem, (queries_page_offset+QUERIES_SIZE));
            munmap(cmd_mem, (cmd_page_offset+CMD_SIZE));
            //munmap(ack_mem, (ack_page_offset+CMD_SIZE));
            munmap(output_mem, (output_page_offset+OUTPUT_SIZE));
            munmap(grad_mem, (grad_page_offset+GRAD_SIZE));
            munmap(emb_mem, (emb_page_offset+emb_size));
            close(queries_fd);
            close(cmd_fd);
            //close(ack_fd);
            close(output_fd);
            close(grad_fd);
            close(emb_fd);
            return -1;
        }
    }
    return 0;
}

