#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <x86intrin.h>
#include <string.h>
#include <errno.h>
#include <time.h>

//#define DEBUG
#define SIZE_ALIGN 4096


typedef enum {false, true} bool;

/*
static inline bool is_aligned_uint(void *pointer) {
    return (((unsigned int)pointer & (SIZE_ALIGN - 1)) == 0);
}
*/
#define is_aligned_uint(POINTER) \
    (((unsigned long)(const void *)(POINTER)) % (SIZE_ALIGN) == 0)


int ack_polling(int fd, off_t mem_offset);
void write_block_data_msg4b(int fd, off_t mem_offset, unsigned int wdata);
void write_block_bulk_random_long(int fd, off_t mem_offset, unsigned int size1, unsigned int size2, unsigned int upper1, unsigned int upper2, bool sort2);
void write_block_bulk_random_float(int fd, off_t mem_offset, unsigned int size1);

int test_main(int argc, char *argv[]);
int static compare (const void* first, const void* second)
{
    if (*(long*)first > *(long*)second)
        return 1;
    else if (*(long*)first < *(long*)second)
        return -1;
    else
        return 0;
}

int main(int argc, char *argv[])
{
    test_main(argc, argv);
    return 0;
}

int ack_polling(int fd, off_t mem_offset) {
    unsigned int *read_buf;
	read_buf = aligned_alloc(SIZE_ALIGN, 4);
	
	printf("start polling\n");
    for (;;) {
        sleep(1);
        lseek(fd, mem_offset, SEEK_SET);
		read(fd, read_buf, 4);
        if (read_buf[0] == 1) {
            break;
        }
    }
    free(read_buf);
    return 1;
}


void write_block_bulk_data(int fd, off_t mem_offset, unsigned int size1, unsigned int* wdata) {
    lseek(fd, mem_offset, SEEK_SET);
    if(is_aligned_uint(wdata))
        write(fd, wdata, 4 * size1);
}


void write_block_bulk_random_long(int fd, off_t mem_offset, unsigned int size1, unsigned int size2, unsigned int upper1, unsigned int upper2, bool sort2) {
    lseek(fd, mem_offset, SEEK_SET);
    long *write_buf;

    size_t bs = sizeof(long) * (size1 + size2);
    write_buf = (long*) aligned_alloc(SIZE_ALIGN, bs);

    for(unsigned int i = 0; i < size1; i++){
        write_buf[i] = (long) (rand() % upper1);  // ex. indices
    }

    for(unsigned int i = size1; i < size2; i++){
        write_buf[i] = (long) (rand() % upper2);  // ex. offset
    }
    if (true == sort2)
        qsort(write_buf + size1, size2, sizeof(long), compare);

    write(fd, write_buf, bs);
    free(write_buf);
}

void write_block_bulk_random_float(int fd, off_t mem_offset, unsigned int size1) {
    lseek(fd, mem_offset, SEEK_SET);
    float *write_buf;
    size_t bs = 4 * (size1);
    write_buf = aligned_alloc(SIZE_ALIGN, bs);

    for(unsigned int i = 0; i < size1; i++){
        write_buf[i] = 2 * ((float)rand()/(float)(RAND_MAX)) - 1;  // ex. weight, grad
    }
    write(fd, write_buf, bs);
    free(write_buf);
}


void write_block_data_msg4b(int fd, off_t mem_offset, unsigned int wdata) {
    lseek(fd, mem_offset, SEEK_SET);
    unsigned int *write_buf;
    write_buf = aligned_alloc(SIZE_ALIGN, 4);
    write_buf[0] = (unsigned int) wdata;
    write(fd, write_buf, 4);
    free(write_buf);
}


int test_main(int argc, char *argv[]) {
    if (argc < 2){
        printf("Usage: %s I\n", argv[0]);
        printf("Usage: %s F\n", argv[0]);
        printf("Usage: %s B\n", argv[0]);
        return 0;
    }

    srand((unsigned int)time(NULL));

    /*
     *  #=================================
     *  # OFFSET CALC.
     *  #=================================
     *  let offset_h=$((offset>>32));
     *  let offset_l=$((offset%0x100000000));
     */
    // offset_h = (unsigned int)atoi(argv[3]);
    // offset_l = (unsigned int)atoi(argv[4]);
    // wdata    = strtoul(argv[5], NULL, 0);
    // offset = (off_t)(offset_h<<32) + (off_t)(offset_l);

    char cmd = argv[1][0];
    off_t phys_addr_query, phys_addr_mail, phys_addr_ack;
    off_t grad_phys_addr;
    if (cmd == 'I' || cmd == 'F') {
        phys_addr_query = 0x800000000;
        phys_addr_mail = 0xa0010000;
        phys_addr_ack = 0xa0010008;
    }
    else if (cmd == 'B') {
        phys_addr_query = 0x800000000;
        phys_addr_mail = 0xa0010000;
        phys_addr_ack = 0xa0010008;
        grad_phys_addr = 0x8000E4000;
    }

    // Declare mailbox
    char *fname_wq = "/dev/qdma84000-MM-0";
    char *fname_rq = "/dev/qdma84000-MM-0";
    int fd_wq = open(fname_wq, O_WRONLY|O_TRUNC);
    if (fd_wq < 0) {
        printf("QDMA queue open error: %s, %s\n", fname_wq, strerror(errno));
        exit(1);
    }
	int fd_rq = open(fname_rq, O_RDONLY);
	if(fd_rq < 0)
	{
		printf("open error: %s, %s\n", fname_rq, strerror(errno));
		exit(1);
	}

    unsigned int msg;
    /////////////////////////////////////
    // Execute Function
    if (cmd == 'I') { // Initialization Emb Table
#ifdef DEBUG
        int emb_num = 10;
        int dimension = 16;
#else
        int emb_num = 1000000;        
        //int emb_num = 200000;         Failed
        //int emb_num = 25000;
        int dimension = 64;
#endif

        printf("emb_num = %d(%#x)\n", emb_num, emb_num);
        // Changing Opcode at last for safety
        msg = (dimension << 24) + (emb_num << 3) + 1;
        write_block_data_msg4b(fd_wq, phys_addr_mail, msg);
        printf("Init Msg Recorded.\n");
        //write_block_data_msg4b(fd_wq, phys_addr_mail, 0);  // Clear Mailbox
        //printf("Msg Cleared.\n");
       //ACK_POLLING();
    }
    else if (cmd == 'F') {  // Forward
#ifdef DEBUG  // offset
        unsigned int offset_size = 3;
        unsigned int indices_size = 7;
        write_block_bulk_random_long(fd_wq, phys_addr_query, offset_size, indices_size, indices_size, 10, true);  // 10: emb_num
#else
        unsigned int offset_size = 1024;
        unsigned int indices_size = 1024*80;
        write_block_bulk_random_long(fd_wq, phys_addr_query, offset_size, indices_size, indices_size, (RAND_MAX / 1000000 + 1), true);  // 10e6: emb_num
#endif

        msg = (indices_size << 14) + (offset_size << 3) + 2;
        write_block_data_msg4b(fd_wq, phys_addr_mail, msg);
        printf("Forward Msg Recorded.\n"); 
        //write_block_data_msg4b(fd_wq, phys_addr_mail, 0);  // Clear Mailbox
        //printf("Msg Cleared.\n");
        //printf("phys_addr_ack = %#08lx", (unsigned long)phys_addr_ack);
        if(ack_polling(fd_rq, phys_addr_ack))
            printf("Received ACK: Forward\n");
    }

    else if (cmd == 'B') {  // Backward
#ifdef DEBUG  // offset
        unsigned int offset_size = 3;
        unsigned int indices_size = 7;
        write_block_bulk_random_long(fd_wq, phys_addr_query, offset_size, indices_size, indices_size, 10, true);  // 10: emb_num
#else
        unsigned int offset_size = 1024;
        unsigned int indices_size = 1024*80;
        write_block_bulk_random_long(fd_wq, phys_addr_query, offset_size, indices_size, indices_size, (RAND_MAX / 1000000 + 1), true);  // 10e6: emb_num
#endif

        // Generate grad
        // Dimension Hardcoded!! 16 or 64
#ifdef DEBUG
        write_block_bulk_random_float(fd_wq, grad_phys_addr, offset_size * 16);  // 1~-1
#else
        write_block_bulk_random_float(fd_wq, grad_phys_addr, offset_size * 64);  // 1~-1
#endif
        // Changing Opcode at last for safety
        msg = (indices_size << 14) + (offset_size << 3) + 3;
        write_block_data_msg4b(fd_wq, phys_addr_mail, msg);
        printf("Msg Recorded.\n");
        //write_block_data_msg4b(fd_wq, phys_addr_mail, 0);  // Clear Mailbox
        //printf("Msg Cleared.\n");
        //ACK_POLLING();
    }

    close(fd_wq);
    close(fd_rq);
    return 0;
}

