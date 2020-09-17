#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <time.h>

#define MAX_PRINT 32
#define MAIL_SIZE 4
#define ACK_SIZE 4
#define CMD_SIZE 64
#define GRAD_SIZE 262144

#define DEBUG

#define ACK_POLLING() ({\
   for (;;) { \
        if(ack_mem[ack_offset] == 1) { \
            printf("RECEIVED ACK! \n"); \
            break; \
        } \
   } \
})

/*
 * Test Script for Embedding Standalone Test
 */
int test_main(int argc, char *argv[]);
int static compare (const void* first, const void* second)
{
    if (*(int*)first > *(int*)second)
        return 1;
    else if (*(int*)first < *(int*)second)
        return -1;
    else
        return 0;
}

int main(int argc, char *argv[]) {
	test_main(argc, argv);

	return 0;
}

int test_main(int argc, char *argv[]) {
    if (argc < 2){
		printf("Usage: ./%s R <phys_addr> <size>\n", argv[0]);
		printf("Usage: ./%s I\n", argv[0]);
        printf("Usage: ./%s F\n", argv[0]);
        printf("Usage: ./%s B\n", argv[0]);
		return 0;
	}
    
    char cmd = argv[1][0];
    size_t size;
    size_t len;
    off_t phys_addr;
    off_t phys_addr_mail;
    off_t phys_addr_ack;
    off_t grad_phys_addr;

    size_t pagesize = sysconf(_SC_PAGE_SIZE);

    srand(time(0)); 

	if(cmd == 'R'){
        phys_addr = strtoul(argv[2], NULL, 0);
		size = strtoul(argv[3], NULL, 0);
		len = size / sizeof(int);
	}
    else if (cmd == 'I' || cmd == 'F') {
        phys_addr = 0x800000000;
        phys_addr_mail = 0xa0010000;
        phys_addr_ack = 0xa0010008;
        size = CMD_SIZE;
    }
    else if (cmd == 'B') {
        phys_addr = 0x800000000;
        phys_addr_mail = 0xa0010000;
        phys_addr_ack = 0xa0010008;
        grad_phys_addr = 0x800093000;
        size = CMD_SIZE;
    }

    // Truncate offset to a multiple of the page size, or mmap will fail.
    off_t page_base = (phys_addr / pagesize) * pagesize;
    off_t page_offset = phys_addr - page_base;

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    unsigned int *mem = (unsigned int*) mmap(NULL, page_offset + size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_base);
    if (mem == MAP_FAILED) {
        perror("Can't map memory");
        return -1;
    }
    int offset = page_offset / sizeof(int);

    // Alloc mailbox
    off_t mail_page_base;
    off_t mail_page_offset;
    int mail_fd;
    unsigned int *mail_mem = NULL;
    if (cmd != 'R') {
        mail_page_base = (phys_addr_mail / pagesize) * pagesize;
        mail_page_offset = phys_addr_mail - mail_page_base;

        mail_fd = open("/dev/mem", O_RDWR | O_SYNC);
        mail_mem = (unsigned int*) mmap(NULL, mail_page_offset + MAIL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mail_fd, mail_page_base);
        if (mail_mem == MAP_FAILED) {
            perror("Can't map mail_memory");
            return -1;
        }
    }
    int mail_offset = mail_page_offset / sizeof(int);

/*
    // Alloc ACK
    off_t ack_page_base;
    off_t ack_page_offset;
    int ack_fd;
    unsigned int *ack_mem = NULL;
    if (cmd == 'F' || cmd == 'B') {
        ack_page_base = (phys_addr_ack / pagesize) * pagesize;
        ack_page_offset = phys_addr_ack - ack_page_base;

        ack_fd = open("/dev/mem", O_RDWR | O_SYNC);
        ack_mem = (unsigned int*) mmap(NULL, ack_page_offset + MAIL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, ack_fd, ack_page_base);
        if (ack_mem == MAP_FAILED) {
            perror("Can't map ack mail_memory");
            return -1;
        }
    }

    int ack_offset = ack_page_offset / sizeof(int);
*/
    if(cmd == 'R'){  // Read
        size_t i;
        //int i;
        int j;
        off_t tmp;
        if(len <= MAX_PRINT){
            for (i=0, tmp=phys_addr; i<len; ++i, tmp+=sizeof(int)){
                printf("mem[%#lx]: %08x\n", tmp, mem[offset+i]);
            }
        }
        else{
            for (i=0, tmp=phys_addr; i<MAX_PRINT; ++i, tmp+=sizeof(int)){
                printf("mem[%#lx]: %08x\n", tmp, mem[offset+i]);
            }
            printf("...\n");
            
            i = len-2;
            tmp = phys_addr + (i * sizeof(int));
            printf("mem[%#lx]: %08x\n", tmp, mem[offset+i]);
            i = len-1;
            tmp = phys_addr + (i * sizeof(int));
            printf("mem[%#lx]: %08x\n", tmp, mem[offset+i]);
        }
        printf("\n");

    }
    else if (cmd == 'I') { // Initialization Emb Table
#ifdef DEBUG
        int emb_num = 10;
        int dimension = 64;
#else
        int emb_num = 1000000;
        int dimension = 64;
#endif
        // Changing Opcode at last for safety
        mail_mem[mail_offset] = (dimension << 24) + (emb_num << 3) + 1;  // dim, num, opcode (last bit)
        mail_mem[mail_offset] = 0;
       //ACK_POLLING();
    }
    else if (cmd == 'F') {  // Forward
#ifdef DEBUG  // offset
        unsigned int offset_size = 3;
        unsigned int indices_size = 7;
        // Rand Offset
        for (int jj = 0 ; jj < offset_size ; ++jj) {
            mem[offset + jj] = rand() % indices_size;
        }
        // Rand Indices
        for (int jj = 0 ; jj < indices_size ; ++jj) {
            mem[offset + offset_size + jj] = rand() % (10);  // plug in embedding size
        }
#else
        unsigned int offset_size = 1024;
        unsigned int indices_size = 1024*80;
        // Rand Offset
        for (int jj = 0 ; jj < offset_size ; ++jj) {
            mem[offset + jj] = rand() % indices_size;
        }
        // Rand Indices
        for (int jj = 0 ; jj < indices_size ; ++jj) {
            mem[offset + offset_size + jj] = rand() % (RAND_MAX / 1000000 + 1);  // plug in embedding size
        }
#endif
        qsort(mem + offset, offset_size, sizeof(int), compare);
        // Changing Opcode at last for safety
        mail_mem[mail_offset] = (indices_size << 14) + (offset_size << 3) + 2;
        mail_mem[mail_offset] = 0;
        //ACK_POLLING();
    }

    else if (cmd == 'B') {  // Backward
        off_t grad_page_base = (grad_phys_addr / pagesize) * pagesize;
        off_t grad_page_offset = grad_phys_addr - grad_page_base;

        int grad_fd = open("/dev/mem", O_RDWR | O_SYNC);
        float *grad_mem = (float*)mmap(NULL, grad_page_offset + GRAD_SIZE, 
                        PROT_READ | PROT_WRITE, MAP_SHARED, grad_fd, grad_page_base);
        if (grad_mem == MAP_FAILED) {
            perror("Can't emb map memory");
            return -1;
        }
        int grad_offset = grad_page_offset / sizeof(float);

#ifdef DEBUG  // offset
        unsigned int offset_size = 3;
        unsigned int indices_size = 7;
        // Rand Offset
        for (int jj = 0 ; jj < offset_size ; ++jj) {
            mem[offset + jj] = rand() % indices_size;
        }
        // Rand Indices
        for (int jj = 0 ; jj < indices_size ; ++jj) {
            mem[offset + offset_size + jj] = rand() % (10);  // plug in embedding size
        }
#else
        unsigned int offset_size = 1024;
        unsigned int indices_size = 1024*80;
        // Rand Offset
        for (int jj = 0 ; jj < offset_size ; ++jj) {
            mem[offset + jj] = rand() % indices_size;
        }
        // Rand Indices
        for (int jj = 0 ; jj < indices_size ; ++jj) {
            mem[offset + offset_size + jj] = rand() % (RAND_MAX / 1000000 + 1);  // plug in embedding size
        }
#endif
        qsort(mem + offset, offset_size, sizeof(int), compare);

        // Generate grad
        // Dimension Hardcoded!!
#ifdef DEBUG
        for (int jj = 0; jj < offset_size * 16; ++jj) {
            grad_mem[grad_offset + jj] = ((float)rand()/(float)(RAND_MAX));
        }
#else
        for (int jj = 0; jj < offset_size * 64; ++jj) {
            grad_mem[grad_offset + jj] = ((float)rand()/(float)(RAND_MAX));
        }
#endif
        // Changing Opcode at last for safety
        mail_mem[mail_offset] = (indices_size << 14) + (offset_size << 3) + 3;
        mail_mem[mail_offset] = 0;
		//ACK_POLLING();
        close(grad_fd);
        munmap(grad_mem, (grad_page_offset+GRAD_SIZE));
    }

    close(fd);
    munmap(mem, (page_offset+size));
    if (cmd != 'R') {
        close(mail_fd);
        munmap(mail_mem, (mail_page_offset+MAIL_SIZE));
        //close(ack_fd);
       // munmap(ack_mem, (ack_page_offset+ACK_SIZE));
    }
    return 0;
}
