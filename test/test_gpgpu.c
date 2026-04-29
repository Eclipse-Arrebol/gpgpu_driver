#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <time.h>

#define GPGPU_IOC_MAGIC     'G'
#define GPGPU_IOC_DISPATCH  _IOW(GPGPU_IOC_MAGIC, 0, struct gpgpu_dispatch_args)

struct gpgpu_dispatch_args {
    uint64_t kernel_addr;
    uint64_t kernel_args;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t shared_mem_size;
};




int main()
{
    int fd;
    int ret;
    struct timespec t1, t2;
    fd = open("/dev/gpgpu", O_RDWR);
    if(fd<0)
    {
        return -1;
    }
    uint32_t *ctrl = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ctrl == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
    uint32_t *vram = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 1*4096);
    if (vram == MAP_FAILED) {
        perror("mmap failed");
        return -1;
    }
    vram[0] = 0x12345678;
    printf("read back: 0x%08x\n", vram[0]);
    printf("======================ioctl测试=================================\n");
    struct gpgpu_dispatch_args args = {
        .kernel_addr    = 0,        // kernel 在 VRAM 里的地址，暂时填 0
        .kernel_args    = 0,        // 参数地址，暂时填 0
        .grid_dim       = {1, 1, 1},  // 1个block
        .block_dim      = {1, 1, 1},  // 每个block 1个线程
        .shared_mem_size = 0,
    };
    clock_gettime(CLOCK_MONOTONIC, &t1);
    ret = ioctl(fd, GPGPU_IOC_DISPATCH, &args);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    long ms = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_nsec - t1.tv_nsec) / 1000000;
    printf("ioctl took %ld ms\n", ms);
    if(ret)
    {
        return ret;
    }
    printf("read back: 0x%08x\n", ctrl[0x0104/4]);

    

    return 0;
}


