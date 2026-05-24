/*
 * test_vram_4gb.c — 一次性测试,证明 device VRAM BAR 已扩容到 4GB
 *
 * 设计原则(对应原则 21):
 *   pattern 自语义编码 offset,fail 时直接读出"地址回绕到哪儿去了"。
 *
 * 唯一证明:BAR 全 4GB 可达,无截断,无地址回绕。
 *   不证明 libgpgpu allocator 配置正确(那是单独的事)。
 *
 * 运行方式:
 *   直接 mmap /sys/bus/pci/devices/.../resource0(或你 device 暴露 VRAM 的 BAR),
 *   不经过 libgpgpu 的任何抽象。
 *
 * 用法:
 *   ./test_vram_4gb /sys/bus/pci/devices/0000:XX:XX.X/resource0
 *   (具体 BAR 编号看你 device,通常 VRAM 是 resource0 或 resource2)
 *
 * 退出码:0 = PASS,非 0 = FAIL(具体见各 case)
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* ─── 常量 ──────────────────────────────────────────────────── */

#define VRAM_SIZE_4GB (4ULL * 1024 * 1024 * 1024) /* 4294967296 */
#define VRAM_SIZE_2GB (2ULL * 1024 * 1024 * 1024) /* 旧上限,用于诊断截断 */

/* sentinel:写前先把目标 8B 设成这个值,防"恰好原本是 pattern" */
#define SENTINEL 0xDEADBEEFCAFEBABEULL

/* pattern 编码:high 32 = magic, low 32 = offset >> 3 (8B 对齐后 offset)
 *
 *   写 offset=X 的 8B,值 = 0xA5A5A5A5_00000000 | (X >> 3)
 *
 *   fail 时 got 的 low 32 直接告诉你"这个位置读到的是哪个 offset 写下去的数据"
 *   ─ 这就是原则 21 的精神:bit-pattern 自语义,fail 输出自带语义。
 */
#define PATTERN_MAGIC 0xA5A5A5A500000000ULL
#define MAKE_PATTERN(off) (PATTERN_MAGIC | ((uint64_t)((off) >> 3)))
#define PATTERN_TO_OFF(p) (((uint64_t)((p) & 0xFFFFFFFFULL)) << 3)
#define PATTERN_MAGIC_OK(p) (((p) & 0xFFFFFFFF00000000ULL) == PATTERN_MAGIC)

/* ─── 测试 offset 表 ─────────────────────────────────────────── */

/* 精心挑选的 spot ─ 每个都有诊断意义。
 *
 *   关键约束:
 *   (1) 全部 8B 对齐
 *   (2) 跨越 2GB 边界,确保任何 BAR 截断都会在某对 spot 间留下回绕指纹
 *   (3) 覆盖 2 的幂(1GB / 2GB / 3GB / 4GB-8),用于排查地址线坏位
 *   (4) 覆盖 2GB 边界的"前后紧贴",这是最容易截断的位置
 */
static const uint64_t test_offsets[] = {
    /* 低位边界 */
    0x00000000ULL, /* offset 0 */
    0x00000008ULL, /* offset 8(最低位邻居) */

    /* 1GB / 1.5GB 中段(健康对照,无截断时也应该过) */
    0x40000000ULL, /* 1GB */
    0x60000000ULL, /* 1.5GB */

    /* 2GB 边界 ─ 旧上限,最危险的截断点 */
    0x7FFFFFF8ULL, /* 2GB - 8(旧上限内最高 8B) */
    0x80000000ULL, /* 2GB(旧上限第一字节) */
    0x80000008ULL, /* 2GB + 8 */

    /* 高 2GB 中段(只有真 4GB 才能正常读写的区域) */
    0xA0000000ULL, /* 2.5GB */
    0xC0000000ULL, /* 3GB */
    0xE0000000ULL, /* 3.5GB */

    /* 高位边界 */
    0xFFFFFFF8ULL, /* 4GB - 8(BAR 最高 8B) */
};

#define N_TESTS (sizeof(test_offsets) / sizeof(test_offsets[0]))

/* ─── 主流程 ────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s /sys/bus/pci/devices/.../resourceN\n",
                argv[0]);
        fprintf(stderr, "       (N is the VRAM BAR index — check lspci -vv)\n");
        return 2;
    }

    const char *resource_path = argv[1];

    /* Step 1: 打开 resource 文件,确认 size = 4GB */
    int fd = open(resource_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "FAIL: open(%s): %s\n", resource_path, strerror(errno));
        return 3;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "FAIL: fstat: %s\n", strerror(errno));
        close(fd);
        return 3;
    }

    printf("BAR file size: %lld bytes (0x%llx)\n", (long long)st.st_size,
           (long long)st.st_size);

    if ((uint64_t)st.st_size != VRAM_SIZE_4GB) {
        fprintf(stderr, "FAIL: BAR size = %lld, expected %lld (4GB)\n",
                (long long)st.st_size, (long long)VRAM_SIZE_4GB);
        if ((uint64_t)st.st_size == VRAM_SIZE_2GB) {
            fprintf(stderr, "  → BAR still at 2GB. Check:\n");
            fprintf(stderr, "    (1) QEMU launch arg vram_size=4294967296\n");
            fprintf(
                stderr,
                "    (2) device PCI BAR registration uses 64-bit MEM type\n");
            fprintf(stderr,
                    "        (PCI_BASE_ADDRESS_MEM_TYPE_64, not _TYPE_32)\n");
        }
        close(fd);
        return 4;
    }

    /* Step 2: mmap 整个 4GB BAR */
    void *vram =
        mmap(NULL, VRAM_SIZE_4GB, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (vram == MAP_FAILED) {
        fprintf(stderr, "FAIL: mmap 4GB: %s\n", strerror(errno));
        fprintf(stderr, "  → If errno=ENOMEM, host RAM may be tight; need 4GB "
                        "free virt addr space.\n");
        close(fd);
        return 5;
    }

    volatile uint64_t *p = (volatile uint64_t *)vram;
    printf("mmap OK: vram @ %p, len 4GB\n\n", vram);

    /* Step 3: 写入 sentinel + pattern */
    printf("Phase 1: write sentinel then pattern at %zu spots...\n", N_TESTS);
    for (size_t i = 0; i < N_TESTS; i++) {
        uint64_t off = test_offsets[i];
        size_t   idx = off >> 3;

        /* 先写 sentinel 防"恰好原本是 pattern" */
        p[idx] = SENTINEL;

        /* 写真正的 pattern */
        p[idx] = MAKE_PATTERN(off);

        printf("  [%2zu] off=0x%010" PRIx64 " wrote 0x%016" PRIx64 "\n", i, off,
               MAKE_PATTERN(off));
    }

    /* Step 4: 读回来验证 + 反向扫描检测地址回绕 */
    printf("\nPhase 2: read back and verify...\n");

    int n_pass = 0, n_fail = 0;
    for (size_t i = 0; i < N_TESTS; i++) {
        uint64_t off      = test_offsets[i];
        uint64_t expected = MAKE_PATTERN(off);
        uint64_t got      = p[off >> 3];

        if (got == expected) {
            printf("  [%2zu] off=0x%010" PRIx64 " PASS  got=0x%016" PRIx64 "\n",
                   i, off, got);
            n_pass++;
        } else {
            printf("  [%2zu] off=0x%010" PRIx64 " FAIL  got=0x%016" PRIx64
                   "  expected=0x%016" PRIx64 "\n",
                   i, off, got, expected);

            /* 原则 21: 用 pattern 自语义解码 fail 形态 */
            if (got == SENTINEL) {
                printf(
                    "       → got = SENTINEL: write was silently dropped.\n");
                printf(
                    "         BAR may be read-only at this offset, or write\n");
                printf(
                    "         routed elsewhere. Suspect MMIO routing bug.\n");
            } else if (got == 0xFFFFFFFFFFFFFFFFULL) {
                printf("       → got = all-0xFF: classic 'PCI device not "
                       "responding'.\n");
                printf("         This offset is BEYOND the actual BAR size.\n");
                printf("         → BAR was truncated, not really 4GB.\n");
            } else if (got == 0x0000000000000000ULL) {
                printf(
                    "       → got = all-0: write went nowhere, or area is\n");
                printf("         unmapped and host returns 0. Suspect partial "
                       "BAR.\n");
            } else if (PATTERN_MAGIC_OK(got)) {
                uint64_t aliased_off = PATTERN_TO_OFF(got);
                printf("       → got magic OK, decoded source offset = "
                       "0x%010" PRIx64 "\n",
                       aliased_off);
                printf("         → ADDRESS ALIASING: offset 0x%" PRIx64
                       " reads back data written to 0x%" PRIx64 ".\n",
                       off, aliased_off);
                printf("         Difference = 0x%" PRIx64
                       " — likely BAR truncated\n",
                       (off > aliased_off ? off - aliased_off
                                          : aliased_off - off));
                printf("         to %s and high bits are being masked.\n",
                       (off - aliased_off == VRAM_SIZE_2GB)
                           ? "2GB"
                           : "some smaller size");
            } else {
                printf("       → got is neither sentinel nor valid pattern.\n");
                printf(
                    "         Could be uninitialized VRAM, foreign data, or\n");
                printf(
                    "         write/read width mismatch. Inspect raw hex.\n");
            }
            n_fail++;
        }
    }

    /* Step 5: 总结 */
    printf("\n=================================\n");
    printf("Results: %d/%zu PASS, %d FAIL\n", n_pass, N_TESTS, n_fail);
    printf("=================================\n");

    munmap(vram, VRAM_SIZE_4GB);
    close(fd);

    return (n_fail == 0) ? 0 : 1;
}