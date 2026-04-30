CC     = riscv64-linux-gnu-gcc
AR     = riscv64-linux-gnu-ar
CFLAGS = -I.. -I../include

SRC = ../libgpgpu.c ../src/buddy.c ../src/slab_cache.c


test_fmadd:
	$(CC) $(CFLAGS) -o test_fmadd test_fmadd.c $(SRC) -lm

test_gemm:
	$(CC) $(CFLAGS) -o test_gemm test_gemm.c $(SRC) -lm

test_diver:
	$(CC) $(CFLAGS) -o test_diver test_divergence.c $(SRC) -lm

deploy:
	scp -P 12055 test_fmadd test_gemm test_diver root@localhost:/root/

clean:
	rm -f test_fmadd


