CC     = riscv64-linux-gnu-gcc
AR     = riscv64-linux-gnu-ar
CFLAGS = -Wall -I./include -I.

SRC = src/buddy.c src/slab_cache.c src/kv_cache.c libgpgpu.c src/weight_loader.c src/transformer.c
OBJ = $(SRC:.c=.o)

libgpgpu.a: $(OBJ)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) libgpgpu.a