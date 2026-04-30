CC     = riscv64-linux-gnu-gcc
AR     = riscv64-linux-gnu-ar
CFLAGS = -I.. -I../include

SRC = ../libgpgpu.c ../src/buddy.c ../src/slab_cache.c

# 自动发现当前目录下所有的 .c 源文件（你的测试程序）
C_SRCS := $(wildcard *.c)
# 生成的可执行文件名：将 .c 替换为 空（去掉后缀）
TARGETS := $(C_SRCS:.c=)

# 默认目标：编译所有可执行文件
all: $(TARGETS)

# 模式规则：如何从任意 .c 文件生成对应的可执行文件
%: %.c
	$(CC) $(CFLAGS) -o $@ $< $(SRC) -lm

# 部署：将所有可执行文件 scp 到远程
deploy:
	scp -P 12055 $(TARGETS) root@localhost:/root/

# 清理：删除所有生成的可执行文件
clean:
	rm -f $(TARGETS)

.PHONY: all deploy clean