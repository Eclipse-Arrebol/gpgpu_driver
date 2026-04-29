obj-m += gpgpu_drv.o

KDIR := /home/hp/gpgpu-driver/kernel-headers
ARCH := riscv
CROSS_COMPILE := riscv64-linux-gnu-
BUILD_DIR := $(PWD)/build
CC := $(CROSS_COMPILE)gcc
VM_USER := root
VM_HOST := localhost
VM_PORT := 12055
VM_DIR  := /root/

.PHONY: all driver test clean deploy

all: driver test


driver:
	make -C $(KDIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules
	mkdir -p $(BUILD_DIR)
	cp $(PWD)/gpgpu_drv.ko $(BUILD_DIR)/

test:
	mkdir -p $(BUILD_DIR)
	$(CC) -o $(BUILD_DIR)/test_gpgpu test/test_gpgpu.c

deploy:
	scp -P $(VM_PORT) $(BUILD_DIR)/gpgpu_drv.ko $(VM_USER)@$(VM_HOST):$(VM_DIR)
	scp -P $(VM_PORT) $(BUILD_DIR)/test_gpgpu $(VM_USER)@$(VM_HOST):$(VM_DIR)
clean:
	make -C $(KDIR) M=$(PWD) clean
	rm -rf $(BUILD_DIR)