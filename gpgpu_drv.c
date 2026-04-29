#include <linux/cdev.h>   // struct cdev, cdev_init, cdev_add
#include <linux/device.h> // class_create, device_create
#include <linux/fs.h>     // file_operations, alloc_chrdev_region
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/wait.h>

// ioctrl
#define GPGPU_IOC_MAGIC 'G'
#define GPGPU_IOC_DISPATCH _IOW(GPGPU_IOC_MAGIC, 0, struct gpgpu_dispatch_args)

/*常用宏*/
#define GPGPU_MSIX_VECTORS 3
#define GPGPU_IRQ_KERNEL_DONE (1 << 0) /* 内核执行完成中断 */
#define GPGPU_IRQ_DMA_DONE (1 << 1)    /* DMA 传输完成中断 */
#define GPGPU_IRQ_ERROR (1 << 2)       /* 错误中断 */

/* 中断控制寄存器组 (0x0200 - 0x02FF): 中断使能和状态管理 */
#define GPGPU_REG_IRQ_ENABLE 0x0200 /* 中断使能掩码 */
#define GPGPU_REG_IRQ_STATUS 0x0204 /* 中断状态 (挂起的中断) */
#define GPGPU_REG_IRQ_ACK 0x0208    /* 中断确认 (写 1 清除) */

// gpu寄存器
#define GPGPU_REG_VRAM_SIZE_LO 0x000C    /* 显存大小低 32 位 */
#define GPGPU_REG_VRAM_SIZE_HI 0x0010    /* 显存大小高 32 位 */
#define GPGPU_REG_KERNEL_ADDR_LO 0x0300  /* 内核代码地址低 32 位 */
#define GPGPU_REG_KERNEL_ADDR_HI 0x0304  /* 内核代码地址高 32 位 */
#define GPGPU_REG_KERNEL_ARGS_LO 0x0308  /* 内核参数地址低 32 位 */
#define GPGPU_REG_KERNEL_ARGS_HI 0x030C  /* 内核参数地址高 32 位 */
#define GPGPU_REG_GRID_DIM_X 0x0310      /* Grid X 维度 (Block 数量) */
#define GPGPU_REG_GRID_DIM_Y 0x0314      /* Grid Y 维度 */
#define GPGPU_REG_GRID_DIM_Z 0x0318      /* Grid Z 维度 */
#define GPGPU_REG_BLOCK_DIM_X 0x031C     /* Block X 维度 (线程数量) */
#define GPGPU_REG_BLOCK_DIM_Y 0x0320     /* Block Y 维度 */
#define GPGPU_REG_BLOCK_DIM_Z 0x0324     /* Block Z 维度 */
#define GPGPU_REG_SHARED_MEM_SIZE 0x0328 /* 每个 Block 的共享内存大小 */
#define GPGPU_REG_DISPATCH 0x0330        /* 写任意值启动内核执行 */

#define GPGPU_VENDOR_ID 0x1234
#define GPGPU_DEVICE_ID 0x1337

#define GPGPU_CTRL_BAR_SIZE (1 * 1024 * 1024)  /* BAR0: 控制寄存器 1MB */
#define GPGPU_VRAM_BAR_SIZE (64 * 1024 * 1024) /* BAR2: 显存 64MB (默认) */
#define GPGPU_DOORBELL_BAR_SIZE (64 * 1024)    /* BAR4: 门铃寄存器 64KB */

struct gpgpu_dev {
    struct pci_dev *pdev;
    void __iomem   *ctrl_base;     // BAR 0
    void __iomem   *vram_base;     // BAR 2
    void __iomem   *doorbell_base; // BAR 4
    dev_t           devno;
    struct cdev     cdev;
    struct class *class;
    wait_queue_head_t wait_queue;  // 等待队列
    int               kernel_done; // 完成标志
};

struct gpgpu_dispatch_args {
    uint64_t kernel_addr;
    uint64_t kernel_args;
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t shared_mem_size;
};

// PCI ID 匹配表 — 告诉内核这个驱动能处理哪些设备
static struct pci_device_id gpgpu_pci_ids[] = {
    {PCI_DEVICE(GPGPU_VENDOR_ID, GPGPU_DEVICE_ID)},
    {
        0,
    }};
MODULE_DEVICE_TABLE(pci, gpgpu_pci_ids);

static irqreturn_t irq_kernel_handler(int irq, void *data) {
    struct gpgpu_dev *gpu_dev = data;
    gpu_dev->kernel_done      = 1;
    wake_up_interruptible(&gpu_dev->wait_queue);
    return IRQ_HANDLED;
}

static int gpgpu_open(struct inode *node, struct file *file) {
    struct gpgpu_dev *gpu_dev =
        container_of(node->i_cdev, struct gpgpu_dev, cdev);
    file->private_data = gpu_dev;
    return 0;
}

static int gpgpu_realse(struct inode *node, struct file *file) { return 0; }

static int gpgpu_mmap(struct file *file, struct vm_area_struct *vma) {
    int ret;

    struct gpgpu_dev *gpgpu_dev = file->private_data;
    unsigned long     pfn;
    switch (vma->vm_pgoff) {
    case 0:
        pfn = pci_resource_start(gpgpu_dev->pdev, 0) >> PAGE_SHIFT;
        break;
    case 1:
        pfn = pci_resource_start(gpgpu_dev->pdev, 2) >> PAGE_SHIFT;
        break;
    default:
        return -EINVAL;
    }
    ret = io_remap_pfn_range(vma, vma->vm_start, pfn,
                             vma->vm_end - vma->vm_start, vma->vm_page_prot);
    if (ret)
        return ret;
    return 0;
}

static long gpgpu_ioctl(struct file *file, unsigned int cmd,
                        unsigned long arg) {
    struct gpgpu_dispatch_args args;
    struct gpgpu_dev          *gpgpu_dev = file->private_data;

    switch (cmd) {
    case GPGPU_IOC_DISPATCH:
        if (copy_from_user(&args, (void __user *)arg, sizeof(args)))
            return -EFAULT;
        gpgpu_dev->kernel_done = 0;
        iowrite32((uint32_t)(args.kernel_addr & 0xFFFFFFFF),
                  gpgpu_dev->ctrl_base + GPGPU_REG_KERNEL_ADDR_LO);
        iowrite32((uint32_t)((args.kernel_addr) >> 32 & 0xFFFFFFFF),
                  gpgpu_dev->ctrl_base + GPGPU_REG_KERNEL_ADDR_HI);
        iowrite32((uint32_t)(args.kernel_args & 0xFFFFFFFF),
                  gpgpu_dev->ctrl_base + GPGPU_REG_KERNEL_ARGS_LO);
        iowrite32((uint32_t)((args.kernel_args) >> 32 & 0xFFFFFFFF),
                  gpgpu_dev->ctrl_base + GPGPU_REG_KERNEL_ARGS_HI);

        iowrite32((uint32_t)((args.grid_dim[0])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_GRID_DIM_X);
        iowrite32((uint32_t)((args.grid_dim[1])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_GRID_DIM_Y);
        iowrite32((uint32_t)((args.grid_dim[2])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_GRID_DIM_Z);

        iowrite32((uint32_t)((args.block_dim[0])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_BLOCK_DIM_X);
        iowrite32((uint32_t)((args.block_dim[1])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_BLOCK_DIM_Y);
        iowrite32((uint32_t)((args.block_dim[2])),
                  gpgpu_dev->ctrl_base + GPGPU_REG_BLOCK_DIM_Z);

        iowrite32((uint32_t)((args.shared_mem_size)),
                  gpgpu_dev->ctrl_base + GPGPU_REG_SHARED_MEM_SIZE);

        iowrite32(1, gpgpu_dev->ctrl_base + GPGPU_REG_DISPATCH);

        wait_event_interruptible(gpgpu_dev->wait_queue, gpgpu_dev->kernel_done);

        break;
    default:
        return -EINVAL;
    }
    return 0;
}

static struct file_operations gpgpu_fops = {.owner          = THIS_MODULE,
                                            .open           = gpgpu_open,
                                            .release        = gpgpu_realse,
                                            .mmap           = gpgpu_mmap,
                                            .unlocked_ioctl = gpgpu_ioctl};

// probe — 内核发现匹配设备时调用
static int gpgpu_probe(struct pci_dev *pdev, const struct pci_device_id *id) {
    // 初始化资源
    int               ret = -ENOMEM;
    u32               DEV_ID;
    struct device    *dev;
    struct gpgpu_dev *gpu_dev = kmalloc(sizeof(struct gpgpu_dev), GFP_KERNEL);
    if (!gpu_dev) {
        printk("molloc memory failed\n");
        goto err_free;
    }
    gpu_dev->pdev = pdev;
    dev_info(&pdev->dev, "GPGPU device found! vendor=0x%04x device=0x%04x\n",
             pdev->vendor, pdev->device);

    /*初始化等待队列*/
    init_waitqueue_head(&gpu_dev->wait_queue);
    gpu_dev->kernel_done = 0;

    /*初始化GPU配置*/
    ret = pci_enable_device(pdev);
    if (ret) {
        printk("enable pci device failed\n");
        goto err_free;
    }
    ret = pci_request_regions(pdev, "gpgpu");
    if (ret) {
        printk("request regions failed\n");
        goto err_disable_device;
    }
    gpu_dev->ctrl_base = pci_iomap(pdev, 0, 0);
    if (!gpu_dev->ctrl_base) {
        printk("molloc memory failed\n");
        goto err_release_regions;
    }
    gpu_dev->vram_base = pci_iomap(pdev, 2, 0);
    if (!gpu_dev->vram_base) {
        printk("molloc memory failed\n");
        goto err_iounmap_ctrl;
    }
    gpu_dev->doorbell_base = pci_iomap(pdev, 4, 0);
    if (!gpu_dev->doorbell_base) {
        printk("molloc memory failed\n");
        goto err_iounmap_vram;
    }
    DEV_ID = ioread32(gpu_dev->ctrl_base + 0x00);
    dev_info(&pdev->dev, "GPGPU device id is 0x%04x", DEV_ID);
    pci_set_drvdata(pdev, gpu_dev);
    u64 vram_size =
        ((u64)ioread32(gpu_dev->ctrl_base + GPGPU_REG_VRAM_SIZE_HI) << 32) |
        ioread32(gpu_dev->ctrl_base + GPGPU_REG_VRAM_SIZE_LO);
    dev_info(&pdev->dev, "VRAM size: %llu MB\n", vram_size / 1024 / 1024);

    // 注册设备节点
    ret = alloc_chrdev_region(&gpu_dev->devno, 0, 1, "gpgpu");
    if (ret) {
        printk("alloc_chrdev_region failed\n");
        goto err_iounmap_doorbell;
    }
    cdev_init(&gpu_dev->cdev, &gpgpu_fops);
    ret = cdev_add(&gpu_dev->cdev, gpu_dev->devno, 1);
    if (ret) {
        printk("cdev_add failed\n");
        goto err_cdev_region;
    }
    gpu_dev->class = class_create("gpgpu");
    if (IS_ERR(gpu_dev->class)) {
        ret = PTR_ERR(gpu_dev->class);
        goto err_cdev_init;
    }
    dev = device_create(gpu_dev->class, NULL, gpu_dev->devno, NULL, "gpgpu");
    if (IS_ERR(dev)) {
        ret = PTR_ERR(dev);
        goto err_cdev_create;
    }

    /* 注册中断 */
    pci_set_master(pdev);
    ret = pci_alloc_irq_vectors(gpu_dev->pdev, 1, 1, PCI_IRQ_LEGACY);
    if (ret < 0) {
        dev_err(&pdev->dev, "pci_alloc_irq_vectors failed: %d\n", ret);
        goto err_device_create;
    }

    ret = request_irq(pci_irq_vector(gpu_dev->pdev, 0), irq_kernel_handler,
                      IRQF_SHARED, "gpgpu-kernel", gpu_dev);
    if (ret)
        goto err_pci_alloc_irq_vectors;
    iowrite32(GPGPU_IRQ_KERNEL_DONE | GPGPU_IRQ_DMA_DONE | GPGPU_IRQ_ERROR,
              gpu_dev->ctrl_base + GPGPU_REG_IRQ_ENABLE);
    return 0;

err_request_irq_kernel:
    free_irq(pci_irq_vector(pdev, 0), gpu_dev);
err_pci_alloc_irq_vectors:
    pci_free_irq_vectors(pdev);
err_device_create:
    device_destroy(gpu_dev->class, gpu_dev->devno);
err_cdev_create:
    class_destroy(gpu_dev->class);
err_cdev_init:
    cdev_del(&gpu_dev->cdev);
err_cdev_region:
    unregister_chrdev_region(gpu_dev->devno, 1);
err_iounmap_doorbell:
    pci_iounmap(pdev, gpu_dev->doorbell_base);
err_iounmap_vram:
    pci_iounmap(pdev, gpu_dev->vram_base);
err_iounmap_ctrl:
    pci_iounmap(pdev, gpu_dev->ctrl_base);
err_release_regions:
    pci_release_regions(pdev);
err_disable_device:
    pci_disable_device(pdev);
err_free:
    kfree(gpu_dev);
    return ret;
}

// remove — 设备移除或驱动卸载时调用
static void gpgpu_remove(struct pci_dev *pdev) {
    struct gpgpu_dev *gpu_dev = pci_get_drvdata(pdev);

    free_irq(pci_irq_vector(pdev, 0), gpu_dev);
    pci_free_irq_vectors(pdev);
    device_destroy(gpu_dev->class, gpu_dev->devno);
    class_destroy(gpu_dev->class);
    cdev_del(&gpu_dev->cdev);
    unregister_chrdev_region(gpu_dev->devno, 1);
    dev_info(&pdev->dev, "GPGPU device removed\n");

    pci_iounmap(pdev, gpu_dev->ctrl_base);
    pci_iounmap(pdev, gpu_dev->vram_base);
    pci_iounmap(pdev, gpu_dev->doorbell_base);

    pci_release_regions(pdev);

    pci_disable_device(pdev);

    kfree(gpu_dev);
}

static struct pci_driver gpgpu_driver = {
    .name     = "gpgpu",
    .id_table = gpgpu_pci_ids,
    .probe    = gpgpu_probe,
    .remove   = gpgpu_remove,
};

module_pci_driver(gpgpu_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Eclipse");
MODULE_DESCRIPTION("GPGPU PCI Driver");