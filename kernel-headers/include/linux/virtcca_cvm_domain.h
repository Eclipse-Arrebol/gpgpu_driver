/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 */

#ifndef __VIRTCCA_CVM_DOMAIN_H
#define __VIRTCCA_CVM_DOMAIN_H

#ifdef CONFIG_HISI_VIRTCCA_GUEST

#include <asm/virtcca_cvm_guest.h>
static inline bool virtcca_cvm_domain(void)
{
	return is_virtcca_cvm_world();
}

extern void enable_swiotlb_for_cvm_dev(struct device *dev, bool enable);

#else
static inline bool virtcca_cvm_domain(void)
{
	return false;
}

static inline void enable_swiotlb_for_cvm_dev(struct device *dev, bool enable) {}

#endif

#ifdef CONFIG_HISI_VIRTCCA_HOST

bool is_virtcca_cvm_enable(void);
u64 virtcca_get_tmi_version(void);

#else

static inline bool is_virtcca_cvm_enable(void)
{
	return 0;
}

static inline u64 virtcca_get_tmi_version(void)
{
	return 0;
}
#endif

#ifdef CONFIG_HISI_VIRTCCA_CODA
size_t virtcca_pci_get_rom_size(void  *pdev, void __iomem *rom,
			       size_t size);
#else
static inline size_t virtcca_pci_get_rom_size(void  *pdev, void __iomem *rom,
			       size_t size)
{
	return 0;
}

#endif

#endif /* __VIRTCCA_CVM_DOMAIN_H */
