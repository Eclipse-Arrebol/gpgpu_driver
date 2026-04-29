# SPDX-License-Identifier: GPL-2.0
# Copyright (C) 2021 - 2023, Shanghai Yunsilicon Technology Co., Ltd.
# All rights reserved.
# Makefile for the Yunsilicon device drivers.
#

ifeq ($(HAVE_TO_USE_M_IB_CORE), 1)
	subdir-ccflags-y += $(xsc-ccflags)
endif

obj-$(CONFIG_YUNSILICON_XSC_ETH) += xsc/net/
obj-$(CONFIG_YUNSILICON_XSC_PCI) += xsc/pci/
