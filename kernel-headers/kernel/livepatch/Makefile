# SPDX-License-Identifier: GPL-2.0-only
obj-$(CONFIG_LIVEPATCH) += livepatch.o

ifeq ($(CONFIG_LIVEPATCH_FTRACE), y)
livepatch-objs := core.o patch.o shadow.o state.o transition.o
else
livepatch-objs := core.o
endif
