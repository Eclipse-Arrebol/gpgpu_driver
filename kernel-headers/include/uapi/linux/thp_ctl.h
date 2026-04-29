/* SPDX-License-Identifier: GPL-2.0+ WITH Linux-syscall-note */
/* Copyright (C) 2024 Huawei Technologies Co., Ltd. */

#ifndef _THP_CTL_H
#define _THP_CTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

struct get_thp_status_arg {
	__kernel_pid_t pid;
	unsigned long	thp_enable;
};

#define IOC_THP_STATUS_GET	_IOWR('M', 0, struct get_thp_status_arg)
#define IOC_THP_SET_DISABLE	_IOW('M', 1, __kernel_pid_t)
#define IOC_THP_SET_ENABLE	_IOW('M', 2, __kernel_pid_t)

#endif
