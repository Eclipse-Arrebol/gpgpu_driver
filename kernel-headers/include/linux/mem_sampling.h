/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * mem_sampling.h: declare the mem_sampling abstract layer and provide
 * unified pmu sampling for NUMA, DAMON, etc.
 *
 * Sample records are converted to mem_sampling_record, and then
 * mem_sampling_record_captured_cb_type invoke the callbacks to
 * pass the record.
 *
 * Copyright (c) 2024-2025, Huawei Technologies Ltd.
 */
#ifndef __MEM_SAMPLING_H
#define __MEM_SAMPLING_H

#include <linux/kfifo.h>

enum mem_sampling_sample_type {
	MEM_SAMPLING_L1D_ACCESS		= 1 << 0,
	MEM_SAMPLING_L1D_MISS		= 1 << 1,
	MEM_SAMPLING_LLC_ACCESS		= 1 << 2,
	MEM_SAMPLING_LLC_MISS		= 1 << 3,
	MEM_SAMPLING_TLB_ACCESS		= 1 << 4,
	MEM_SAMPLING_TLB_MISS		= 1 << 5,
	MEM_SAMPLING_BRANCH_MISS	= 1 << 6,
	MEM_SAMPLING_REMOTE_ACCESS	= 1 << 7,
};

enum mem_sampling_op_type {
	MEM_SAMPLING_LD	= 1 << 0,
	MEM_SAMPLING_ST	= 1 << 1,
};

enum arm_spe_user_e {
	SPE_USER_PERF,
	SPE_USER_MEM_SAMPLING,
};
DECLARE_PER_CPU(enum arm_spe_user_e, arm_spe_user);

struct mem_sampling_record {
	enum mem_sampling_sample_type	type;
	int				err;
	u32				op;
	u32				latency;
	u64				from_ip;
	u64				to_ip;
	u64				timestamp;
	u64				virt_addr;
	u64				phys_addr;
	u64				context_id;
	u64				boost_spe_addr[8];
	u64				rem_addr;
	u16				source;
};

struct mem_sampling_ops_struct {
	int (*sampling_start)(void);
	void (*sampling_stop)(void);
	void (*sampling_continue)(void);
	void (*sampling_decoding)(void);
	struct mm_spe_buf* (*mm_spe_getbuf_addr)(void);
	int (*mm_spe_getnum_record)(void);

};
extern struct mem_sampling_ops_struct mem_sampling_ops;

enum mem_sampling_type_enum {
	MEM_SAMPLING_ARM_SPE,
	MEM_SAMPLING_UNSUPPORTED
};

enum user_switch_type {
	USER_SWITCH_AWAY_FROM_MEM_SAMPLING,
	USER_SWITCH_BACK_TO_MEM_SAMPLING,
	USER_SWITCH_TYPE_MAX,
};

DECLARE_STATIC_KEY_FALSE(sched_numabalancing_mem_sampling);
extern struct static_key_false mem_sampling_access_hints;

#ifdef CONFIG_ARM_SPE_MEM_SAMPLING
int mm_spe_start(void);
void mm_spe_stop(void);
void mm_spe_continue(void);
void mm_spe_decoding(void);
int mm_spe_getnum_record(void);
struct mm_spe_buf *mm_spe_getbuf_addr(void);
int mm_spe_enabled(void);
void arm_spe_set_probe_status(int status);
#else
static inline void mm_spe_stop(void) { }
static inline void mm_spe_continue(void) { }
static inline void mm_spe_decoding(void) { }
static inline void arm_spe_set_probe_status(int status) { }
static inline int mm_spe_start(void) { return 0; }
static inline int mm_spe_getnum_record(void) { return 0; }
static inline struct mm_spe_buf *mm_spe_getbuf_addr(void) { return NULL; }
static inline int mm_spe_enabled(void) { return 0; }
#endif /* CONFIG_ARM_SPE_MEM_SAMPLING */

#if IS_ENABLED(CONFIG_MEM_SAMPLING)
void mem_sampling_process(void);
void arm_spe_set_user(enum arm_spe_user_e user);
void set_mem_sampling_state(bool enabled);
void mem_sampling_user_switch_process(enum user_switch_type type);
static inline bool spe_user_is_perf(void)
{
	return __this_cpu_read(arm_spe_user) == SPE_USER_PERF;
}
static inline bool spe_user_is_mem_sampling(void)
{
	return __this_cpu_read(arm_spe_user) == SPE_USER_MEM_SAMPLING;
}
void mem_sampling_sched_in(struct task_struct *prev, struct task_struct *curr);
#else
static inline void set_mem_sampling_state(bool enabled) { }
static inline void mem_sampling_sched_in(struct task_struct *prev, struct task_struct *curr) { }
#endif /* CONFIG_MEM_SAMPLING */

#ifdef CONFIG_DAMON_MEM_SAMPLING
#define	DAMOS_FIFO_MAX_RECORD		(1024)
struct damon_mem_sampling_record {
	u64 vaddr;
};

struct damon_mem_sampling_fifo {
	struct kfifo rx_kfifo;
	spinlock_t rx_kfifo_lock; /* protect SPE Rx data kfifo */
};

bool damon_use_mem_sampling(void);
#else
static inline bool damon_use_mem_sampling(void) { return false; }
#endif /* CONFIG_DAMON_MEM_SAMPLING */
#endif	/* __MEM_SAMPLING_H */
