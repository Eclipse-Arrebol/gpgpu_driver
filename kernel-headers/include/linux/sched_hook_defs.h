/* SPDX-License-Identifier: GPL-2.0 */
BPF_SCHED_HOOK(int, -1, cfs_select_rq, struct sched_migrate_ctx *ctx)
BPF_SCHED_HOOK(int, -1, cfs_can_migrate_task, struct task_struct *p,
			struct sched_migrate_node *migrate_node)
BPF_SCHED_HOOK(int, -1, cfs_tag_entity_eligible, struct sched_entity *se)
BPF_SCHED_HOOK(int, -1, cfs_tag_pick_next_entity,
			const struct sched_entity *curr,
			const struct sched_entity *next)
