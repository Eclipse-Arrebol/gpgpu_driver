/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2024
 * All rights reserved.
 *
 * CAQM An implementation of 802.1 CAQM tagging.
 * Authors:
 *	Chengjun Jia <jiachengjun2@huawei.com>
 *	Shurui Ding <dongshurui@huawei.com>
 */
#ifndef _LINUX_IF_CAQM_H_
#define _LINUX_IF_CAQM_H_

#include <linux/types.h>
#include <linux/jump_label.h>
#include <linux/skbuff.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/rtnetlink.h>
#include <linux/bug.h>
#include <linux/if_vlan.h>
#include <linux/neighbour.h>

#ifdef CONFIG_ETH_CAQM

#define CAQM_HLEN	(4)
#define CAQM_MAX_DEPTH	(2)
#define CAQM_RECV_EN	(true)
#define CAQM_SEND_EN	(true)

#define FIXED_POINT_8 (8U)
#define FIXED_POINT_20 (20U)
#define FIXED_POINT_8_UNIT (1<<8U)
#define FIXED_POINT_20_UNIT (1<<20U)

extern int sysctl_caqm_cc_type;
extern int sysctl_caqm_debug_info;
extern int sysctl_caqm_alpha_fx_8;
extern int sysctl_caqm_beta;
extern unsigned int sysctl_caqm_min_cwnd;
extern int sysctl_caqm_mtu_unit;
extern int sysctl_caqm_data_hint_unit;
extern unsigned int sysctl_caqm_ack_hint_unit;
extern struct static_key_false sysctl_caqm_enable;
extern u8 sysctl_caqm_en_data;
extern u64 sysctl_caqm_filter_nics;
extern u32 sysctl_caqm_rtt_standard;

/**
 *	struct caqm_hdr_info - caqm ethernet header congestion control information
 *	@cc_type: 3'b000 = CAQM
 *	@is_last_hop: Location, indicate whether the congestion is the last hop of network
 *	@padding: Bit[11]: useless
 *	@caqm_en: Enable caqm, 0: Disable; 1: Enable
 *	@c_bit: Congestion status, 0: None-congestion; 1: Congestion
 *	@i_bit: Hint valid status, 0: Ignore the value of Hint field, see it as 0; 1: Hint is valid
 *	@hint: carries the CAQM Hint value
 */
struct caqm_hdr_info {
#if defined(__BIG_ENDIAN_BITFIELD)
	__u8 cc_type:3,
		is_last_hop:1,
		padding:1,
		caqm_en:1,
		c_bit:1,
		i_bit:1;
#elif defined(__LITTLE_ENDIAN_BITFIELD)
	__u8 i_bit:1,
		c_bit:1,
		caqm_en:1,
		padding:1,
		is_last_hop:1,
		cc_type:3;
#else
#error  "Please fix <asm/byteorder.h>"
#endif
	__u8 hint;
} __packed;

static inline int get_caqm_real_hint_bytes(struct caqm_hdr_info *cinfo)
{
	if (cinfo->cc_type == sysctl_caqm_cc_type) {
		if (cinfo->c_bit == cinfo->i_bit) {
			return 0;
		} else if (cinfo->caqm_en) {
			if (cinfo->c_bit)
				return 0 - sysctl_caqm_beta;
			else
				return cinfo->hint * sysctl_caqm_data_hint_unit;
		} else if (cinfo->c_bit) {
			return (0 - (cinfo->hint)) * sysctl_caqm_ack_hint_unit;
		} else {
			return cinfo->hint * sysctl_caqm_ack_hint_unit;
		}
	}
	return 0;
}

/*
 *	struct caqm_hdr - caqm header
 *	@h_caqm_info: Congestion Control Information
 *	@h_caqm_encapsulated_proto: packet type ID or len
 */
struct caqm_hdr {
		__be16 h_caqm_info;
		__be16 h_caqm_encapsulated_proto;
};

#define TCP_CONG_NEEDS_CAQM 0x4
// #define TCP_CONG_MASK	(TCP_CONG_NON_RESTRICTED | TCP_CONG_NEEDS_ECN | TCP_CONG_NEEDS_CAQM)

/**
 * eth_type_caqm - check for valid caqm ether type.
 * @ethertype: ether type to check
 *
 * Returns true if the ether type is a caqm ether type.
 */
static inline bool eth_type_caqm(__be16 ethertype)
{
	return ethertype == htons(CONFIG_ETH_P_CAQM);
}

#define CAQM_PKT_ACK (0)
#define CAQM_PKT_DATA (1)

/**
 * skb_caqm_info - caqm info in skbuff
 * @send_en: true if need send caqm hdr
 * @recv_en: true if be recived caqm hdr
 * @pkt_type: CAQM_PKT_ACK or CAQM_PTK_DATA
 * @send_hdr: the caqm hdr will send out
 * @recv_hint: the hint value in recive packet */
struct skb_caqm_info {
	__u16	send_en:1,
			recv_en:1,
			pkt_type:1;
	__u16	send_hdr;
	__s32	recv_hint; // unit is Byte
};

static inline struct skb_caqm_info *get_skb_caqm_info(struct sk_buff *skb)
{
	return (struct skb_caqm_info *)&(skb->caqm_info);
}

static inline void set_skb_caqm_info_send_en(struct sk_buff *skb, bool send_en)
{
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	cinfo->send_en = send_en;
}

static inline void set_skb_caqm_info_recv_en(struct sk_buff *skb, bool recv_en)
{
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	cinfo->recv_en = recv_en;
}

static inline void set_skb_caqm_info_pkt_type(struct sk_buff *skb, bool pkt_type)
{
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	cinfo->pkt_type = pkt_type;
}

static inline void set_skb_caqm_info_send_hdr(struct sk_buff *skb, u16 send_hdr)
{
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	cinfo->send_hdr = send_hdr;
}

static inline void set_skb_caqm_info_recv_hint(struct sk_buff *skb, __s32 recv_hint)
{
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	cinfo->recv_hint = recv_hint;
}

static inline void caqm_set_encap_proto(struct sk_buff *skb,
					struct caqm_hdr *chdr)
{
	__be16 proto;
	unsigned short *rawp;

	proto = chdr->h_caqm_encapsulated_proto;
	if (eth_proto_is_802_3(proto)) {
		skb->protocol = proto;
		return;
	}

	rawp = (unsigned short *)(chdr + 1);
	if (*rawp == 0xFFFF)
		/*
		 * This is a magic hack to spot IPX packets. Older Novell
		 * breaks the protocol design and runs IPX over 802.3 without
		 * an 802.2 LLC layer. We look for FFFF which isn't a used
		 * 802.2 SSAP/DSAP. This won't work for fault tolerant netware
		 * but does for the rest.
		 */
		skb->protocol = htons(ETH_P_802_3);
	else
		/*
		 * Real 802.2 LLC
		 */
		skb->protocol = htons(ETH_P_802_2);
}

static inline void __caqm_put_tag(struct sk_buff *skb, u16 hdr_info)
{
	struct caqm_hdr_info *ptr = (struct caqm_hdr_info *)&hdr_info;
	int hint = get_caqm_real_hint_bytes(ptr);

	skb->caqm_info = 0; // clear caqm_info
	set_skb_caqm_info_recv_en(skb, 1);

	if (ptr->caqm_en)
		set_skb_caqm_info_pkt_type(skb, CAQM_PKT_DATA);
	else
		set_skb_caqm_info_pkt_type(skb, CAQM_PKT_ACK);

	set_skb_caqm_info_recv_hint(skb, hint);
}

static struct sk_buff *skb_reorder_caqm_header(struct sk_buff *skb)
{
	int mac_len, meta_len;
	void *meta;

	if (skb_cow(skb, skb_headroom(skb)) < 0) {
		kfree_skb(skb);
		return NULL;
	}

	mac_len = skb->data - skb_mac_header(skb);
	if (likely(mac_len > CAQM_HLEN + ETH_TLEN)) {
		memmove(skb_mac_header(skb) + CAQM_HLEN, skb_mac_header(skb),
			mac_len - CAQM_HLEN - ETH_TLEN);
	}

	meta_len = skb_metadata_len(skb);
	if (meta_len) {
		meta = skb_metadata_end(skb) - meta_len;
		memmove(meta + CAQM_HLEN, meta, meta_len);
	}

	skb->mac_header += CAQM_HLEN;
	return skb;
}

/**
 * __caqm_get_protocol_after_vlan - get protocol EtherType.
 * @skb: skbuff to query
 * @type: first vlan protocol
 * @depth: buffer to store length of eth and vlan tags in bytes
 *
 * Returns the EtherType of the packet, regardless of whether it is
 * vlan encapsulated (normal or hardware accelerated) or not.
 */
static inline __be16 __caqm_get_protocol_after_vlan(const struct sk_buff *skb, __be16 type,
					 int *depth)
{
	int caqm_depth = 0, parse_depth = CAQM_MAX_DEPTH;

	if (eth_type_caqm(type)) {
		caqm_depth -= CAQM_HLEN;
		do {
			struct caqm_hdr chdr, *ch;

			ch = skb_header_pointer(skb, caqm_depth, sizeof(chdr), &chdr);
			if (unlikely(!ch || !--parse_depth))
				return 0;

			type = ch->h_caqm_encapsulated_proto;
			caqm_depth += CAQM_HLEN;
		} while (eth_type_caqm(type));
	}

	if (depth)
		*depth = caqm_depth;

	return type;
}

static inline __be16 caqm_get_protocol_and_depth_after_vlan(struct sk_buff *skb,
						 __be16 type, int *depth)
{
	int maclen;
	int caqmlen;

	type = __caqm_get_protocol_after_vlan(skb, type, &caqmlen);
	if (depth)
		maclen = *depth + caqmlen;
	else
		return type;

	if (type) {
		if (!pskb_may_pull(skb, maclen))
			type = 0;
		else if (depth)
			*depth = maclen;
	}
	return type;
}
#endif

static inline struct sk_buff *skb_caqm_untag(struct sk_buff *skb)
{
#ifdef CONFIG_ETH_CAQM
	struct caqm_hdr *chdr;
	u16 caqm_hdr_info;

	if (!static_branch_unlikely(&sysctl_caqm_enable))
		return skb;

	skb = skb_share_check(skb, GFP_ATOMIC);
	if (unlikely(!skb))
		goto err_free;
	/* We may access the two bytes after caqm_hdr in caqm_set_encap_proto(). */
	if (unlikely(!pskb_may_pull(skb, CAQM_HLEN + sizeof(unsigned short))))
		goto err_free;

	chdr = (struct caqm_hdr *)skb->data;
	caqm_hdr_info = chdr->h_caqm_info;
	__caqm_put_tag(skb, caqm_hdr_info);

	skb_pull_rcsum(skb, CAQM_HLEN);
	caqm_set_encap_proto(skb, chdr);

	skb = skb_reorder_caqm_header(skb);
	if (unlikely(!skb))
		goto err_free;

	skb_reset_network_header(skb);
	if (!skb_transport_header_was_set(skb))
		skb_reset_transport_header(skb);
	skb_reset_mac_len(skb);

	return skb;

err_free:
	kfree_skb(skb);
	return NULL;
#else
	return skb;
#endif
}

static inline void skb_gro_caqm_untag(struct sk_buff *skb)
{
#ifdef CONFIG_ETH_CAQM
	struct caqm_hdr *chdr;
	struct vlan_hdr *vlanhdr;
	u16 caqm_hdr_info;
	__be16 proto = skb->protocol;
	u8 skip_time = 0;
	int mac_len, meta_len;
	void *meta;
	struct caqm_hdr tmp_caqm_data;

	if (!static_branch_unlikely(&sysctl_caqm_enable) || skb_shared(skb))
		return;

	/* We may access the two bytes after caqm_hdr in caqm_set_encap_proto(). */
	if (unlikely(!pskb_may_pull(skb, VLAN_HLEN * 2 + CAQM_HLEN + sizeof(unsigned short))))
		return;

	vlanhdr = (struct vlan_hdr *)(skb->data - VLAN_HLEN);
	// Skip at most 2 vlan hdr
	while (skip_time < 2 && eth_type_vlan(proto)) {
		vlanhdr++; // chdr move a VLAN len
		skip_time += 1;
		proto = vlanhdr->h_vlan_encapsulated_proto;
	}
	// Look for vlan header, 2 times
	if (!eth_type_caqm(proto) || skb_cow(skb, skb_headroom(skb)) < 0)
		return;

	chdr = (struct caqm_hdr *)(((void *)vlanhdr) + VLAN_HLEN);
	caqm_hdr_info = chdr->h_caqm_info;
	__caqm_put_tag(skb, caqm_hdr_info);

	if ((unsigned char *) chdr != skb->data) {
		// adjust the caqm header before vlan header
		// 1. Set the vlanhdr->proto as the next
		vlanhdr->h_vlan_encapsulated_proto = chdr->h_caqm_encapsulated_proto;
		// 2. Exchange caqm and vlan hdr
		tmp_caqm_data = *chdr;
		memmove(skb->data + CAQM_HLEN, skb->data, skip_time * VLAN_HLEN);
		*(struct caqm_hdr *)(skb->data) = tmp_caqm_data;
	} else
		caqm_set_encap_proto(skb, chdr);

	skb_pull_rcsum(skb, CAQM_HLEN);

	mac_len = skb->data - skb_mac_header(skb);
	if (likely(mac_len > CAQM_HLEN + ETH_TLEN)) {
		memmove(skb_mac_header(skb) + CAQM_HLEN, skb_mac_header(skb),
			mac_len - CAQM_HLEN - ETH_TLEN);
	}

	meta_len = skb_metadata_len(skb);
	if (meta_len) {
		meta = skb_metadata_end(skb) - meta_len;
		memmove(meta + CAQM_HLEN, meta, meta_len);
	}

	skb->mac_header += CAQM_HLEN;

	skb_reset_network_header(skb);
	if (!skb_transport_header_was_set(skb))
		skb_reset_transport_header(skb);
	skb_reset_mac_len(skb);

#endif
}

#ifdef CONFIG_ETH_CAQM
static inline __be16 caqm_get_protocol_and_depth(struct sk_buff *skb,
						 __be16 type, int *depth)
{
	if (static_branch_unlikely(&sysctl_caqm_enable) && eth_type_caqm(type))
		return caqm_get_protocol_and_depth_after_vlan(skb, type, depth);
	else
		return type;
}
#endif

static inline void caqm_update_hint_in_gro(struct sk_buff *skb, struct sk_buff *p)
{
#ifdef CONFIG_ETH_CAQM
	struct skb_caqm_info *cinfo_p = get_skb_caqm_info(p);
	struct skb_caqm_info *cinfo_skb = get_skb_caqm_info(skb);

	if (static_branch_unlikely(&sysctl_caqm_enable) && cinfo_p->recv_en && cinfo_skb->recv_en) {
		cinfo_p->recv_hint += cinfo_skb->recv_hint;
		cinfo_skb->recv_en = 0;
	}
#endif
}

static inline void caqm_add_eth_header(struct sk_buff *skb, unsigned short *type,
			       struct net_device *dev)
{
#ifdef CONFIG_ETH_CAQM
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	if (!static_branch_unlikely(&sysctl_caqm_enable))
		return;

	if ((sysctl_caqm_filter_nics & (1UL << dev->ifindex)) == 0)
		cinfo->send_en = 0;
	if (cinfo->send_en) {
		cinfo->send_en = 0;
		if (unlikely(skb_headroom(skb) <  ETH_HLEN + CAQM_HLEN))
			return; // No enough room
		u16 *chdr = skb_push(skb, CAQM_HLEN);

		chdr[0] = (cinfo->send_hdr);
		chdr[1] = htons(*type);
		*type = CONFIG_ETH_P_CAQM;
	}
#endif
}

static inline bool is_caqm_out_enable(struct sk_buff *skb,
				   struct net_device *dev)

{
#ifdef CONFIG_ETH_CAQM
	struct skb_caqm_info *cinfo = get_skb_caqm_info(skb);

	if (!static_branch_unlikely(&sysctl_caqm_enable))
		return false;

	// If the nic is not configed, the output packet has no caqm header
	if ((sysctl_caqm_filter_nics & (1UL << dev->ifindex)) == 0)
		cinfo->send_en = 0;
	if (cinfo->send_en)
		return true;
#endif
	return false;
}

#endif /* _LINUX_IF_CAQM_H_ */
