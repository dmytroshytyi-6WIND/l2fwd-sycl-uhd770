/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2021 Marvell
 */

#ifndef TEST_CRYPTODEV_SECURITY_IPSEC_TEST_VECTORS_H_
#define TEST_CRYPTODEV_SECURITY_IPSEC_TEST_VECTORS_H_

#include <rte_crypto.h>
#include <rte_security.h>

#include "test_cryptodev_security_ipsec.h"

/*
 * Known vectors
 *
 * AES-GCM vectors are based on :
 * https://datatracker.ietf.org/doc/html/draft-mcgrew-gcm-test-01
 *
 * Vectors are updated to have corrected L4 checksum and sequence number 1.
 */

struct ipsec_test_data pkt_aes_128_gcm = {
    .key =
        {
            .data = {0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c, 0x6d, 0x6a,
                     0x8f, 0x94, 0x67, 0x30, 0x83, 0x08},
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x3e,
                    0x69,
                    0x8f,
                    0x00,
                    0x00,
                    0x80,
                    0x11,
                    0x4d,
                    0xcc,
                    0xc0,
                    0xa8,
                    0x01,
                    0x02,
                    0xc0,
                    0xa8,
                    0x01,
                    0x01,

                    /* UDP */
                    0x0a,
                    0x98,
                    0x00,
                    0x35,
                    0x00,
                    0x2a,
                    0x23,
                    0x43,
                    0xb2,
                    0xd0,
                    0x01,
                    0x00,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x03,
                    0x73,
                    0x69,
                    0x70,
                    0x09,
                    0x63,
                    0x79,
                    0x62,
                    0x65,
                    0x72,
                    0x63,
                    0x69,
                    0x74,
                    0x79,
                    0x02,
                    0x64,
                    0x6b,
                    0x00,
                    0x00,
                    0x01,
                    0x00,
                    0x01,
                },
            .len = 62,
        },
    .output_text =
        {
            .data =
                {
                    /* IP - outer header */
                    0x45,
                    0x00,
                    0x00,
                    0x74,
                    0x69,
                    0x8f,
                    0x00,
                    0x00,
                    0x80,
                    0x32,
                    0x4d,
                    0x75,
                    0xc0,
                    0xa8,
                    0x01,
                    0x02,
                    0xc0,
                    0xa8,
                    0x01,
                    0x01,

                    /* ESP */
                    0x00,
                    0x00,
                    0xa5,
                    0xf8,
                    0x00,
                    0x00,
                    0x00,
                    0x01,

                    /* IV */
                    0xfa,
                    0xce,
                    0xdb,
                    0xad,
                    0xde,
                    0xca,
                    0xf8,
                    0x88,

                    /* Data */
                    0xde,
                    0xb2,
                    0x2c,
                    0xd9,
                    0xb0,
                    0x7c,
                    0x72,
                    0xc1,
                    0x6e,
                    0x3a,
                    0x65,
                    0xbe,
                    0xeb,
                    0x8d,
                    0xf3,
                    0x04,
                    0xa5,
                    0xa5,
                    0x89,
                    0x7d,
                    0x33,
                    0xae,
                    0x53,
                    0x0f,
                    0x1b,
                    0xa7,
                    0x6d,
                    0x5d,
                    0x11,
                    0x4d,
                    0x2a,
                    0x5c,
                    0x3d,
                    0xe8,
                    0x18,
                    0x27,
                    0xc1,
                    0x0e,
                    0x9a,
                    0x4f,
                    0x51,
                    0x33,
                    0x0d,
                    0x0e,
                    0xec,
                    0x41,
                    0x66,
                    0x42,
                    0xcf,
                    0xbb,
                    0x85,
                    0xa5,
                    0xb4,
                    0x7e,
                    0x48,
                    0xa4,
                    0xec,
                    0x3b,
                    0x9b,
                    0xa9,
                    0x5d,
                    0x91,
                    0x8b,
                    0xd4,
                    0x29,
                    0xc7,
                    0x37,
                    0x57,
                    0x9f,
                    0xf1,
                    0x9e,
                    0x58,
                    0xcf,
                    0xfc,
                    0x60,
                    0x7a,
                    0x3b,
                    0xce,
                    0x89,
                    0x94,

                },
            .len = 116,
        },
    .salt =
        {
            .data = {0xca, 0xfe, 0xba, 0xbe},
            .len = 4,
        },

    .iv =
        {
            .data = {0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88},
        },

    .ipsec_xform =
        {
            .spi = 0xa5f8,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .options.ip_csum_enable = 0,
            .options.l4_csum_enable = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
            .esn.low = 1,
        },

    .aead = true,

    .xform =
        {
            .aead =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AEAD,
                    .aead =
                        {
                            .op = RTE_CRYPTO_AEAD_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_AEAD_AES_GCM,
                            .key.length = 16,
                            .iv.length = 12,
                            .iv.offset = IV_OFFSET,
                            .digest_length = 16,
                            .aad_length = 12,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_192_gcm = {
	.key = {
		.data = {
			0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c,
			0x6d, 0x6a, 0x8f, 0x94, 0x67, 0x30, 0x83, 0x08,
			0xfe, 0xff, 0xe9, 0x92, 0x86, 0x65, 0x73, 0x1c
		},
	},
	.input_text = {
		.data = {
			/* IP */
			0x45, 0x00, 0x00, 0x28, 0xa4, 0xad, 0x40, 0x00,
			0x40, 0x06, 0x78, 0x80, 0x0a, 0x01, 0x03, 0x8f,
			0x0a, 0x01, 0x06, 0x12,

			/* TCP */
			0x80, 0x23, 0x06, 0xb8, 0xcb, 0x71, 0x26, 0x02,
			0xdd, 0x6b, 0xb0, 0x3e, 0x50, 0x10, 0x16, 0xd0,
			0x75, 0x67, 0x00, 0x01
		},
		.len = 40,
	},
	.output_text = {
		.data = {
			/* IP - outer header */
			0x45, 0x00, 0x00, 0x60, 0x69, 0x8f, 0x00, 0x00,
			0x80, 0x32, 0x4d, 0x89, 0xc0, 0xa8, 0x01, 0x02,
			0xc0, 0xa8, 0x01, 0x01,

			/* ESP */
			0x00, 0x00, 0xa5, 0xf8, 0x00, 0x00, 0x00, 0x01,

			/* IV */
			0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88,

			/* Data */
			0xa5, 0xb1, 0xf8, 0x06, 0x60, 0x29, 0xae, 0xa4,
			0x0e, 0x59, 0x8b, 0x81, 0x22, 0xde, 0x02, 0x42,
			0x09, 0x38, 0xb3, 0xab, 0x33, 0xf8, 0x28, 0xe6,
			0x87, 0xb8, 0x85, 0x8b, 0x5b, 0xfb, 0xdb, 0xd0,
			0x31, 0x5b, 0x27, 0x45, 0x21, 0x4b, 0xcc, 0x77,
			0x82, 0xac, 0x91, 0x38, 0xf2, 0xbb, 0xbe, 0xe4,
			0xcf, 0x03, 0x36, 0x89, 0xdd, 0x40, 0xd3, 0x6e,
			0x54, 0x05, 0x22, 0x22,
		},
		.len = 96,
	},
	.salt = {
		.data = {
			0xca, 0xfe, 0xba, 0xbe
		},
		.len = 4,
	},

	.iv = {
		.data = {
			0xfa, 0xce, 0xdb, 0xad, 0xde, 0xca, 0xf8, 0x88
		},
	},

	.ipsec_xform = {
		.spi = 0xa5f8,
		.options.esn = 0,
		.options.udp_encap = 0,
		.options.copy_dscp = 0,
		.options.copy_flabel = 0,
		.options.copy_df = 0,
		.options.dec_ttl = 0,
		.options.ecn = 0,
		.options.stats = 0,
		.options.tunnel_hdr_verify = 0,
		.options.ip_csum_enable = 0,
		.options.l4_csum_enable = 0,
		.direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
		.proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
		.mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
		.tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
		.replay_win_sz = 0,
	},

	.aead = true,

	.xform = {
		.aead = {
			.next = NULL,
			.type = RTE_CRYPTO_SYM_XFORM_AEAD,
			.aead = {
				.op = RTE_CRYPTO_AEAD_OP_ENCRYPT,
				.algo = RTE_CRYPTO_AEAD_AES_GCM,
				.key.length = 24,
				.iv.length = 12,
				.iv.offset = IV_OFFSET,
				.digest_length = 16,
				.aad_length = 12,
			},
		},
	},
};

struct ipsec_test_data pkt_aes_256_gcm = {
    .key =
        {
            .data =
                {
                    0xab, 0xbc, 0xcd, 0xde, 0xf0, 0x01, 0x12, 0x23,
                    0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x9a, 0xab,
                    0xab, 0xbc, 0xcd, 0xde, 0xf0, 0x01, 0x12, 0x23,
                    0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x9a, 0xab,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x30,
                    0x69,
                    0xa6,
                    0x40,
                    0x00,
                    0x80,
                    0x06,
                    0x26,
                    0x90,
                    0xc0,
                    0xa8,
                    0x01,
                    0x02,
                    0x93,
                    0x89,
                    0x15,
                    0x5e,

                    /* TCP */
                    0x0a,
                    0x9e,
                    0x00,
                    0x8b,
                    0x2d,
                    0xc5,
                    0x7e,
                    0xe0,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x70,
                    0x02,
                    0x40,
                    0x00,
                    0x20,
                    0xbf,
                    0x00,
                    0x00,
                    0x02,
                    0x04,
                    0x05,
                    0xb4,
                    0x01,
                    0x01,
                    0x04,
                    0x02,
                },
            .len = 48,
        },
    .output_text =
        {
            .data =
                {
                    /* IP - outer header */
                    0x45,
                    0x00,
                    0x00,
                    0x68,
                    0x69,
                    0x8f,
                    0x00,
                    0x00,
                    0x80,
                    0x32,
                    0x4d,
                    0x81,
                    0xc0,
                    0xa8,
                    0x01,
                    0x02,
                    0xc0,
                    0xa8,
                    0x01,
                    0x01,

                    /* ESP */
                    0x4a,
                    0x2c,
                    0xbf,
                    0xe3,
                    0x00,
                    0x00,
                    0x00,
                    0x01,

                    /* IV */
                    0x01,
                    0x02,
                    0x03,
                    0x04,
                    0x05,
                    0x06,
                    0x07,
                    0x08,

                    /* Data */
                    0xff,
                    0x42,
                    0x5c,
                    0x9b,
                    0x72,
                    0x45,
                    0x99,
                    0xdf,
                    0x7a,
                    0x3b,
                    0xcd,
                    0x51,
                    0x01,
                    0x94,
                    0xe0,
                    0x0d,
                    0x6a,
                    0x78,
                    0x10,
                    0x7f,
                    0x1b,
                    0x0b,
                    0x1c,
                    0xbf,
                    0x06,
                    0xef,
                    0xae,
                    0x9d,
                    0x65,
                    0xa5,
                    0xd7,
                    0x63,
                    0x74,
                    0x8a,
                    0x63,
                    0x79,
                    0x85,
                    0x77,
                    0x1d,
                    0x34,
                    0x7f,
                    0x05,
                    0x45,
                    0x65,
                    0x9f,
                    0x14,
                    0xe9,
                    0x9d,
                    0xef,
                    0x84,
                    0x2d,
                    0x8b,
                    0x00,
                    0x14,
                    0x4a,
                    0x1f,
                    0xec,
                    0x6a,
                    0xdf,
                    0x0c,
                    0x9a,
                    0x92,
                    0x7f,
                    0xee,
                    0xa6,
                    0xc5,
                    0x11,
                    0x60,
                },
            .len = 104,
        },
    .salt =
        {
            .data = {0x11, 0x22, 0x33, 0x44},
            .len = 4,
        },

    .iv =
        {
            .data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08},
        },

    .ipsec_xform =
        {
            .spi = 0x4a2cbfe3,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .options.ip_csum_enable = 0,
            .options.l4_csum_enable = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = true,

    .xform =
        {
            .aead =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AEAD,
                    .aead =
                        {
                            .op = RTE_CRYPTO_AEAD_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_AEAD_AES_GCM,
                            .key.length = 32,
                            .iv.length = 12,
                            .iv.offset = IV_OFFSET,
                            .digest_length = 16,
                            .aad_length = 12,
                        },
                },
        },
};

/* Known vectors for AES-CBC
 * https://datatracker.ietf.org/doc/html/rfc3602#section-4
 */

struct ipsec_test_data pkt_aes_128_cbc_null = {
    .key =
        {
            .data =
                {
                    0x01,
                    0x23,
                    0x45,
                    0x67,
                    0x89,
                    0xab,
                    0xcd,
                    0xef,
                    0x01,
                    0x23,
                    0x45,
                    0x67,
                    0x89,
                    0xab,
                    0xcd,
                    0xef,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP - outer header */
                    0x45,
                    0x00,
                    0x00,
                    0x8c,
                    0x00,
                    0x02,
                    0x00,
                    0x00,
                    0x40,
                    0x32,
                    0x27,
                    0xbc,
                    0x00,
                    0x01,
                    0xa8,
                    0xc0,
                    0x01,
                    0x01,
                    0xa8,
                    0xc0,

                    /* ESP */
                    0x00,
                    0x00,
                    0x87,
                    0x65,
                    0x00,
                    0x00,
                    0x00,
                    0x02,

                    /* IV */
                    0xf4,
                    0xe7,
                    0x65,
                    0x24,
                    0x4f,
                    0x64,
                    0x07,
                    0xad,
                    0xf1,
                    0x3d,
                    0xc1,
                    0x38,
                    0x0f,
                    0x67,
                    0x3f,
                    0x37,

                    /* Data */
                    0x77,
                    0x3b,
                    0x52,
                    0x41,
                    0xa4,
                    0xc4,
                    0x49,
                    0x22,
                    0x5e,
                    0x4f,
                    0x3c,
                    0xe5,
                    0xed,
                    0x61,
                    0x1b,
                    0x0c,
                    0x23,
                    0x7c,
                    0xa9,
                    0x6c,
                    0xf7,
                    0x4a,
                    0x93,
                    0x01,
                    0x3c,
                    0x1b,
                    0x0e,
                    0xa1,
                    0xa0,
                    0xcf,
                    0x70,
                    0xf8,
                    0xe4,
                    0xec,
                    0xae,
                    0xc7,
                    0x8a,
                    0xc5,
                    0x3a,
                    0xad,
                    0x7a,
                    0x0f,
                    0x02,
                    0x2b,
                    0x85,
                    0x92,
                    0x43,
                    0xc6,
                    0x47,
                    0x75,
                    0x2e,
                    0x94,
                    0xa8,
                    0x59,
                    0x35,
                    0x2b,
                    0x8a,
                    0x4d,
                    0x4d,
                    0x2d,
                    0xec,
                    0xd1,
                    0x36,
                    0xe5,
                    0xc1,
                    0x77,
                    0xf1,
                    0x32,
                    0xad,
                    0x3f,
                    0xbf,
                    0xb2,
                    0x20,
                    0x1a,
                    0xc9,
                    0x90,
                    0x4c,
                    0x74,
                    0xee,
                    0x0a,
                    0x10,
                    0x9e,
                    0x0c,
                    0xa1,
                    0xe4,
                    0xdf,
                    0xe9,
                    0xd5,
                    0xa1,
                    0x00,
                    0xb8,
                    0x42,
                    0xf1,
                    0xc2,
                    0x2f,
                    0x0d,
                },
            .len = 140,
        },
    .output_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x54,
                    0x09,
                    0x04,
                    0x00,
                    0x00,
                    0x40,
                    0x01,
                    0xf9,
                    0x88,
                    0xc0,
                    0xa8,
                    0x7b,
                    0x03,
                    0xc0,
                    0xa8,
                    0x7b,
                    0xc8,

                    /* ICMP */
                    0x08,
                    0x00,
                    0x9f,
                    0x76,
                    0xa9,
                    0x0a,
                    0x01,
                    0x00,
                    0xb4,
                    0x9c,
                    0x08,
                    0x3d,
                    0x02,
                    0xa2,
                    0x04,
                    0x00,
                    0x08,
                    0x09,
                    0x0a,
                    0x0b,
                    0x0c,
                    0x0d,
                    0x0e,
                    0x0f,
                    0x10,
                    0x11,
                    0x12,
                    0x13,
                    0x14,
                    0x15,
                    0x16,
                    0x17,
                    0x18,
                    0x19,
                    0x1a,
                    0x1b,
                    0x1c,
                    0x1d,
                    0x1e,
                    0x1f,
                    0x20,
                    0x21,
                    0x22,
                    0x23,
                    0x24,
                    0x25,
                    0x26,
                    0x27,
                    0x28,
                    0x29,
                    0x2a,
                    0x2b,
                    0x2c,
                    0x2d,
                    0x2e,
                    0x2f,
                    0x30,
                    0x31,
                    0x32,
                    0x33,
                    0x34,
                    0x35,
                    0x36,
                    0x37,
                    0x01,
                    0x02,
                    0x03,
                    0x04,
                    0x05,
                    0x06,
                    0x07,
                    0x08,
                    0x09,
                    0x0a,
                    0x0a,
                    0x04,
                },
            .len = 84,
        },
    .iv =
        {
            .data =
                {
                    0xf4,
                    0xe7,
                    0x65,
                    0x24,
                    0x4f,
                    0x64,
                    0x07,
                    0xad,
                    0xf1,
                    0x3d,
                    0xc1,
                    0x38,
                    0x0f,
                    0x67,
                    0x3f,
                    0x37,
                },
        },

    .ipsec_xform =
        {
            .spi = 0x8765,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_INGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = false,

    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_DECRYPT,
                            .algo = RTE_CRYPTO_CIPHER_AES_CBC,
                            .key.length = 16,
                            .iv.length = 16,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .algo = RTE_CRYPTO_AUTH_NULL,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_256_gcm_v6 = {
    .key =
        {
            .data =
                {
                    0xde, 0x12, 0xbe, 0x56, 0xde, 0xad, 0xbe, 0xef,
                    0xde, 0xad, 0xbe, 0xef, 0xde, 0xad, 0xbe, 0xef,
                    0x12, 0x78, 0xbe, 0x34, 0x01, 0x02, 0x03, 0x07,
                    0xaa, 0xbb, 0xcc, 0xf1, 0x08, 0x07, 0x06, 0x05,
                },
        },
    .input_text =
        {
            .data =
                {
                    0x60, 0x00, 0x00, 0x00, 0x00, 0x20, 0x06, 0x38, 0x26,
                    0x07, 0xf8, 0xb0, 0x40, 0x0c, 0x0c, 0x03, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0x20, 0x01, 0x04,
                    0x70, 0xe5, 0xbf, 0xde, 0xad, 0x49, 0x57, 0x21, 0x74,
                    0xe8, 0x2c, 0x48, 0x87, 0x00, 0x19, 0xf9, 0xc7, 0x95,
                    0x63, 0x97, 0x9c, 0x03, 0xa0, 0x88, 0x31, 0x80, 0x12,
                    0xa7, 0xd6, 0x25, 0x83, 0x00, 0x00, 0x02, 0x04, 0x05,
                    0x6a, 0x01, 0x01, 0x04, 0x02, 0x01, 0x03, 0x03, 0x07,
                },
            .len = 72,
        },
    .output_text =
        {
            .data =
                {
                    0x60, 0x00, 0x00, 0x00, 0x00, 0x6c, 0x32, 0x40, 0x12, 0x34,
                    0x12, 0x21, 0x17, 0x45, 0x11, 0x34, 0x11, 0xfc, 0x89, 0x71,
                    0xdf, 0x22, 0x56, 0x78, 0x12, 0x34, 0x12, 0x21, 0x17, 0x45,
                    0x11, 0x34, 0x11, 0xfc, 0x89, 0x71, 0xdf, 0x22, 0x34, 0x56,
                    0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x01, 0x45, 0xad,
                    0xfe, 0x23, 0x78, 0x56, 0x12, 0x00, 0xe7, 0xdf, 0xc4, 0x7e,
                    0x21, 0xbd, 0xec, 0x1b, 0x74, 0x5a, 0xe4, 0x7e, 0x2e, 0x94,
                    0x21, 0x0a, 0x9b, 0x0e, 0x59, 0xbe, 0x06, 0x2a, 0xda, 0xb8,
                    0x6b, 0x48, 0x7f, 0x0b, 0x88, 0x3a, 0xa9, 0xfd, 0x3c, 0xfe,
                    0x9f, 0xb1, 0x8c, 0x67, 0xd2, 0xf8, 0xaf, 0xb5, 0xad, 0x16,
                    0xdb, 0xff, 0x8d, 0x50, 0xd3, 0x48, 0xf5, 0x6c, 0x3c, 0x0c,
                    0x27, 0x34, 0x2b, 0x65, 0xc8, 0xff, 0xeb, 0x5f, 0xb8, 0xff,
                    0x12, 0x00, 0x1c, 0x9f, 0xb7, 0x85, 0xdd, 0x7d, 0x40, 0x19,
                    0xcb, 0x18, 0xeb, 0x15, 0xc4, 0x88, 0xe1, 0xc2, 0x91, 0xc7,
                    0xb1, 0x65, 0xc3, 0x27, 0x16, 0x06, 0x8f, 0xf2,
                },
            .len = 148,
        },
    .salt =
        {
            .data = {0x11, 0x22, 0x33, 0x44},
            .len = 4,
        },

    .iv =
        {
            .data =
                {
                    0x45,
                    0xad,
                    0xfe,
                    0x23,
                    0x78,
                    0x56,
                    0x12,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV6,
            .replay_win_sz = 0,
        },

    .aead = true,

    .xform =
        {
            .aead =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AEAD,
                    .aead =
                        {
                            .op = RTE_CRYPTO_AEAD_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_AEAD_AES_GCM,
                            .key.length = 32,
                            .iv.length = 12,
                            .iv.offset = IV_OFFSET,
                            .digest_length = 16,
                            .aad_length = 12,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_128_cbc_hmac_sha256 = {
    .key =
        {
            .data =
                {
                    0x00,
                    0x04,
                    0x05,
                    0x01,
                    0x23,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x0a,
                    0x0b,
                    0x0c,
                    0x0f,
                    0x00,
                    0x00,
                },
        },
    .auth_key =
        {
            .data =
                {
                    0xde, 0x34, 0x56, 0x00, 0x00, 0x00, 0x78, 0x00,
                    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
                    0x10, 0x30, 0x40, 0x00, 0x01, 0x02, 0x03, 0x04,
                    0x0a, 0x0b, 0x0c, 0x0d, 0x05, 0x06, 0x07, 0x08,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x32,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x1f,
                    0x11,
                    0x17,
                    0x8b,
                    0xc0,
                    0xa8,
                    0x01,
                    0x6f,
                    0xc0,
                    0xa8,
                    0x01,
                    0x70,

                    /* UDP */
                    0x00,
                    0x09,
                    0x00,
                    0x09,
                    0x00,
                    0x1e,
                    0x00,
                    0x00,
                    0xbe,
                    0x9b,
                    0xe9,
                    0x55,
                    0x00,
                    0x00,
                    0x00,
                    0x21,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
            .len = 50,
        },
    .output_text =
        {
            .data =
                {
                    /* IP - outer header */
                    0x45,
                    0x00,
                    0x00,
                    0x7c,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x40,
                    0x32,
                    0x52,
                    0x4d,
                    0x14,
                    0x00,
                    0x00,
                    0x01,
                    0x14,
                    0x00,
                    0x00,
                    0x02,

                    /* ESP */
                    0x00,
                    0x00,
                    0x00,
                    0x34,
                    0x00,
                    0x00,
                    0x00,
                    0x01,

                    /* IV */
                    0x34,
                    0x12,
                    0x67,
                    0x45,
                    0xff,
                    0xff,
                    0x00,
                    0x00,
                    0x20,
                    0xbf,
                    0xe8,
                    0x39,
                    0x00,
                    0x00,
                    0x00,
                    0x00,

                    /* Data */
                    0x67,
                    0xb5,
                    0x46,
                    0x6e,
                    0x78,
                    0x17,
                    0xd3,
                    0x5a,
                    0xac,
                    0x62,
                    0x62,
                    0x62,
                    0xb0,
                    0x57,
                    0x9b,
                    0x09,
                    0x19,
                    0x4f,
                    0x06,
                    0x59,
                    0xc8,
                    0xb0,
                    0x30,
                    0x65,
                    0x1f,
                    0x45,
                    0x57,
                    0x41,
                    0x72,
                    0x17,
                    0x28,
                    0xe9,
                    0xad,
                    0x50,
                    0xbe,
                    0x44,
                    0x1d,
                    0x2d,
                    0x9a,
                    0xd0,
                    0x48,
                    0x75,
                    0x0d,
                    0x1c,
                    0x8d,
                    0x24,
                    0xa8,
                    0x6f,
                    0x6b,
                    0x24,
                    0xb6,
                    0x5d,
                    0x43,
                    0x1e,
                    0x55,
                    0xf0,
                    0xf7,
                    0x14,
                    0x1f,
                    0xf2,
                    0x61,
                    0xd4,
                    0xe0,
                    0x30,
                    0x16,
                    0xbe,
                    0x1b,
                    0x5c,
                    0xcc,
                    0xb7,
                    0x66,
                    0x1c,
                    0x47,
                    0xad,
                    0x07,
                    0x6c,
                    0xd5,
                    0xcb,
                    0xce,
                    0x6c,
                },
            .len = 124,
        },
    .iv =
        {
            .data =
                {
                    0x34,
                    0x12,
                    0x67,
                    0x45,
                    0xff,
                    0xff,
                    0x00,
                    0x00,
                    0x20,
                    0xbf,
                    0xe8,
                    0x39,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = false,

    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_CIPHER_AES_CBC,
                            .key.length = 16,
                            .iv.length = 16,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .op = RTE_CRYPTO_AUTH_OP_GENERATE,
                            .algo = RTE_CRYPTO_AUTH_SHA256_HMAC,
                            .key.length = 32,
                            .digest_length = 16,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_128_cbc_hmac_sha384 = {
    .key =
        {
            .data =
                {
                    0x00,
                    0x04,
                    0x05,
                    0x01,
                    0x23,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x0a,
                    0x0b,
                    0x0c,
                    0x0f,
                    0x00,
                    0x00,
                },
        },
    .auth_key =
        {
            .data =
                {
                    0x10, 0x30, 0x40, 0x00, 0x01, 0x02, 0x03, 0x04, 0x0a, 0x0b,
                    0x0c, 0x0d, 0x05, 0x06, 0x07, 0x08, 0xde, 0x34, 0x56, 0x00,
                    0x00, 0x00, 0x78, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
                    0x00, 0x02, 0x10, 0x30, 0x40, 0x00, 0x01, 0x02, 0x03, 0x34,
                    0x1a, 0x0b, 0x0c, 0x0d, 0x05, 0x06, 0x07, 0x08,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x32,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x1f,
                    0x11,
                    0x17,
                    0x8b,
                    0xc0,
                    0xa8,
                    0x01,
                    0x6f,
                    0xc0,
                    0xa8,
                    0x01,
                    0x70,

                    /* UDP */
                    0x00,
                    0x09,
                    0x00,
                    0x09,
                    0x00,
                    0x1e,
                    0x00,
                    0x00,
                    0xbe,
                    0x9b,
                    0xe9,
                    0x55,
                    0x00,
                    0x00,
                    0x00,
                    0x21,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
            .len = 50,
        },
    .output_text =
        {
            .data =
                {
                    0x45, 0x00, 0x00, 0x84, 0x00, 0x01, 0x00, 0x00, 0x40, 0x32,
                    0x52, 0x45, 0x14, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x02,
                    0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x44, 0x24, 0xb9, 0xd8, 0x0f, 0xbe,
                    0xa3, 0x3f, 0xc9, 0xc0, 0xa2, 0xcb, 0xaa, 0xda, 0x3f, 0xc6,
                    0x0e, 0x88, 0x75, 0x96, 0x25, 0x50, 0x07, 0x4d, 0x52, 0xf4,
                    0x75, 0xec, 0xd8, 0xcd, 0xe4, 0xcf, 0x85, 0x9a, 0xbc, 0x9e,
                    0x84, 0x0f, 0xbb, 0x83, 0x72, 0x0c, 0x7f, 0x58, 0x02, 0x46,
                    0xeb, 0x86, 0x6e, 0xd1, 0xcf, 0x05, 0x6a, 0xd1, 0xd2, 0xc6,
                    0xb5, 0x94, 0x09, 0x0a, 0x3e, 0xdf, 0x09, 0xfb, 0x0a, 0xb7,
                    0xb4, 0x97, 0x17, 0xf2, 0x20, 0xaf, 0xfa, 0x90, 0x92, 0x4d,
                    0xe4, 0x0e, 0xef, 0x5a, 0xe8, 0x43, 0x46, 0xa8, 0x5e, 0x3f,
                    0x52, 0x46,
                },
            .len = 132,
        },
    .iv =
        {
            .data =
                {
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = false,

    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_CIPHER_AES_CBC,
                            .key.length = 16,
                            .iv.length = 16,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .op = RTE_CRYPTO_AUTH_OP_GENERATE,
                            .algo = RTE_CRYPTO_AUTH_SHA384_HMAC,
                            .key.length = 48,
                            .digest_length = 24,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_128_cbc_hmac_sha512 = {
    .key =
        {
            .data =
                {
                    0x00,
                    0x04,
                    0x05,
                    0x01,
                    0x23,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x0a,
                    0x0b,
                    0x0c,
                    0x0f,
                    0x00,
                    0x00,
                },
        },
    .auth_key =
        {
            .data =
                {
                    0xde, 0x34, 0x56, 0x00, 0x00, 0x00, 0x78, 0x00, 0x00, 0x10,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x10, 0x30, 0x40, 0x00,
                    0x01, 0x02, 0x03, 0x04, 0x0a, 0x0b, 0x0c, 0x0d, 0x05, 0x06,
                    0x07, 0x08, 0xde, 0x34, 0x56, 0x00, 0x00, 0x00, 0x78, 0x00,
                    0x00, 0x10, 0x00, 0x00, 0x00, 0x10, 0x00, 0x02, 0x10, 0x30,
                    0x40, 0x00, 0x01, 0x02, 0x03, 0x34, 0x1a, 0x0b, 0x0c, 0x0d,
                    0x05, 0x06, 0x07, 0x08,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x32,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x1f,
                    0x11,
                    0x17,
                    0x8b,
                    0xc0,
                    0xa8,
                    0x01,
                    0x6f,
                    0xc0,
                    0xa8,
                    0x01,
                    0x70,

                    /* UDP */
                    0x00,
                    0x09,
                    0x00,
                    0x09,
                    0x00,
                    0x1e,
                    0x00,
                    0x00,
                    0xbe,
                    0x9b,
                    0xe9,
                    0x55,
                    0x00,
                    0x00,
                    0x00,
                    0x21,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
            .len = 50,
        },
    .output_text =
        {
            .data =
                {
                    0x45, 0x00, 0x00, 0x8c, 0x00, 0x01, 0x00, 0x00, 0x40, 0x32,
                    0x52, 0x3d, 0x14, 0x00, 0x00, 0x01, 0x14, 0x00, 0x00, 0x02,
                    0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x01, 0x42, 0x32,
                    0x76, 0x65, 0x45, 0x35, 0x24, 0x41, 0xf0, 0xc1, 0xb4, 0x40,
                    0x00, 0x00, 0x00, 0x00, 0xd0, 0x32, 0x23, 0xf7, 0xcd, 0x3d,
                    0xdb, 0xd5, 0x70, 0x19, 0x1b, 0xf5, 0x8f, 0xeb, 0x98, 0x3d,
                    0x41, 0x5c, 0x28, 0xdd, 0xfd, 0xcc, 0xdd, 0xa2, 0xeb, 0x43,
                    0x4c, 0x13, 0x2d, 0xa1, 0x98, 0x87, 0x92, 0x3a, 0x1f, 0x67,
                    0x20, 0x8d, 0x9e, 0x8e, 0x51, 0x21, 0x4c, 0xa9, 0xff, 0xad,
                    0xfb, 0x5d, 0x57, 0xa3, 0x16, 0x91, 0xaa, 0x75, 0xc7, 0x28,
                    0x42, 0x4e, 0x8f, 0x8e, 0x84, 0x37, 0x94, 0x09, 0x74, 0xfa,
                    0x70, 0x0d, 0xd1, 0x37, 0xe2, 0x7c, 0x54, 0xdd, 0x2e, 0xb4,
                    0xf4, 0x54, 0x4b, 0x12, 0xe0, 0xaf, 0x4a, 0x0a, 0x0b, 0x52,
                    0x57, 0x9d, 0x36, 0xdc, 0xac, 0x02, 0xfb, 0x55, 0x34, 0x05,
                },
            .len = 140,
        },
    .iv =
        {
            .data =
                {
                    0x42,
                    0x32,
                    0x76,
                    0x65,
                    0x45,
                    0x35,
                    0x24,
                    0x41,
                    0xf0,
                    0xc1,
                    0xb4,
                    0x40,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = false,

    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_CIPHER_AES_CBC,
                            .key.length = 16,
                            .iv.length = 16,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .op = RTE_CRYPTO_AUTH_OP_GENERATE,
                            .algo = RTE_CRYPTO_AUTH_SHA512_HMAC,
                            .key.length = 64,
                            .digest_length = 32,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_128_cbc_hmac_sha256_v6 = {
    .key =
        {
            .data =
                {
                    0x00,
                    0x04,
                    0x05,
                    0x01,
                    0x23,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x0a,
                    0x0b,
                    0x0c,
                    0x0f,
                    0x00,
                    0x00,
                },
        },
    .auth_key =
        {
            .data =
                {
                    0xde, 0x34, 0x56, 0x00, 0x00, 0x00, 0x78, 0x00,
                    0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
                    0x10, 0x30, 0x40, 0x00, 0x01, 0x02, 0x03, 0x04,
                    0x0a, 0x0b, 0x0c, 0x0d, 0x05, 0x06, 0x07, 0x08,
                },
        },
    .input_text =
        {
            .data =
                {
                    0x60, 0x00, 0x00, 0x00, 0x00, 0x20, 0x06, 0x38, 0x26,
                    0x07, 0xf8, 0xb0, 0x40, 0x0c, 0x0c, 0x03, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x1a, 0x20, 0x01, 0x04,
                    0x70, 0xe5, 0xbf, 0xde, 0xad, 0x49, 0x57, 0x21, 0x74,
                    0xe8, 0x2c, 0x48, 0x87, 0x00, 0x19, 0xf9, 0xc7, 0x95,
                    0x63, 0x97, 0x9c, 0x03, 0xa0, 0x88, 0x31, 0x80, 0x12,
                    0xa7, 0xd6, 0x25, 0x83, 0x00, 0x00, 0x02, 0x04, 0x05,
                    0x6a, 0x01, 0x01, 0x04, 0x02, 0x01, 0x03, 0x03, 0x07,
                },
            .len = 72,
        },
    .output_text =
        {
            .data =
                {
                    0x60, 0x00, 0x00, 0x00, 0x00, 0x78, 0x32, 0x40, 0x12, 0x34,
                    0x12, 0x21, 0x17, 0x45, 0x11, 0x34, 0x11, 0xfc, 0x89, 0x71,
                    0xdf, 0x22, 0x56, 0x78, 0x12, 0x34, 0x12, 0x21, 0x17, 0x45,
                    0x11, 0x34, 0x11, 0xfc, 0x89, 0x71, 0xdf, 0x22, 0x34, 0x56,
                    0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x01, 0x45, 0xad,
                    0xfe, 0x23, 0x78, 0x56, 0x12, 0x00, 0xf0, 0xc1, 0x05, 0x3c,
                    0x00, 0x00, 0x00, 0x00, 0x1b, 0x1c, 0x98, 0x6e, 0x2a, 0xce,
                    0x61, 0xef, 0xc1, 0xdd, 0x25, 0x96, 0x5c, 0xb1, 0xb0, 0x15,
                    0x47, 0x25, 0xb7, 0x8b, 0x00, 0xb6, 0xbb, 0xe6, 0x2e, 0x29,
                    0xcb, 0x4a, 0x94, 0x00, 0xf0, 0x73, 0xdb, 0x14, 0x32, 0xd9,
                    0xa2, 0xdf, 0x22, 0x2f, 0x52, 0x3e, 0x79, 0x77, 0xf3, 0x17,
                    0xaa, 0x40, 0x1c, 0x57, 0x27, 0x12, 0x82, 0x44, 0x35, 0xb8,
                    0x64, 0xe0, 0xaa, 0x5c, 0x10, 0xc7, 0x97, 0x35, 0x9c, 0x6b,
                    0x1c, 0xf7, 0xe7, 0xbd, 0x83, 0x33, 0x77, 0x48, 0x44, 0x7d,
                    0xa4, 0x13, 0x74, 0x3b, 0x6a, 0x91, 0xd0, 0xd8, 0x7d, 0x41,
                    0x45, 0x23, 0x5d, 0xc9, 0x2d, 0x08, 0x7a, 0xd8, 0x25, 0x8e,
                },
            .len = 160,
        },
    .iv =
        {
            .data =
                {
                    0x45,
                    0xad,
                    0xfe,
                    0x23,
                    0x78,
                    0x56,
                    0x12,
                    0x00,
                    0xf0,
                    0xc1,
                    0x05,
                    0x3c,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV6,
            .replay_win_sz = 0,
        },

    .aead = false,

    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_CIPHER_AES_CBC,
                            .key.length = 16,
                            .iv.length = 16,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .op = RTE_CRYPTO_AUTH_OP_GENERATE,
                            .algo = RTE_CRYPTO_AUTH_SHA256_HMAC,
                            .key.length = 32,
                            .digest_length = 16,
                        },
                },
        },
};

struct ipsec_test_data pkt_aes_128_gcm_frag = {
    .key =
        {
            .data =
                {
                    0xde,
                    0xad,
                    0xbe,
                    0xef,
                    0xde,
                    0xad,
                    0xbe,
                    0xef,
                    0xde,
                    0xad,
                    0xbe,
                    0xef,
                    0xde,
                    0xad,
                    0xbe,
                    0xef,
                },
        },
    .input_text =
        {
            .data =
                {
                    0x45, 0x00, 0x00, 0x6e, 0x00, 0x01, 0x00, 0x17, 0x40, 0x06,
                    0xed, 0x48, 0xc6, 0x12, 0x00, 0x00, 0xc6, 0x12, 0x01, 0x05,
                    0x00, 0x14, 0x00, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x50, 0x00, 0x00, 0x00, 0x55, 0x05, 0x00, 0x00,
                    0x00, 0x01, 0x02, 0x03, 0xf2, 0xf6, 0xe9, 0x21, 0xf9, 0xf2,
                    0xf6, 0xe9, 0x21, 0xf9, 0xf2, 0xf6, 0xe9, 0x21, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                },
            .len = 110,
        },
    .output_text =
        {
            .data =
                {
                    0x45, 0x00, 0x00, 0xa4, 0x00, 0x01, 0x00, 0x00, 0x40, 0x32,
                    0xf6, 0x0c, 0xc0, 0xa8, 0x01, 0x70, 0xc0, 0xa8, 0x01, 0x5a,
                    0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x01, 0x45, 0xad,
                    0xfe, 0x23, 0x78, 0x56, 0x12, 0x00, 0x49, 0x26, 0xac, 0x4e,
                    0x8d, 0xf3, 0x74, 0x26, 0x18, 0x3f, 0x65, 0x94, 0x73, 0x2e,
                    0xe4, 0xcf, 0x84, 0x6d, 0x03, 0x8a, 0x4c, 0xdd, 0x2d, 0xef,
                    0xcd, 0x9f, 0x84, 0x76, 0x93, 0xe1, 0xee, 0x21, 0x92, 0x8b,
                    0xf7, 0x7a, 0xb1, 0x6a, 0x7f, 0xd6, 0x10, 0x66, 0xdd, 0xa1,
                    0x8b, 0x17, 0x56, 0x99, 0x9a, 0x40, 0xd0, 0x6b, 0x2d, 0xe0,
                    0x55, 0x40, 0x2f, 0xb8, 0x38, 0xe3, 0x08, 0x46, 0xe2, 0x69,
                    0xc9, 0xa1, 0x85, 0x9d, 0x7b, 0xec, 0x33, 0x2a, 0x2d, 0x1d,
                    0x1f, 0x1a, 0x9e, 0xf0, 0x1e, 0xc3, 0x33, 0x64, 0x35, 0x82,
                    0xbb, 0xb5, 0x7a, 0x91, 0x2e, 0x8d, 0xd5, 0x5b, 0x3a, 0xbe,
                    0x95, 0x94, 0xba, 0x40, 0x73, 0x4e, 0xa4, 0x15, 0xe4, 0x4a,
                    0xf9, 0x14, 0x2c, 0x4f, 0x63, 0x2e, 0x23, 0x6e, 0xeb, 0x06,
                    0xe7, 0x52, 0xe1, 0xc7, 0x91, 0x7f, 0x19, 0xc0, 0x4a, 0xd2,
                    0xd5, 0x3e, 0x84, 0xa8,
                },
            .len = 164,
        },
    .salt =
        {
            .data =
                {
                    0xde,
                    0xad,
                    0xbe,
                    0xef,
                },
            .len = 4,
        },

    .iv =
        {
            .data =
                {
                    0x45,
                    0xad,
                    0xfe,
                    0x23,
                    0x78,
                    0x56,
                    0x12,
                    0x00,
                },
        },

    .ipsec_xform =
        {
            .spi = 52,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .options.ip_csum_enable = 0,
            .options.l4_csum_enable = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },

    .aead = true,

    .xform =
        {
            .aead =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AEAD,
                    .aead =
                        {
                            .op = RTE_CRYPTO_AEAD_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_AEAD_AES_GCM,
                            .key.length = 16,
                            .iv.length = 12,
                            .iv.offset = IV_OFFSET,
                            .digest_length = 16,
                            .aad_length = 12,
                        },
                },
        },
};

struct ipsec_test_data pkt_null_aes_xcbc = {
    .auth_key =
        {
            .data =
                {
                    0x61,
                    0x31,
                    0x62,
                    0x32,
                    0x63,
                    0x33,
                    0x64,
                    0x34,
                    0x65,
                    0x35,
                    0x66,
                    0x36,
                    0x67,
                    0x37,
                    0x68,
                    0x38,
                },
        },
    .input_text =
        {
            .data =
                {
                    /* IP */
                    0x45, 0x00, 0x00, 0x2f, 0x49, 0x37, 0x00, 0x00, 0x40, 0x11,
                    0x22, 0x84, 0x0d, 0x00, 0x00, 0x02, 0x02, 0x00, 0x00, 0x02,
                    0x08, 0x00, 0x08, 0x00, 0x00, 0x1b, 0x6d, 0x99, 0x58, 0x58,
                    0x58, 0x58, 0x58, 0x58, 0x58, 0x58, 0x58, 0x58, 0x58, 0x58,
                    0x58, 0x58, 0x58, 0x58, 0x58, 0x58, 0x58,
                },
            .len = 47,
        },
    .output_text =
        {
            .data =
                {
                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x5c,
                    0x06,
                    0x00,
                    0x00,
                    0x00,
                    0x40,
                    0x32,
                    0x13,
                    0x6c,
                    0x0a,
                    0x00,
                    0x6f,
                    0x02,
                    0x0a,
                    0x00,
                    0xde,
                    0x02,

                    /* ESP */
                    0x00,
                    0x00,
                    0x01,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x01,

                    /* IP */
                    0x45,
                    0x00,
                    0x00,
                    0x2f,
                    0x49,
                    0x37,
                    0x00,
                    0x00,
                    0x40,
                    0x11,
                    0x22,
                    0x84,
                    0x0d,
                    0x00,
                    0x00,
                    0x02,
                    0x02,
                    0x00,
                    0x00,
                    0x02,
                    0x08,
                    0x00,
                    0x08,
                    0x00,
                    0x00,
                    0x1b,
                    0x6d,
                    0x99,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,
                    0x58,

                    /* ESP trailer */
                    0x01,
                    0x02,
                    0x03,
                    0x03,
                    0x04,

                    /* ICV */
                    0xf1,
                    0x52,
                    0x64,
                    0xd1,
                    0x9b,
                    0x62,
                    0x24,
                    0xdd,
                    0xcc,
                    0x14,
                    0xf5,
                    0xc1,
                },
            .len = 92,
        },
    .ipsec_xform =
        {
            .spi = 0x100,
            .options.esn = 0,
            .options.udp_encap = 0,
            .options.copy_dscp = 0,
            .options.copy_flabel = 0,
            .options.copy_df = 0,
            .options.dec_ttl = 0,
            .options.ecn = 0,
            .options.stats = 0,
            .options.tunnel_hdr_verify = 0,
            .options.ip_csum_enable = 0,
            .options.l4_csum_enable = 0,
            .direction = RTE_SECURITY_IPSEC_SA_DIR_EGRESS,
            .proto = RTE_SECURITY_IPSEC_SA_PROTO_ESP,
            .mode = RTE_SECURITY_IPSEC_SA_MODE_TUNNEL,
            .tunnel.type = RTE_SECURITY_IPSEC_TUNNEL_IPV4,
            .replay_win_sz = 0,
        },
    .aead = false,
    .xform =
        {
            .chain.cipher =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_CIPHER,
                    .cipher =
                        {
                            .op = RTE_CRYPTO_CIPHER_OP_ENCRYPT,
                            .algo = RTE_CRYPTO_CIPHER_NULL,
                            .key.length = 0,
                            .iv.length = 0,
                        },
                },
            .chain.auth =
                {
                    .next = NULL,
                    .type = RTE_CRYPTO_SYM_XFORM_AUTH,
                    .auth =
                        {
                            .op = RTE_CRYPTO_AUTH_OP_GENERATE,
                            .algo = RTE_CRYPTO_AUTH_AES_XCBC_MAC,
                            .key.length = 16,
                            .digest_length = 12,
                        },
                },
        },
};

#endif /* TEST_CRYPTODEV_SECURITY_IPSEC_TEST_VECTORS_H_ */