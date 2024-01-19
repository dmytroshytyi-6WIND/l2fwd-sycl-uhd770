/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright (C) 2020 Marvell International Ltd.
 */

#ifndef __TEST_CRYPTODEV_ECPM_TEST_VECTORS_H__
#define __TEST_CRYPTODEV_ECPM_TEST_VECTORS_H__

#include "rte_crypto_asym.h"

struct crypto_testsuite_ecpm_params {
  rte_crypto_param gen_x;
  rte_crypto_param gen_y;
  rte_crypto_param privkey;
  rte_crypto_param pubkey_x;
  rte_crypto_param pubkey_y;
  int curve;
};

/* SECP192R1 (P-192 NIST) test vectors */

static uint8_t gen_x_secp192r1[] = {
    0x18, 0x8d, 0xa8, 0x0e, 0xb0, 0x30, 0x90, 0xf6, 0x7c, 0xbf, 0x20, 0xeb,
    0x43, 0xa1, 0x88, 0x00, 0xf4, 0xff, 0x0a, 0xfd, 0x82, 0xff, 0x10, 0x12};

static uint8_t gen_y_secp192r1[] = {
    0x07, 0x19, 0x2b, 0x95, 0xff, 0xc8, 0xda, 0x78, 0x63, 0x10, 0x11, 0xed,
    0x6b, 0x24, 0xcd, 0xd5, 0x73, 0xf9, 0x77, 0xa1, 0x1e, 0x79, 0x48, 0x11};

static uint8_t privkey_secp192r1[] = {
    0x24, 0xed, 0xd2, 0x2f, 0x7d, 0xdd, 0x6f, 0xa5, 0xbc, 0x61, 0xfc, 0x06,
    0x53, 0x47, 0x9a, 0xa4, 0x08, 0x09, 0xef, 0x86, 0x5c, 0xf2, 0x7a, 0x47};
static uint8_t pubkey_x_secp192r1[] = {
    0x9b, 0xf1, 0x2d, 0x71, 0x74, 0xb7, 0x70, 0x8a, 0x07, 0x6a, 0x38, 0xbc,
    0x80, 0xaa, 0x28, 0x66, 0x2f, 0x25, 0x1e, 0x2e, 0xd8, 0xd4, 0x14, 0xdc};

static uint8_t pubkey_y_secp192r1[] = {
    0x48, 0x54, 0xc8, 0xd0, 0x7d, 0xfc, 0x08, 0x82, 0x4e, 0x9e, 0x47, 0x1c,
    0xa2, 0xfe, 0xdc, 0xfc, 0xff, 0x3d, 0xdc, 0xb0, 0x11, 0x57, 0x34, 0x98};

struct crypto_testsuite_ecpm_params ecpm_param_secp192r1 = {
    .gen_x =
        {
            .data = gen_x_secp192r1,
            .length = sizeof(gen_x_secp192r1),
        },
    .gen_y =
        {
            .data = gen_y_secp192r1,
            .length = sizeof(gen_y_secp192r1),
        },
    .privkey =
        {
            .data = privkey_secp192r1,
            .length = sizeof(privkey_secp192r1),
        },
    .pubkey_x =
        {
            .data = pubkey_x_secp192r1,
            .length = sizeof(pubkey_x_secp192r1),
        },
    .pubkey_y =
        {
            .data = pubkey_y_secp192r1,
            .length = sizeof(pubkey_y_secp192r1),
        },
    .curve = RTE_CRYPTO_EC_GROUP_SECP192R1};

/* SECP224R1 (P-224 NIST) test vectors */

static uint8_t gen_x_secp224r1[] = {0xb7, 0x0e, 0x0c, 0xbd, 0x6b, 0xb4, 0xbf,
                                    0x7f, 0x32, 0x13, 0x90, 0xb9, 0x4a, 0x03,
                                    0xc1, 0xd3, 0x56, 0xc2, 0x11, 0x22, 0x34,
                                    0x32, 0x80, 0xd6, 0x11, 0x5c, 0x1d, 0x21};

static uint8_t gen_y_secp224r1[] = {0xbd, 0x37, 0x63, 0x88, 0xb5, 0xf7, 0x23,
                                    0xfb, 0x4c, 0x22, 0xdf, 0xe6, 0xcd, 0x43,
                                    0x75, 0xa0, 0x5a, 0x07, 0x47, 0x64, 0x44,
                                    0xd5, 0x81, 0x99, 0x85, 0x00, 0x7e, 0x34};

static uint8_t privkey_secp224r1[] = {0x88, 0x8f, 0xc9, 0x92, 0x89, 0x3b, 0xdd,
                                      0x8a, 0xa0, 0x2c, 0x80, 0x76, 0x88, 0x32,
                                      0x60, 0x5d, 0x02, 0x0b, 0x81, 0xae, 0x0b,
                                      0x25, 0x47, 0x41, 0x54, 0xec, 0x89, 0xaa};

static uint8_t pubkey_x_secp224r1[] = {
    0x4c, 0x74, 0x1e, 0x4d, 0x20, 0x10, 0x36, 0x70, 0xb7, 0x16,
    0x1a, 0xe7, 0x22, 0x71, 0x08, 0x21, 0x55, 0x83, 0x84, 0x18,
    0x08, 0x43, 0x35, 0x33, 0x8a, 0xc3, 0x8f, 0xa4};

static uint8_t pubkey_y_secp224r1[] = {
    0xdb, 0x79, 0x19, 0x15, 0x1a, 0xc2, 0x85, 0x87, 0xb7, 0x2b,
    0xad, 0x7a, 0xb1, 0x80, 0xec, 0x8e, 0x95, 0xab, 0x9e, 0x2c,
    0x8d, 0x81, 0xd9, 0xb9, 0xd7, 0xe2, 0xe3, 0x83};

struct crypto_testsuite_ecpm_params ecpm_param_secp224r1 = {
    .gen_x =
        {
            .data = gen_x_secp224r1,
            .length = sizeof(gen_x_secp224r1),
        },
    .gen_y =
        {
            .data = gen_y_secp224r1,
            .length = sizeof(gen_y_secp224r1),
        },
    .privkey =
        {
            .data = privkey_secp224r1,
            .length = sizeof(privkey_secp224r1),
        },
    .pubkey_x =
        {
            .data = pubkey_x_secp224r1,
            .length = sizeof(pubkey_x_secp224r1),
        },
    .pubkey_y =
        {
            .data = pubkey_y_secp224r1,
            .length = sizeof(pubkey_y_secp224r1),
        },
    .curve = RTE_CRYPTO_EC_GROUP_SECP224R1};

/* SECP256R1 (P-256 NIST) test vectors */

static uint8_t gen_x_secp256r1[] = {
    0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47, 0xf8, 0xbc, 0xe6,
    0xe5, 0x63, 0xa4, 0x40, 0xf2, 0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb,
    0x33, 0xa0, 0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96};

static uint8_t gen_y_secp256r1[] = {
    0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b, 0x8e, 0xe7, 0xeb,
    0x4a, 0x7c, 0x0f, 0x9e, 0x16, 0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31,
    0x5e, 0xce, 0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5};

static uint8_t pubkey_x_secp256r1[] = {
    0x06, 0x27, 0x5d, 0x38, 0x7b, 0x8f, 0xcd, 0x29, 0x12, 0x02, 0xa5,
    0xad, 0x72, 0x35, 0x55, 0xd4, 0xe1, 0xca, 0xd6, 0x32, 0x91, 0xe7,
    0x8c, 0xb7, 0xf9, 0x85, 0xfe, 0xb5, 0xca, 0x61, 0xfd, 0xa7,
};

static uint8_t pubkey_y_secp256r1[] = {
    0x6d, 0x28, 0x17, 0x9c, 0x88, 0x2a, 0x06, 0x8c, 0x85, 0x52, 0x44,
    0xc1, 0x2f, 0xf6, 0x45, 0x80, 0x63, 0x1c, 0x52, 0xe4, 0xa5, 0xf8,
    0x21, 0x43, 0xec, 0xeb, 0xe2, 0xbb, 0x39, 0xff, 0x1f, 0xd8};

static uint8_t privkey_secp256r1[] = {
    0x36, 0xf7, 0xe3, 0x07, 0x84, 0xfa, 0xb5, 0x8d, 0x8d, 0x1d, 0x00,
    0x21, 0x8b, 0x59, 0xd1, 0x70, 0x14, 0x94, 0x86, 0x69, 0xec, 0xd3,
    0x99, 0xc8, 0x7a, 0xf0, 0x2d, 0x05, 0xbf, 0x16, 0xed, 0x36};

struct crypto_testsuite_ecpm_params ecpm_param_secp256r1 = {
    .gen_x =
        {
            .data = gen_x_secp256r1,
            .length = sizeof(gen_x_secp256r1),
        },
    .gen_y =
        {
            .data = gen_y_secp256r1,
            .length = sizeof(gen_y_secp256r1),
        },
    .privkey =
        {
            .data = privkey_secp256r1,
            .length = sizeof(privkey_secp256r1),
        },
    .pubkey_x =
        {
            .data = pubkey_x_secp256r1,
            .length = sizeof(pubkey_x_secp256r1),
        },
    .pubkey_y =
        {
            .data = pubkey_y_secp256r1,
            .length = sizeof(pubkey_y_secp256r1),
        },
    .curve = RTE_CRYPTO_EC_GROUP_SECP256R1};

/* SECP384R1 (P-384 NIST) test vectors */

static uint8_t gen_x_secp384r1[] = {
    0xAA, 0x87, 0xCA, 0x22, 0xBE, 0x8B, 0x05, 0x37, 0x8E, 0xB1, 0xC7, 0x1E,
    0xF3, 0x20, 0xAD, 0x74, 0x6E, 0x1D, 0x3B, 0x62, 0x8B, 0xA7, 0x9B, 0x98,
    0x59, 0xF7, 0x41, 0xE0, 0x82, 0x54, 0x2A, 0x38, 0x55, 0x02, 0xF2, 0x5D,
    0xBF, 0x55, 0x29, 0x6C, 0x3A, 0x54, 0x5E, 0x38, 0x72, 0x76, 0x0A, 0xB7};

static uint8_t gen_y_secp384r1[] = {
    0x36, 0x17, 0xDE, 0x4A, 0x96, 0x26, 0x2C, 0x6F, 0x5D, 0x9E, 0x98, 0xBF,
    0x92, 0x92, 0xDC, 0x29, 0xF8, 0xF4, 0x1D, 0xBD, 0x28, 0x9A, 0x14, 0x7C,
    0xE9, 0xDA, 0x31, 0x13, 0xB5, 0xF0, 0xB8, 0xC0, 0x0A, 0x60, 0xB1, 0xCE,
    0x1D, 0x7E, 0x81, 0x9D, 0x7A, 0x43, 0x1D, 0x7C, 0x90, 0xEA, 0x0E, 0x5F};

static uint8_t privkey_secp384r1[] = {
    0xc6, 0x02, 0xbc, 0x74, 0xa3, 0x45, 0x92, 0xc3, 0x11, 0xa6, 0x56, 0x96,
    0x61, 0xe0, 0x83, 0x2c, 0x84, 0xf7, 0x20, 0x72, 0x74, 0x67, 0x6c, 0xc4,
    0x2a, 0x89, 0xf0, 0x58, 0x16, 0x26, 0x30, 0x18, 0x4b, 0x52, 0xf0, 0xd9,
    0x9b, 0x85, 0x5a, 0x77, 0x83, 0xc9, 0x87, 0x47, 0x6d, 0x7f, 0x9e, 0x6b};

static uint8_t pubkey_x_secp384r1[] = {
    0x04, 0x00, 0x19, 0x3b, 0x21, 0xf0, 0x7c, 0xd0, 0x59, 0x82, 0x6e, 0x94,
    0x53, 0xd3, 0xe9, 0x6d, 0xd1, 0x45, 0x04, 0x1c, 0x97, 0xd4, 0x9f, 0xf6,
    0xb7, 0x04, 0x7f, 0x86, 0xbb, 0x0b, 0x04, 0x39, 0xe9, 0x09, 0x27, 0x4c,
    0xb9, 0xc2, 0x82, 0xbf, 0xab, 0x88, 0x67, 0x4c, 0x07, 0x65, 0xbc, 0x75};

static uint8_t pubkey_y_secp384r1[] = {
    0xf7, 0x0d, 0x89, 0xc5, 0x2a, 0xcb, 0xc7, 0x04, 0x68, 0xd2, 0xc5, 0xae,
    0x75, 0xc7, 0x6d, 0x7f, 0x69, 0xb7, 0x6a, 0xf6, 0x2d, 0xcf, 0x95, 0xe9,
    0x9e, 0xba, 0x5d, 0xd1, 0x1a, 0xdf, 0x8f, 0x42, 0xec, 0x9a, 0x42, 0x5b,
    0x0c, 0x5e, 0xc9, 0x8e, 0x2f, 0x23, 0x4a, 0x92, 0x6b, 0x82, 0xa1, 0x47};

struct crypto_testsuite_ecpm_params ecpm_param_secp384r1 = {
    .gen_x =
        {
            .data = gen_x_secp384r1,
            .length = sizeof(gen_x_secp384r1),
        },
    .gen_y =
        {
            .data = gen_y_secp384r1,
            .length = sizeof(gen_y_secp384r1),
        },
    .privkey =
        {
            .data = privkey_secp384r1,
            .length = sizeof(privkey_secp384r1),
        },
    .pubkey_x =
        {
            .data = pubkey_x_secp384r1,
            .length = sizeof(pubkey_x_secp384r1),
        },
    .pubkey_y =
        {
            .data = pubkey_y_secp384r1,
            .length = sizeof(pubkey_y_secp384r1),
        },
    .curve = RTE_CRYPTO_EC_GROUP_SECP384R1};

/* SECP521R1 (P-521 NIST) test vectors */

static uint8_t gen_x_secp521r1[] = {
    0xc6, 0x85, 0x8e, 0x06, 0xb7, 0x04, 0x04, 0xe9, 0xcd, 0x9e, 0x3e,
    0xcb, 0x66, 0x23, 0x95, 0xb4, 0x42, 0x9c, 0x64, 0x81, 0x39, 0x05,
    0x3f, 0xb5, 0x21, 0xf8, 0x28, 0xaf, 0x60, 0x6b, 0x4d, 0x3d, 0xba,
    0xa1, 0x4b, 0x5e, 0x77, 0xef, 0xe7, 0x59, 0x28, 0xfe, 0x1d, 0xc1,
    0x27, 0xa2, 0xff, 0xa8, 0xde, 0x33, 0x48, 0xb3, 0xc1, 0x85, 0x6a,
    0x42, 0x9b, 0xf9, 0x7e, 0x7e, 0x31, 0xc2, 0xe5, 0xbd, 0x66};

static uint8_t gen_y_secp521r1[] = {
    0x01, 0x18, 0x39, 0x29, 0x6a, 0x78, 0x9a, 0x3b, 0xc0, 0x04, 0x5c,
    0x8a, 0x5f, 0xb4, 0x2c, 0x7d, 0x1b, 0xd9, 0x98, 0xf5, 0x44, 0x49,
    0x57, 0x9b, 0x44, 0x68, 0x17, 0xaf, 0xbd, 0x17, 0x27, 0x3e, 0x66,
    0x2c, 0x97, 0xee, 0x72, 0x99, 0x5e, 0xf4, 0x26, 0x40, 0xc5, 0x50,
    0xb9, 0x01, 0x3f, 0xad, 0x07, 0x61, 0x35, 0x3c, 0x70, 0x86, 0xa2,
    0x72, 0xc2, 0x40, 0x88, 0xbe, 0x94, 0x76, 0x9f, 0xd1, 0x66, 0x50};

static uint8_t privkey_secp521r1[] = {
    0x01, 0xe8, 0xc0, 0x59, 0x96, 0xb8, 0x5e, 0x6f, 0x3f, 0x87, 0x57,
    0x12, 0xa0, 0x9c, 0x1b, 0x40, 0x67, 0x2b, 0x5e, 0x7a, 0x78, 0xd5,
    0x85, 0x2d, 0xe0, 0x15, 0x85, 0xc5, 0xfb, 0x99, 0x0b, 0xf3, 0x81,
    0x2c, 0x32, 0x45, 0x53, 0x4a, 0x71, 0x43, 0x89, 0xae, 0x90, 0x14,
    0xd6, 0x77, 0xa4, 0x49, 0xef, 0xd6, 0x58, 0x25, 0x4e, 0x61, 0x0d,
    0xa8, 0xe6, 0xca, 0xd3, 0x34, 0x14, 0xb9, 0xd3, 0x3e, 0x0d, 0x7a};

static uint8_t pubkey_x_secp521r1[] = {
    0x00, 0x7d, 0x04, 0x2c, 0xa1, 0x94, 0x08, 0x52, 0x4e, 0x68, 0xb9,
    0x81, 0xf1, 0x41, 0x93, 0x51, 0xe3, 0xb8, 0x47, 0x36, 0xc7, 0x7f,
    0xe5, 0x8f, 0xee, 0x7d, 0x11, 0x31, 0x7d, 0xf2, 0xe8, 0x50, 0xd9,
    0x60, 0xc7, 0xdd, 0x10, 0xd1, 0x0b, 0xa7, 0x14, 0xc8, 0xa6, 0x09,
    0xd1, 0x63, 0x50, 0x2b, 0x79, 0xd6, 0x82, 0xe8, 0xbb, 0xec, 0xd4,
    0xf5, 0x25, 0x91, 0xd2, 0x74, 0x85, 0x33, 0xe4, 0x5a, 0x86, 0x7a};

static uint8_t pubkey_y_secp521r1[] = {
    0x01, 0x97, 0xac, 0x64, 0x16, 0x11, 0x1c, 0xcf, 0x98, 0x7d, 0x29,
    0x04, 0x59, 0xeb, 0xc8, 0xad, 0x9e, 0xc5, 0x6e, 0x49, 0x05, 0x9c,
    0x99, 0x21, 0x55, 0x53, 0x9a, 0x36, 0xa6, 0x26, 0x63, 0x1f, 0x4a,
    0x2d, 0x89, 0x16, 0x4b, 0x98, 0x51, 0x54, 0xf2, 0xdd, 0xdc, 0x02,
    0x81, 0xee, 0x5b, 0x51, 0x78, 0x27, 0x1f, 0x3a, 0x76, 0xa0, 0x91,
    0x4c, 0x3f, 0xcd, 0x1f, 0x97, 0xbe, 0x8e, 0x83, 0x76, 0xef, 0xb3};

struct crypto_testsuite_ecpm_params ecpm_param_secp521r1 = {
    .gen_x =
        {
            .data = gen_x_secp521r1,
            .length = sizeof(gen_x_secp521r1),
        },
    .gen_y =
        {
            .data = gen_y_secp521r1,
            .length = sizeof(gen_y_secp521r1),
        },
    .privkey =
        {
            .data = privkey_secp521r1,
            .length = sizeof(privkey_secp521r1),
        },
    .pubkey_x =
        {
            .data = pubkey_x_secp521r1,
            .length = sizeof(pubkey_x_secp521r1),
        },
    .pubkey_y =
        {
            .data = pubkey_y_secp521r1,
            .length = sizeof(pubkey_y_secp521r1),
        },
    .curve = RTE_CRYPTO_EC_GROUP_SECP521R1};

#endif /* __TEST_CRYPTODEV_ECPM_TEST_VECTORS_H__ */