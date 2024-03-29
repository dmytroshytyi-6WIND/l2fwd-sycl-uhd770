/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2021 6WIND S.A.
 * Copyright 2021 Mellanox Technologies, Ltd
 */

#include "mlx5_tx.h"

/* Generate routines with Enhanced Multi-Packet Write support. */
MLX5_TXOFF_DECL(full_empw, MLX5_TXOFF_CONFIG_FULL | MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(none_empw, MLX5_TXOFF_CONFIG_NONE | MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(md_empw, MLX5_TXOFF_CONFIG_METADATA | MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(mt_empw, MLX5_TXOFF_CONFIG_MULTI | MLX5_TXOFF_CONFIG_TSO |
                             MLX5_TXOFF_CONFIG_METADATA |
                             MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(mtsc_empw, MLX5_TXOFF_CONFIG_MULTI | MLX5_TXOFF_CONFIG_TSO |
                               MLX5_TXOFF_CONFIG_SWP | MLX5_TXOFF_CONFIG_CSUM |
                               MLX5_TXOFF_CONFIG_METADATA |
                               MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(mti_empw, MLX5_TXOFF_CONFIG_MULTI | MLX5_TXOFF_CONFIG_TSO |
                              MLX5_TXOFF_CONFIG_INLINE |
                              MLX5_TXOFF_CONFIG_METADATA |
                              MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(mtv_empw, MLX5_TXOFF_CONFIG_MULTI | MLX5_TXOFF_CONFIG_TSO |
                              MLX5_TXOFF_CONFIG_VLAN |
                              MLX5_TXOFF_CONFIG_METADATA |
                              MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(mtiv_empw,
                MLX5_TXOFF_CONFIG_MULTI | MLX5_TXOFF_CONFIG_TSO |
                    MLX5_TXOFF_CONFIG_INLINE | MLX5_TXOFF_CONFIG_VLAN |
                    MLX5_TXOFF_CONFIG_METADATA | MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(sc_empw, MLX5_TXOFF_CONFIG_SWP | MLX5_TXOFF_CONFIG_CSUM |
                             MLX5_TXOFF_CONFIG_METADATA |
                             MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(sci_empw, MLX5_TXOFF_CONFIG_SWP | MLX5_TXOFF_CONFIG_CSUM |
                              MLX5_TXOFF_CONFIG_INLINE |
                              MLX5_TXOFF_CONFIG_METADATA |
                              MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(scv_empw, MLX5_TXOFF_CONFIG_SWP | MLX5_TXOFF_CONFIG_CSUM |
                              MLX5_TXOFF_CONFIG_VLAN |
                              MLX5_TXOFF_CONFIG_METADATA |
                              MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(sciv_empw,
                MLX5_TXOFF_CONFIG_SWP | MLX5_TXOFF_CONFIG_CSUM |
                    MLX5_TXOFF_CONFIG_INLINE | MLX5_TXOFF_CONFIG_VLAN |
                    MLX5_TXOFF_CONFIG_METADATA | MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(i_empw, MLX5_TXOFF_CONFIG_INLINE | MLX5_TXOFF_CONFIG_METADATA |
                            MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(v_empw, MLX5_TXOFF_CONFIG_VLAN | MLX5_TXOFF_CONFIG_METADATA |
                            MLX5_TXOFF_CONFIG_EMPW)

MLX5_TXOFF_DECL(iv_empw, MLX5_TXOFF_CONFIG_INLINE | MLX5_TXOFF_CONFIG_VLAN |
                             MLX5_TXOFF_CONFIG_METADATA |
                             MLX5_TXOFF_CONFIG_EMPW)
