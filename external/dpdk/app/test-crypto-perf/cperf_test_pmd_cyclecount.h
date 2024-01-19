/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Intel Corporation
 */

#ifndef _CPERF_TEST_PMD_CYCLECOUNT_H_
#define _CPERF_TEST_PMD_CYCLECOUNT_H_

#include <stdint.h>

#include <rte_mbuf.h>

#include "cperf.h"
#include "cperf_ops.h"
#include "cperf_options.h"
#include "cperf_test_vectors.h"

void *cperf_pmd_cyclecount_test_constructor(
    struct rte_mempool *sess_mp, struct rte_mempool *sess_priv_mp,
    uint8_t dev_id, uint16_t qp_id, const struct cperf_options *options,
    const struct cperf_test_vector *test_vector,
    const struct cperf_op_fns *ops_fn);

int cperf_pmd_cyclecount_test_runner(void *test_ctx);

void cperf_pmd_cyclecount_test_destructor(void *test_ctx);

#endif /* _CPERF_TEST_PMD_CYCLECOUNT_H_ */
