/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2019-2021 Broadcom
 * All rights reserved.
 */

#ifndef _TF_DEVICE_P4_H_
#define _TF_DEVICE_P4_H_

#include "cfa_resource_types.h"
#include "tf_core.h"
#include "tf_global_cfg.h"
#include "tf_if_tbl.h"
#include "tf_rm.h"

extern struct tf_rm_element_cfg tf_tbl_p4[TF_DIR_MAX][TF_TBL_TYPE_MAX];

struct tf_rm_element_cfg tf_ident_p4[TF_IDENT_TYPE_MAX] = {
    [TF_IDENT_TYPE_L2_CTXT_HIGH] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                    CFA_RESOURCE_TYPE_P4_L2_CTXT_REMAP_HIGH, 0,
                                    0},
    [TF_IDENT_TYPE_L2_CTXT_LOW] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                   CFA_RESOURCE_TYPE_P4_L2_CTXT_REMAP_LOW, 0,
                                   0},
    [TF_IDENT_TYPE_PROF_FUNC] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                 CFA_RESOURCE_TYPE_P4_PROF_FUNC, 0, 0},
    [TF_IDENT_TYPE_WC_PROF] = {TF_RM_ELEM_CFG_HCAPI_BA,
                               CFA_RESOURCE_TYPE_P4_WC_TCAM_PROF_ID, 0, 0},
    [TF_IDENT_TYPE_EM_PROF] = {TF_RM_ELEM_CFG_HCAPI_BA,
                               CFA_RESOURCE_TYPE_P4_EM_PROF_ID, 0, 0},
};

struct tf_rm_element_cfg tf_tcam_p4[TF_TCAM_TBL_TYPE_MAX] = {
    [TF_TCAM_TBL_TYPE_L2_CTXT_TCAM_HIGH] =
        {TF_RM_ELEM_CFG_HCAPI_BA, CFA_RESOURCE_TYPE_P4_L2_CTXT_TCAM_HIGH, 0, 0},
    [TF_TCAM_TBL_TYPE_L2_CTXT_TCAM_LOW] =
        {TF_RM_ELEM_CFG_HCAPI_BA, CFA_RESOURCE_TYPE_P4_L2_CTXT_TCAM_LOW, 0, 0},
    [TF_TCAM_TBL_TYPE_PROF_TCAM] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                    CFA_RESOURCE_TYPE_P4_PROF_TCAM, 0, 0},
    [TF_TCAM_TBL_TYPE_WC_TCAM] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                  CFA_RESOURCE_TYPE_P4_WC_TCAM, 0, 0},
    [TF_TCAM_TBL_TYPE_SP_TCAM] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                  CFA_RESOURCE_TYPE_P4_SP_TCAM, 0, 0},
};

struct tf_rm_element_cfg tf_em_ext_p4[TF_EM_TBL_TYPE_MAX] = {
    [TF_EM_TBL_TYPE_TBL_SCOPE] = {TF_RM_ELEM_CFG_HCAPI_BA,
                                  CFA_RESOURCE_TYPE_P4_TBL_SCOPE, 0, 0},
};

struct tf_rm_element_cfg tf_em_int_p4[TF_EM_TBL_TYPE_MAX] = {
    [TF_EM_TBL_TYPE_EM_RECORD] = {TF_RM_ELEM_CFG_HCAPI,
                                  CFA_RESOURCE_TYPE_P4_EM_REC, 0, 0},
};

/* Note that hcapi_types from this table are from hcapi_cfa_p4.h
 * These are not CFA resource types because they are not allocated
 * CFA resources - they are identifiers for the interface tables
 * shared between the firmware and the host.  It may make sense to
 * move these types to cfa_resource_types.h.
 */
struct tf_if_tbl_cfg tf_if_tbl_p4[TF_IF_TBL_TYPE_MAX] = {
    [TF_IF_TBL_TYPE_PROF_SPIF_DFLT_L2_CTXT] =
        {TF_IF_TBL_CFG, CFA_P4_TBL_PROF_SPIF_DFLT_L2CTXT},
    [TF_IF_TBL_TYPE_PROF_PARIF_DFLT_ACT_REC_PTR] =
        {TF_IF_TBL_CFG, CFA_P4_TBL_PROF_PARIF_DFLT_ACT_REC_PTR},
    [TF_IF_TBL_TYPE_PROF_PARIF_ERR_ACT_REC_PTR] =
        {TF_IF_TBL_CFG, CFA_P4_TBL_PROF_PARIF_ERR_ACT_REC_PTR},
    [TF_IF_TBL_TYPE_LKUP_PARIF_DFLT_ACT_REC_PTR] =
        {TF_IF_TBL_CFG, CFA_P4_TBL_LKUP_PARIF_DFLT_ACT_REC_PTR},
};

struct tf_global_cfg_cfg tf_global_cfg_p4[TF_GLOBAL_CFG_TYPE_MAX] = {
    [TF_TUNNEL_ENCAP] = {TF_GLOBAL_CFG_CFG_HCAPI, TF_TUNNEL_ENCAP},
    [TF_ACTION_BLOCK] = {TF_GLOBAL_CFG_CFG_HCAPI, TF_ACTION_BLOCK},
};

const struct tf_hcapi_resource_map
    tf_hcapi_res_map_p4[CFA_RESOURCE_TYPE_P4_LAST + 1] = {
        [CFA_RESOURCE_TYPE_P4_L2_CTXT_REMAP_HIGH] =
            {TF_MODULE_TYPE_IDENTIFIER, 1 << TF_IDENT_TYPE_L2_CTXT_HIGH},
        [CFA_RESOURCE_TYPE_P4_L2_CTXT_REMAP_LOW] =
            {TF_MODULE_TYPE_IDENTIFIER, 1 << TF_IDENT_TYPE_L2_CTXT_LOW},
        [CFA_RESOURCE_TYPE_P4_PROF_FUNC] = {TF_MODULE_TYPE_IDENTIFIER,
                                            1 << TF_IDENT_TYPE_PROF_FUNC},
        [CFA_RESOURCE_TYPE_P4_WC_TCAM_PROF_ID] = {TF_MODULE_TYPE_IDENTIFIER,
                                                  1 << TF_IDENT_TYPE_WC_PROF},
        [CFA_RESOURCE_TYPE_P4_EM_PROF_ID] = {TF_MODULE_TYPE_IDENTIFIER,
                                             1 << TF_IDENT_TYPE_EM_PROF},
        [CFA_RESOURCE_TYPE_P4_L2_CTXT_TCAM_HIGH] =
            {TF_MODULE_TYPE_TCAM, 1 << TF_TCAM_TBL_TYPE_L2_CTXT_TCAM_HIGH},
        [CFA_RESOURCE_TYPE_P4_L2_CTXT_TCAM_LOW] =
            {TF_MODULE_TYPE_TCAM, 1 << TF_TCAM_TBL_TYPE_L2_CTXT_TCAM_LOW},
        [CFA_RESOURCE_TYPE_P4_PROF_TCAM] = {TF_MODULE_TYPE_TCAM,
                                            1 << TF_TCAM_TBL_TYPE_PROF_TCAM},
        [CFA_RESOURCE_TYPE_P4_WC_TCAM] = {TF_MODULE_TYPE_TCAM,
                                          1 << TF_TCAM_TBL_TYPE_WC_TCAM},
        [CFA_RESOURCE_TYPE_P4_SP_TCAM] = {TF_MODULE_TYPE_TCAM,
                                          1 << TF_TCAM_TBL_TYPE_SP_TCAM},
        [CFA_RESOURCE_TYPE_P4_NAT_IPV4] = {TF_MODULE_TYPE_TABLE,
                                           1 << TF_TBL_TYPE_ACT_MODIFY_IPV4},
        [CFA_RESOURCE_TYPE_P4_METER_PROF] = {TF_MODULE_TYPE_TABLE,
                                             1 << TF_TBL_TYPE_METER_PROF},
        [CFA_RESOURCE_TYPE_P4_METER] = {TF_MODULE_TYPE_TABLE,
                                        1 << TF_TBL_TYPE_METER_INST},
        [CFA_RESOURCE_TYPE_P4_MIRROR] = {TF_MODULE_TYPE_TABLE,
                                         1 << TF_TBL_TYPE_MIRROR_CONFIG},
        [CFA_RESOURCE_TYPE_P4_FULL_ACTION] = {TF_MODULE_TYPE_TABLE,
                                              1 << TF_TBL_TYPE_FULL_ACT_RECORD},
        [CFA_RESOURCE_TYPE_P4_MCG] = {TF_MODULE_TYPE_TABLE,
                                      1 << TF_TBL_TYPE_MCAST_GROUPS},
        [CFA_RESOURCE_TYPE_P4_ENCAP_8B] = {TF_MODULE_TYPE_TABLE,
                                           1 << TF_TBL_TYPE_ACT_ENCAP_8B},
        [CFA_RESOURCE_TYPE_P4_ENCAP_16B] = {TF_MODULE_TYPE_TABLE,
                                            1 << TF_TBL_TYPE_ACT_ENCAP_16B},
        [CFA_RESOURCE_TYPE_P4_ENCAP_64B] = {TF_MODULE_TYPE_TABLE,
                                            1 << TF_TBL_TYPE_ACT_ENCAP_64B},
        [CFA_RESOURCE_TYPE_P4_SP_MAC] = {TF_MODULE_TYPE_TABLE,
                                         1 << TF_TBL_TYPE_ACT_SP_SMAC},
        [CFA_RESOURCE_TYPE_P4_SP_MAC_IPV4] =
            {TF_MODULE_TYPE_TABLE, 1 << TF_TBL_TYPE_ACT_SP_SMAC_IPV4},
        [CFA_RESOURCE_TYPE_P4_SP_MAC_IPV6] =
            {TF_MODULE_TYPE_TABLE, 1 << TF_TBL_TYPE_ACT_SP_SMAC_IPV6},
        [CFA_RESOURCE_TYPE_P4_COUNTER_64B] = {TF_MODULE_TYPE_TABLE,
                                              1 << TF_TBL_TYPE_ACT_STATS_64},
        [CFA_RESOURCE_TYPE_P4_EM_REC] = {TF_MODULE_TYPE_EM,
                                         1 << TF_EM_TBL_TYPE_EM_RECORD},
        [CFA_RESOURCE_TYPE_P4_TBL_SCOPE] = {TF_MODULE_TYPE_EM,
                                            1 << TF_EM_TBL_TYPE_TBL_SCOPE},
};

#endif /* _TF_DEVICE_P4_H_ */
