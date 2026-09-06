// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Shape-specialized, not model-name or configured TP-degree dispatch.
#define FI_GDN_IMPL_NAMESPACE flashinfer::sm70::gdn::h2560_q8_v24
#define FI_GDN_TORCH_NAMESPACE _C_flashinfer_gdn_sm70_h2560_q8_v24
#define FI_GDN_HIDDEN 2560
#define FI_GDN_N_BA 48
#define FI_GDN_QKV_DIM 5120
#define FI_GDN_H_Q 8
#define FI_GDN_HV 24
#define FI_GDN_D 128
#define FI_GDN_CONV_WIDTH 4
#define FI_GDN_CONV_STATE_LEN 3
#include "gdn_bridge.cuh"
