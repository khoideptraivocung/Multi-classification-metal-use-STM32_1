/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-21T17:37:54+0700
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x9bb12521ebb97b4e6eaf6170ce923daf"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2026-01-21T17:37:54+0700"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_layer_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 333, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  gating_conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  gating_dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  gating_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gating_output_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gating_output_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_softmax_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_softmax_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  weighted_expert_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_softmax_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_softmax_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  weighted_expert_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1328, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_softmax_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_softmax_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  weighted_expert_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  moe_output_left_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  moe_output_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 7, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  gating_conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  gating_conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  gating_dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21248, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  gating_dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  gating_output_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 48, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  gating_output_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  gate_select_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21248, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_softmax_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_softmax_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21248, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_softmax_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_softmax_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 24, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 896, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 21248, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_softmax_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 112, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_softmax_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  gating_conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  gating_conv1_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv1_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  expert_3_conv2_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv1_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  expert_2_conv2_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv1_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 16, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 56, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  expert_1_conv2_scratch1_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 32, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv1_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_1_conv1_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv1_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 166), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_1_conv1_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv1_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 3), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &expert_1_conv1_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv1_scratch1, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 2), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_1_conv1_scratch1_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv1_weights, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 3, 8), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_1_conv1_weights_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_conv2_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 83), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_conv2_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_output0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 1328, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5312, 5312),
  1, &expert_1_conv2_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_scratch0, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 7), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_1_conv2_scratch0_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_scratch1, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 2), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_conv2_scratch1_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_conv2_weights, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 7, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &expert_1_conv2_weights_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_dense_dense_bias, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_dense_dense_bias_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_dense_dense_output, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_dense_dense_output_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_dense_dense_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1328, 16, 1, 1), AI_STRIDE_INIT(4, 4, 5312, 84992, 84992),
  1, &expert_1_dense_dense_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_dense_output, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_1_dense_output_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_softmax_dense_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_1_softmax_dense_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_softmax_dense_output, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_1_softmax_dense_output_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_softmax_dense_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 16, 7, 1, 1), AI_STRIDE_INIT(4, 4, 64, 448, 448),
  1, &expert_1_softmax_dense_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  expert_1_softmax_output, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_1_softmax_output_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv1_bias, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_2_conv1_bias_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv1_output, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 166), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_2_conv1_output_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv1_scratch0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 3), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &expert_2_conv1_scratch0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv1_scratch1, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 2), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_2_conv1_scratch1_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv1_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 3, 8), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_2_conv1_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_conv2_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 83), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_conv2_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_output0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 1328, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5312, 5312),
  1, &expert_2_conv2_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_scratch0, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 7), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_2_conv2_scratch0_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_scratch1, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 2), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_conv2_scratch1_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_conv2_weights, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 7, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &expert_2_conv2_weights_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_dense_dense_bias, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_dense_dense_bias_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_dense_dense_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_dense_dense_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_dense_dense_weights, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1328, 16, 1, 1), AI_STRIDE_INIT(4, 4, 5312, 84992, 84992),
  1, &expert_2_dense_dense_weights_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_dense_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_2_dense_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_softmax_dense_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_2_softmax_dense_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_softmax_dense_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_2_softmax_dense_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_softmax_dense_weights, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 16, 7, 1, 1), AI_STRIDE_INIT(4, 4, 64, 448, 448),
  1, &expert_2_softmax_dense_weights_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  expert_2_softmax_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_2_softmax_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv1_bias, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_3_conv1_bias_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv1_output, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 166), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_3_conv1_output_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv1_scratch0, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 3), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &expert_3_conv1_scratch0_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv1_scratch1, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 2), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_3_conv1_scratch1_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv1_weights, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 3, 8), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_3_conv1_weights_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_bias, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_conv2_bias_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 83), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_conv2_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_output0, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 1328, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5312, 5312),
  1, &expert_3_conv2_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 7), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &expert_3_conv2_scratch0_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_scratch1, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 2), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_conv2_scratch1_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_conv2_weights, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 8, 1, 7, 16), AI_STRIDE_INIT(4, 4, 32, 512, 512),
  1, &expert_3_conv2_weights_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_dense_dense_bias, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_dense_dense_bias_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_dense_dense_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_dense_dense_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_dense_dense_weights, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1328, 16, 1, 1), AI_STRIDE_INIT(4, 4, 5312, 84992, 84992),
  1, &expert_3_dense_dense_weights_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_dense_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &expert_3_dense_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_softmax_dense_bias, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_3_softmax_dense_bias_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_softmax_dense_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_3_softmax_dense_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_softmax_dense_weights, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 16, 7, 1, 1), AI_STRIDE_INIT(4, 4, 64, 448, 448),
  1, &expert_3_softmax_dense_weights_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  expert_3_softmax_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &expert_3_softmax_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_1_output, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gate_select_1_output_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_1_weights, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 3, 1, 1, 1), AI_STRIDE_INIT(4, 4, 12, 12, 12),
  1, &gate_select_1_weights_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_2_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gate_select_2_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_2_weights, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 3, 1, 1, 1), AI_STRIDE_INIT(4, 4, 12, 12, 12),
  1, &gate_select_2_weights_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_3_output, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gate_select_3_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  gate_select_3_weights, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 3, 1, 1, 1), AI_STRIDE_INIT(4, 4, 12, 12, 12),
  1, &gate_select_3_weights_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_bias, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &gating_conv1_bias_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 166), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &gating_conv1_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_output0, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 1328, 1, 1), AI_STRIDE_INIT(4, 4, 4, 5312, 5312),
  1, &gating_conv1_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 5), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &gating_conv1_scratch0_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_scratch1, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 2), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &gating_conv1_scratch1_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  gating_conv1_weights, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 5, 8), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &gating_conv1_weights_array, NULL)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  gating_dense_dense_bias, AI_STATIC,
  69, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gating_dense_dense_bias_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  gating_dense_dense_output, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gating_dense_dense_output_array, NULL)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  gating_dense_dense_weights, AI_STATIC,
  71, 0x0,
  AI_SHAPE_INIT(4, 1328, 16, 1, 1), AI_STRIDE_INIT(4, 4, 5312, 84992, 84992),
  1, &gating_dense_dense_weights_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  gating_dense_output, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &gating_dense_output_array, NULL)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  gating_output_dense_bias, AI_STATIC,
  73, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &gating_output_dense_bias_array, NULL)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  gating_output_dense_output, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &gating_output_dense_output_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  gating_output_dense_weights, AI_STATIC,
  75, 0x0,
  AI_SHAPE_INIT(4, 16, 3, 1, 1), AI_STRIDE_INIT(4, 4, 64, 192, 192),
  1, &gating_output_dense_weights_array, NULL)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  gating_output_output, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &gating_output_output_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  input_layer_output, AI_STATIC,
  77, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 333), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_layer_output_array, NULL)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  moe_output_left_0_output, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &moe_output_left_0_output_array, NULL)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  moe_output_output, AI_STATIC,
  79, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &moe_output_output_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  weighted_expert_1_output, AI_STATIC,
  80, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &weighted_expert_1_output_array, NULL)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  weighted_expert_2_output, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &weighted_expert_2_output_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  weighted_expert_3_output, AI_STATIC,
  82, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &weighted_expert_3_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  moe_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &moe_output_left_0_output, &weighted_expert_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &moe_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  moe_output_layer, 50,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &moe_output_chain,
  NULL, &moe_output_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  moe_output_left_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &weighted_expert_1_output, &weighted_expert_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &moe_output_left_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  moe_output_left_0_layer, 50,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &moe_output_left_0_chain,
  NULL, &moe_output_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  weighted_expert_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_1_softmax_output, &gate_select_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &weighted_expert_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  weighted_expert_1_layer, 47,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &weighted_expert_1_chain,
  NULL, &moe_output_left_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_softmax_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_softmax_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_1_softmax_layer, 41,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &expert_1_softmax_chain,
  NULL, &weighted_expert_1_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_softmax_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_1_softmax_dense_weights, &expert_1_softmax_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_1_softmax_dense_layer, 41,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_1_softmax_dense_chain,
  NULL, &expert_1_softmax_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_1_dense_layer, 33,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &expert_1_dense_chain,
  NULL, &expert_1_softmax_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_conv2_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_1_dense_dense_weights, &expert_1_dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_1_dense_dense_layer, 33,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_1_dense_dense_chain,
  NULL, &expert_1_dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_conv2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_conv2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_1_conv2_weights, &expert_1_conv2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_1_conv2_scratch0, &expert_1_conv2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_1_conv2_layer, 25,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_1_conv2_chain,
  NULL, &expert_1_dense_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 0, 3, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_1_conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_layer_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_1_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_1_conv1_weights, &expert_1_conv1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_1_conv1_scratch0, &expert_1_conv1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_1_conv1_layer, 10,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_1_conv1_chain,
  NULL, &expert_1_conv2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  weighted_expert_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_2_softmax_output, &gate_select_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &weighted_expert_2_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  weighted_expert_2_layer, 48,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &weighted_expert_2_chain,
  NULL, &expert_1_conv1_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_softmax_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_softmax_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_2_softmax_layer, 43,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &expert_2_softmax_chain,
  NULL, &weighted_expert_2_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_softmax_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_2_softmax_dense_weights, &expert_2_softmax_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_2_softmax_dense_layer, 43,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_2_softmax_dense_chain,
  NULL, &expert_2_softmax_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_2_dense_layer, 35,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &expert_2_dense_chain,
  NULL, &expert_2_softmax_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_conv2_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_2_dense_dense_weights, &expert_2_dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_2_dense_dense_layer, 35,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_2_dense_dense_chain,
  NULL, &expert_2_dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_conv2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_conv2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_2_conv2_weights, &expert_2_conv2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_2_conv2_scratch0, &expert_2_conv2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_2_conv2_layer, 27,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_2_conv2_chain,
  NULL, &expert_2_dense_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 0, 3, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_2_conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_layer_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_2_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_2_conv1_weights, &expert_2_conv1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_2_conv1_scratch0, &expert_2_conv1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_2_conv1_layer, 11,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_2_conv1_chain,
  NULL, &expert_2_conv2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  weighted_expert_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_3_softmax_output, &gate_select_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &weighted_expert_3_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  weighted_expert_3_layer, 49,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &weighted_expert_3_chain,
  NULL, &expert_2_conv1_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_softmax_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_softmax_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_3_softmax_layer, 45,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &expert_3_softmax_chain,
  NULL, &weighted_expert_3_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_softmax_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_softmax_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_3_softmax_dense_weights, &expert_3_softmax_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_3_softmax_dense_layer, 45,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_3_softmax_dense_chain,
  NULL, &expert_3_softmax_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_3_dense_layer, 36,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &expert_3_dense_chain,
  NULL, &expert_3_softmax_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_conv2_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_3_dense_dense_weights, &expert_3_dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  expert_3_dense_dense_layer, 36,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &expert_3_dense_dense_chain,
  NULL, &expert_3_dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_conv2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_conv2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_3_conv2_weights, &expert_3_conv2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_3_conv2_scratch0, &expert_3_conv2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_3_conv2_layer, 28,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_3_conv2_chain,
  NULL, &expert_3_dense_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 3, 0, 3, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  expert_3_conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_layer_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &expert_3_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &expert_3_conv1_weights, &expert_3_conv1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &expert_3_conv1_scratch0, &expert_3_conv1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  expert_3_conv1_layer, 12,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &expert_3_conv1_chain,
  NULL, &expert_3_conv2_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 0, 1, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gate_select_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_1_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gate_select_1_layer, 42,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gate_select_1_chain,
  NULL, &expert_3_conv1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gate_select_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_2_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gate_select_2_layer, 44,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gate_select_2_chain,
  NULL, &gate_select_1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gate_select_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_3_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gate_select_3_weights),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gate_select_3_layer, 46,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gate_select_3_chain,
  NULL, &gate_select_2_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gating_output_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gating_output_layer, 38,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm,
  &gating_output_chain,
  NULL, &gate_select_3_layer, AI_STATIC, 
  .nl_params = NULL, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gating_output_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_output_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gating_output_dense_weights, &gating_output_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gating_output_dense_layer, 38,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gating_output_dense_chain,
  NULL, &gating_output_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gating_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gating_dense_layer, 34,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &gating_dense_chain,
  NULL, &gating_output_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gating_dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_conv1_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gating_dense_dense_weights, &gating_dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  gating_dense_dense_layer, 34,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &gating_dense_dense_chain,
  NULL, &gating_dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gating_conv1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_layer_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gating_conv1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &gating_conv1_weights, &gating_conv1_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gating_conv1_scratch0, &gating_conv1_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  gating_conv1_layer, 26,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_if32of32wf32_nl_pool,
  &gating_conv1_chain,
  NULL, &gating_dense_dense_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = AI_HANDLE_PTR(forward_lite_nl_relu_if32of32), 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .pool_size = AI_SHAPE_2D_INIT(1, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_f32), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 353412, 1, 1),
    353412, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12320, 1, 1),
    12320, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_layer_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &moe_output_output),
  &gating_conv1_layer, 0x43a87ffc, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 353412, 1, 1),
      353412, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 12320, 1, 1),
      12320, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_layer_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &moe_output_output),
  &gating_conv1_layer, 0x43a87ffc, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_layer_output_array.data = AI_PTR(g_network_activations_map[0] + 228);
    input_layer_output_array.data_start = AI_PTR(g_network_activations_map[0] + 228);
    gating_conv1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1560);
    gating_conv1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1560);
    gating_conv1_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 164);
    gating_conv1_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 164);
    gating_conv1_output_array.data = AI_PTR(g_network_activations_map[0] + 1580);
    gating_conv1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1580);
    gating_dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 164);
    gating_dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 164);
    gating_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 1560);
    gating_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1560);
    gating_output_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 164);
    gating_output_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 164);
    gating_output_output_array.data = AI_PTR(g_network_activations_map[0] + 164);
    gating_output_output_array.data_start = AI_PTR(g_network_activations_map[0] + 164);
    gate_select_3_output_array.data = AI_PTR(g_network_activations_map[0] + 1560);
    gate_select_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1560);
    gate_select_2_output_array.data = AI_PTR(g_network_activations_map[0] + 1564);
    gate_select_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1564);
    gate_select_1_output_array.data = AI_PTR(g_network_activations_map[0] + 224);
    gate_select_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 224);
    expert_3_conv1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 6880);
    expert_3_conv1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 6880);
    expert_3_conv1_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 160);
    expert_3_conv1_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 160);
    expert_3_conv1_output_array.data = AI_PTR(g_network_activations_map[0] + 1568);
    expert_3_conv1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1568);
    expert_3_conv2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_conv2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_conv2_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 6880);
    expert_3_conv2_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 6880);
    expert_3_conv2_output_array.data = AI_PTR(g_network_activations_map[0] + 7008);
    expert_3_conv2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7008);
    expert_3_dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 64);
    expert_3_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 64);
    expert_3_softmax_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_softmax_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    expert_3_softmax_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    expert_3_softmax_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    weighted_expert_3_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    weighted_expert_3_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    expert_2_conv1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_conv1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_conv1_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 40);
    expert_2_conv1_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 40);
    expert_2_conv1_output_array.data = AI_PTR(g_network_activations_map[0] + 7008);
    expert_2_conv1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 7008);
    expert_2_conv2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 1568);
    expert_2_conv2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 1568);
    expert_2_conv2_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_conv2_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_conv2_output_array.data = AI_PTR(g_network_activations_map[0] + 6656);
    expert_2_conv2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6656);
    expert_2_dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 92);
    expert_2_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92);
    expert_2_softmax_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_softmax_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    expert_2_softmax_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    expert_2_softmax_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    weighted_expert_2_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    weighted_expert_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    expert_1_conv1_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_conv1_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_conv1_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 68);
    expert_1_conv1_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 68);
    expert_1_conv1_output_array.data = AI_PTR(g_network_activations_map[0] + 1560);
    expert_1_conv1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1560);
    expert_1_conv2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 228);
    expert_1_conv2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 228);
    expert_1_conv2_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_conv2_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_conv2_output_array.data = AI_PTR(g_network_activations_map[0] + 6872);
    expert_1_conv2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6872);
    expert_1_dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 120);
    expert_1_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 120);
    expert_1_softmax_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_softmax_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    expert_1_softmax_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    expert_1_softmax_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
    weighted_expert_1_output_array.data = AI_PTR(g_network_activations_map[0] + 56);
    weighted_expert_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 56);
    moe_output_left_0_output_array.data = AI_PTR(g_network_activations_map[0] + 84);
    moe_output_left_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 84);
    moe_output_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    moe_output_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    gating_conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    gating_conv1_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    gating_conv1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    gating_conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    gating_conv1_bias_array.data = AI_PTR(g_network_weights_map[0] + 160);
    gating_conv1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 160);
    gating_dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    gating_dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 192);
    gating_dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 192);
    gating_dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    gating_dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 85184);
    gating_dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 85184);
    gating_output_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    gating_output_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 85248);
    gating_output_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85248);
    gating_output_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    gating_output_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 85440);
    gating_output_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 85440);
    gate_select_3_weights_array.format |= AI_FMT_FLAG_CONST;
    gate_select_3_weights_array.data = AI_PTR(g_network_weights_map[0] + 85452);
    gate_select_3_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85452);
    gate_select_2_weights_array.format |= AI_FMT_FLAG_CONST;
    gate_select_2_weights_array.data = AI_PTR(g_network_weights_map[0] + 85464);
    gate_select_2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85464);
    gate_select_1_weights_array.format |= AI_FMT_FLAG_CONST;
    gate_select_1_weights_array.data = AI_PTR(g_network_weights_map[0] + 85476);
    gate_select_1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85476);
    expert_3_conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_3_conv1_weights_array.data = AI_PTR(g_network_weights_map[0] + 85488);
    expert_3_conv1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85488);
    expert_3_conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_3_conv1_bias_array.data = AI_PTR(g_network_weights_map[0] + 85584);
    expert_3_conv1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 85584);
    expert_3_conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_3_conv2_weights_array.data = AI_PTR(g_network_weights_map[0] + 85616);
    expert_3_conv2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 85616);
    expert_3_conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_3_conv2_bias_array.data = AI_PTR(g_network_weights_map[0] + 89200);
    expert_3_conv2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 89200);
    expert_3_dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_3_dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 89264);
    expert_3_dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 89264);
    expert_3_dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_3_dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 174256);
    expert_3_dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 174256);
    expert_3_softmax_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_3_softmax_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 174320);
    expert_3_softmax_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 174320);
    expert_3_softmax_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_3_softmax_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 174768);
    expert_3_softmax_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 174768);
    expert_2_conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_2_conv1_weights_array.data = AI_PTR(g_network_weights_map[0] + 174796);
    expert_2_conv1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 174796);
    expert_2_conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_2_conv1_bias_array.data = AI_PTR(g_network_weights_map[0] + 174892);
    expert_2_conv1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 174892);
    expert_2_conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_2_conv2_weights_array.data = AI_PTR(g_network_weights_map[0] + 174924);
    expert_2_conv2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 174924);
    expert_2_conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_2_conv2_bias_array.data = AI_PTR(g_network_weights_map[0] + 178508);
    expert_2_conv2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 178508);
    expert_2_dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_2_dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 178572);
    expert_2_dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 178572);
    expert_2_dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_2_dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 263564);
    expert_2_dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 263564);
    expert_2_softmax_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_2_softmax_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 263628);
    expert_2_softmax_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 263628);
    expert_2_softmax_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_2_softmax_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 264076);
    expert_2_softmax_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 264076);
    expert_1_conv1_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_1_conv1_weights_array.data = AI_PTR(g_network_weights_map[0] + 264104);
    expert_1_conv1_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 264104);
    expert_1_conv1_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_1_conv1_bias_array.data = AI_PTR(g_network_weights_map[0] + 264200);
    expert_1_conv1_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 264200);
    expert_1_conv2_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_1_conv2_weights_array.data = AI_PTR(g_network_weights_map[0] + 264232);
    expert_1_conv2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 264232);
    expert_1_conv2_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_1_conv2_bias_array.data = AI_PTR(g_network_weights_map[0] + 267816);
    expert_1_conv2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 267816);
    expert_1_dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_1_dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 267880);
    expert_1_dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 267880);
    expert_1_dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_1_dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 352872);
    expert_1_dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 352872);
    expert_1_softmax_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    expert_1_softmax_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 352936);
    expert_1_softmax_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 352936);
    expert_1_softmax_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    expert_1_softmax_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 353384);
    expert_1_softmax_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 353384);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 606732,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x43a87ffc,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 606732,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x43a87ffc,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

