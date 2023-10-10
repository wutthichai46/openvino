// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/fullyconnected_implementations.h"

#include <memory>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "debug_messages.hpp"
#include "implementation_utils.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_fullyconnected.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mlas/mlas_gemm.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "nodes/executors/precision_translation.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpp/maybe_unused.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

using namespace ov::element;
using namespace TypeMaskAlias;
using namespace executor;

struct LayoutConfig {
    std::vector<LayoutType> src;
    LayoutType dst;
};

// clang-format off
static const LayoutConfig dnnlFCLayoutConfig {
    {LayoutType::ncsp, LayoutType::ncsp, LayoutType::ncsp}, LayoutType::ncsp
};

static const TypeMapping dnnlFCTypeMapping {
    // {{src,   wei, bia} dst} ->                          PT<src, wei, bias, dst>
    {{{_bf16, _bf16, _any}, _bf16 | _f32},                 pt<bypass, bypass, out, out>{}},
    {{{_f16, _f16, _any}, _f16 | _f32},                    pt<bypass, bypass, out, out>{}},
    // integer precision outputs are not supported for float precision inputs
    {{{_f32 | _bf16 | _f16, _any, _any}, _i8 | _u8},       pt<bypass, bypass, in<0>, in<0>>{}},
    // compresses float weights which do not match input data precision
    {{{_f32, _half_float, _any}, _any | _any},             pt<bypass, bypass, in<0>, in<0>>{}},
    {{{_bf16, _f16, _any}, _any | _any},                   pt<bypass, bypass, in<0>, in<0>>{}},
    {{{_f16, _bf16, _any}, _any | _any},                   pt<bypass, bypass, in<0>, in<0>>{}},
    // quantization configuration
    {{{_u8 | _i8, _i8, _any}, _any},                       pt<bypass, bypass, bypass, out>{}},
    // compresses int weights
    {{{_f32 | _bf16, _u8 | _nf4 | _u4 | _i4, _any}, _any}, pt<bypass, bypass, in<0>, in<0>>{}},
    // fallback to _f32 if weights are not _i8
    {{{_u8 | _i8, !_i8, _any}, _any},                      everyone<just<f32>>{}},
    // @todo should we fallback to FPXX instead of _f32?
    {{{_any, _any, _any}, _any},                           everyone<just<f32>>{}},
};

static const TypeMapping dnnlConvolutionTypeMapping {
    // {{src,   wei, bia} dst} ->                    PT<src, wei, bias, dst>
    {{{_bf16, _bf16, _any}, _bf16 | _f32},           pt<bypass, bypass, out, out>{}},
    {{{_f16, _f16, _any}, _f16 | _f32},              pt<bypass, bypass, out, out>{}},
    // integer precision outputs are not supported for float precision inputs
    {{{_f32 | _bf16 | _f16, _any, _any}, _i8 | _u8}, pt<bypass, bypass, in<0>, in<0>>{}},
    // compresses float weights which do not match input data precision
    {{{_f32, _half_float, _any}, _any | _any},       pt<bypass, bypass, in<0>, in<0>>{}},
    {{{_bf16, _f16, _any}, _any | _any},             pt<bypass, bypass, in<0>, in<0>>{}},
    {{{_f16, _bf16, _any}, _any | _any},             pt<bypass, bypass, in<0>, in<0>>{}},
    // quantization configuration
    {{{_u8 | _i8, _i8, _any}, _any},                 pt<bypass, bypass, out, out>{}},
    // @todo should we fallback to _fxx instead of _f32 (currenly legacy logic is replicated)
    {{{_any, _any, _any}, _any},                     everyone<just<f32>>{}},
};
// clang-format on

static bool fullyMatchConfiguration(const MemoryDescArgs& currentDescriptors,
                                    const InOutTypes& typeConfig,
                                    const LayoutConfig& layoutConfig) {
    const auto& srcTypeConfig = typeConfig.first;
    const auto& srcLayoutConfig = layoutConfig.src;
    assert(currentDescriptors.src.size() <= srcTypeConfig.size());
    assert(currentDescriptors.src.size() <= srcLayoutConfig.size());

    for (size_t i = 0; i < currentDescriptors.src.size(); i++) {
        const auto& desc = currentDescriptors.src[i];
        if ((!one_of(desc->getPrecision(), srcTypeConfig[i], ov::element::undefined)) ||
            !desc->hasLayoutType(srcLayoutConfig[i]))
            return false;
    }

    const auto& dstTypeConfig = typeConfig.second;
    const auto& dstLayoutConfig = layoutConfig.dst;

    const auto& dstDesc = currentDescriptors.dst[0];
    if (!(one_of(dstDesc->getPrecision(), dstTypeConfig, ov::element::undefined)) ||
        !dstDesc->hasLayoutType(dstLayoutConfig))
        return false;

    return true;
}

static MemoryDescArgs createOptimalDescriptors(const MemoryDescArgs& currentDescriptors,
                                               const InOutTypes& typeConfig,
                                               const LayoutConfig& layoutConfig) {
    const auto& srcTypesConfig = typeConfig.first;
    const auto dstTypeConfig = typeConfig.second;

    MemoryDescArgs descs;

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    for (size_t i = 0; i < currentDescriptors.src.size(); i++) {
        const auto srcPrecision = currentDescriptors.src[i]->getPrecision();
        if (srcPrecision == ov::element::undefined || srcPrecision == srcTypesConfig[i]) {
            descs.src.push_back(currentDescriptors.src[i]);
            continue;
        }
        const auto newType = srcTypesConfig[i];
        descs.src.push_back(
            creatorsMap.at(layoutConfig.src[i])->createSharedDesc(newType, currentDescriptors.src[i]->getShape()));
    }

    const auto dstPrecision = currentDescriptors.dst[0]->getPrecision();
    if (dstPrecision == ov::element::undefined || dstPrecision == dstTypeConfig) {
        descs.dst.push_back(currentDescriptors.dst[0]);
    } else {
        const auto newType = dstTypeConfig;
        descs.dst.push_back(
            creatorsMap.at(layoutConfig.dst)->createSharedDesc(newType, currentDescriptors.dst[0]->getShape()));
    }

    return descs;
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noWeightsDecompression(const FCConfig& key) {
    return !DnnlFCPrimitive::useWeightsDecompressionImpl(srcType(key), weiType(key));
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noSparseDecompression(const FCConfig& key) {
    return !(key.attrs.sparseWeights);
}

OV_CPU_MAYBE_UNUSED_FUNCTION static inline bool noPostOps(const FCConfig& key) {
    return key.postOps.empty();
}

template <typename Attrs>
std::pair<bool, executor::Config<Attrs>> isFullyCompliantCommon(const executor::Config<Attrs>& key,
                                                                const TypeMapping& typeMapping,
                                                                const LayoutConfig& layoutConfig) {
    const auto typeConfig = getTypeConfiguration(typeMapping, key.descs);

    if (fullyMatchConfiguration(key.descs, typeConfig, layoutConfig)) {
        return std::make_pair(true, key);
    }

    const auto optimalDescriptors = createOptimalDescriptors(key.descs, typeConfig, layoutConfig);

    return std::make_pair(false, FCConfig{optimalDescriptors, key.attrs, key.postOps});
}

static const std::vector<ExecutorImplementation<FCAttrs>> fullyconnectedImplementations {
    OV_CPU_INSTANCE_X64(
        // @todo executor type should not be a part of the name (can be appended if necessary)
        "fullyconnected_mlas",  // name
        ExecutorType::Mlas,     // type
        OperationType::MatMul,
        ShapeTolerance::Agnostic,
        // isSupported
        [](const FCConfig& key) -> bool {
            // @todo probably there is no need of having implementation name in the debug message
            // since it can be distinguished from the context of other logs anyway.
            VERIFY(noPostOps(key), "fullyconnected_mlas", UNSUPPORTED_POST_OPS);
            VERIFY(noSparseDecompression(key), "fullyconnected_mlas", UNSUPPORTED_SPARSE_WEIGHTS);
            VERIFY(noWeightsDecompression(key), "fullyconnected_mlas", UNSUPPORTED_WEIGHTS_DECOMPRESSION);
            VERIFY(everyone_is(f32, srcType(key), weiType(key), dstType(key)),
                   "fullyconnected_mlas",
                   UNSUPPORTED_SRC_PRECISIONS);

            return MlasGemmExecutor::isSupported(key);
        },
        // isFullyCompliant
        [](const FCConfig& key) {
            return std::make_pair(true, key);
        },
        // isShapeSuitable
        [](const FCConfig& key) -> bool {
            return true;
        },
        // create
        [](const FCConfig& key, const MemoryArgs& memory, const ExecutorContext::CPtr context) {
            return std::make_shared<MlasGemmExecutor>(key, memory, context);
        })
    OV_CPU_INSTANCE_X64(
        "convolution_1x1_dnnl",  // name
        ExecutorType::Dnnl,      // type
        OperationType::Convolution,
        ShapeTolerance::Dependant,
        // isSupported
        [](const FCConfig& key) -> bool {
            VERIFY(noSparseDecompression(key), "convolution_1x1_dnnl", UNSUPPORTED_SPARSE_WEIGHTS);
            VERIFY(noWeightsDecompression(key), "convolution_1x1_dnnl", UNSUPPORTED_WEIGHTS_DECOMPRESSION);
            auto getOffset0 = [](const MemoryDescPtr& desc) {
                DnnlMemoryDescCPtr dnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(desc);
                dnnl::impl::memory_desc_wrapper wrapped(dnnlDesc->getDnnlDesc().get());
                return wrapped.offset0();
            };

            VERIFY(dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core), UNSUPPORTED_ISA);
            VERIFY(srcType(key) == ov::element::f32, UNSUPPORTED_SRC_PRECISIONS);
            // disable rank=4:
            // if layout is nhwc:
            //   A matrix: N * IC * H * W --> N * (IC*H*W), the M, N', K of matrix multiply will be:
            //   M = 1, K = (IC*H*W), when M = 1 it should not be efficient since acts as a vector multiply
            // if layout is nchw/nChw16c: brg1x1 not support. Although jit supports, it should have similar
            //   problems with the above.
            VERIFY(one_of(srcRank(key), 2u, 3u), UNSUPPORTED_SRC_RANK);
            VERIFY(weiRank(key) == 2, UNSUPPORTED_WEI_RANK);
            // brg convolution does not support stride
            VERIFY(getOffset0(key.descs.dst[0]) == 0, UNSUPPORTED_DST_STRIDES);
            return true;
        },
        // isFullyCompliant
        [](const FCConfig& key) {
            // @todo use dnnlConvolutionLayoutConfig after one is implemented
            return isFullyCompliantCommon(key, dnnlConvolutionTypeMapping, dnnlFCLayoutConfig);
        },
        // isShapeSuitable
        [](const FCConfig& key) -> bool {
            const auto inRank = srcRank(key);
            const auto& inDims = srcDims(key);
            const auto& weightDims = weiDims(key);
            // for original inner product semantics:
            //  when input is 2D tensor -> M in oneDNN will map to widthInConv
            //  when input is 3D tensor -> M in oneDNN will map to widthInConv*minibatch
            // currently nwc mapping in brg::
            //  when input is 2D tensor -> widthInConv will map to 'w', 'n' will be 1
            //  when input is 3D tensor -> widthInConv will map to 'w', 'n' will be minibatch
            Dim widthInConv = inDims[inRank - 2];
            Dim K = inDims[inRank - 1];
            Dim N = weightDims[0];

            const auto& weightsSize = weiMemSize(key);
            // Disable Conv1x1 when weight size >= 16M to avoid different weight layout when having different input
            // activation shapes. As a consuquence, peak memory consumption in LLM can be decreased.
            VERIFY(weightsSize < (16 * 1 << 20), " weights size is to big");
            VERIFY(widthInConv >= 2 && widthInConv <= 3136 && K >= 96 && K <= 4096 && N >= 96 && N <= K * 4,
                   HEURISTICS_MISMATCH);

            return true;
        },
        // create
        [](const FCConfig& key, const MemoryArgs& memory, ExecutorContext::CPtr context) {
            struct ConvolutionInstantiator {
                std::shared_ptr<DnnlConvolutionPrimitive> operator()(
                    const MemoryDescArgs& descs,
                    const FCAttrs& attrs,
                    const ExecutorContext::CPtr context,
                    std::shared_ptr<DnnlShapeAgnosticData> shareAgnosticData) const {
                    ConvAttrs convAttrs{attrs.withBias};
                    return DefaultInstantiator<DnnlConvolutionPrimitive, ConvAttrs, DnnlShapeAgnosticData>{}(
                        descs,
                        convAttrs,
                        context,
                        shareAgnosticData);
                }
            };

            return std::make_shared<
                DnnlFCExecutor<DnnlConvolutionPrimitive, FCAttrs, DnnlShapeAgnosticData, ConvolutionInstantiator>>(
                key,
                memory,
                context,
                false);
        })
    OV_CPU_INSTANCE_DNNL(
        // @todo executor type should not be a part of the name (can be appended if necessary)
        "fullyconnected_dnnl",  // name
        ExecutorType::Dnnl,     // type
        OperationType::FullyConnected,
        ShapeTolerance::Dependant,
        // isSupported
        [](const FCConfig& key) -> bool {
            return true;
        },
        // isFullyCompliant
        [](const FCConfig& key) {
            return isFullyCompliantCommon(key, dnnlFCTypeMapping, dnnlFCLayoutConfig);
        },
        // isShapeSuitable
        [](const FCConfig& key) -> bool {
            return true;
        },
        // create
        [](const FCConfig& key, const MemoryArgs& memory, ExecutorContext::CPtr context) {
            return std::make_shared<DnnlFCExecutor<DnnlFCPrimitive, FCAttrs, DnnlShapeAgnosticData>>(key,
                                                                                                     memory,
                                                                                                     context,
                                                                                                     false);
        })
};

template <>
const std::vector<ExecutorImplementation<FCAttrs>>& getImplementations() {
    return fullyconnectedImplementations;
}
}  // namespace intel_cpu
}  // namespace ov
