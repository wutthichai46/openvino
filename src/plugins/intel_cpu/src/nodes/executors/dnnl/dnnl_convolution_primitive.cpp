// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_convolution_primitive.hpp"

#include <common/primitive_desc_iface.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include "dnnl_extension_utils.h"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

using namespace dnnl;
using namespace executor;

// @todo generalize caching for dnnl backend
size_t DnnlConvKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {src, wei, bias, dst}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash::combine(seed, get_attr_hash(*attr.get()));

    return seed;
}

bool DnnlConvKey::operator==(const DnnlConvKey& rhs) const {
    bool result = true;

    if (src != rhs.src) {
        result = result && src && rhs.src && src->getDnnlDesc() == rhs.src->getDnnlDesc();
    }
    if (wei != rhs.wei) {
        result = result && wei && rhs.wei && wei->getDnnlDesc() == rhs.wei->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        result = result && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (dst != rhs.dst) {
        result = result && dst && rhs.dst && dst->getDnnlDesc() == rhs.dst->getDnnlDesc();
    }

    result = result && *attr.get() == *rhs.attr.get();

    return result;
}

static primitive_desc createPrimitiveDesc(const dnnl::engine& engine,
                                          const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const dnnl::primitive_attr& attr,
                                          const std::vector<impl_desc_type>& implPriorities);

std::shared_ptr<DnnlConvolutionPrimitive> DnnlConvolutionPrimitive::create(
    const MemoryDescArgs& descs,
    const ConvAttrs& attrs,
    const ExecutorContext::CPtr context,
    const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.src[0]);
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.src[1]);
    const DnnlMemoryDescPtr biaDesc = descs.src[2]->getCurrentMemSize() != 0
                                          ? MemoryDescUtils::convertToDnnlMemoryDesc(descs.src[2])
                                          : DnnlExtensionUtils::makeDescriptor(dnnl::memory::desc{});
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.dst[0]);

    const DnnlConvKey dnnlConvKey{srcDesc, weiDesc, biaDesc, dstDesc, shapeAgnosticData->primAttrs.attr};

    auto builder = [&context](const DnnlConvKey& dnnlKey) {
        return std::make_shared<DnnlConvolutionPrimitive>(dnnlKey, context->getEngine(), context->getImplPriorities());
    };

    auto executorCache = context->getRuntimeCache();
    const auto result = executorCache->getOrCreate(dnnlConvKey, builder);
    const auto& executor = result.first;
    assert(executor);

    return executor;
}

DnnlConvolutionPrimitive::DnnlConvolutionPrimitive(const DnnlConvKey& key,
                                                   const dnnl::engine& engine,
                                                   const std::vector<impl_desc_type>& implPriorities)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(engine,
                                     key.src->getDnnlDesc(),
                                     key.wei->getDnnlDesc(),
                                     key.bias->getDnnlDesc(),
                                     key.dst->getDnnlDesc(),
                                     key.attr,
                                     implPriorities)),
      m_implType(parse_impl_name(m_primDesc.impl_info_str())),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)) {}

// make a fake shape: N, C, W
static dnnl::memory::dims normalizeDims(const dnnl::memory::dims& dims) {
    assert(one_of(static_cast<int>(dims.size()), 2, 3));

    if (dims.size() == 3) {
        return {dims[0], dims[2], dims[1]};
    }

    return {dnnl::memory::dim{1}, dims[1], dims[0]};
}

static dnnl::convolution_forward::primitive_desc createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                                                                          const dnnl::memory::desc& weightDesc,
                                                                          const dnnl::memory::desc& biasDesc,
                                                                          const dnnl::memory::desc& outputDesc,
                                                                          const dnnl::primitive_attr& attr,
                                                                          const dnnl::engine& engine) {
    const auto normalizedInDims = normalizeDims(inputDesc.get_dims());
    const auto convInDesc = dnnl::memory::desc(normalizedInDims, inputDesc.get_data_type(), memory::format_tag::nwc);
    const auto normalizedOutDims = normalizeDims(outputDesc.get_dims());
    const auto convOutDesc = dnnl::memory::desc(normalizedOutDims, outputDesc.get_data_type(), memory::format_tag::nwc);

    // @todo create general mapping from node configuration to backend configuration
    static const std::map<memory::data_type, memory::data_type> weightsTypeByInputType{
  // input data type        weights data type
        {memory::data_type::f32,  memory::data_type::f32 },
        {memory::data_type::f16,  memory::data_type::f16 },
        {memory::data_type::bf16, memory::data_type::bf16},
        {memory::data_type::u8,   memory::data_type::s8  },
        {memory::data_type::s8,   memory::data_type::s8  },
    };

    // make a fake shape: OC, IC, 1
    const auto& weightDims = weightDesc.get_dims();
    const dnnl::memory::dims normalizedWeightDims{static_cast<dnnl::memory::dim>(weightDims[0]),
                                                  static_cast<dnnl::memory::dim>(weightDims[1]),
                                                  dnnl::memory::dim{1}};
    const auto weightDataType = weightsTypeByInputType.at(inputDesc.get_data_type());
    const auto convWeightDescAny =
        dnnl::memory::desc(normalizedWeightDims, weightDataType, dnnl::memory::format_tag::any);

    return dnnl::convolution_forward::primitive_desc(engine,
                                                     prop_kind::forward_inference,
                                                     dnnl::algorithm::convolution_direct,
                                                     convInDesc,
                                                     convWeightDescAny,
                                                     biasDesc,
                                                     convOutDesc,
                                                     dnnl::memory::dims{1},  // stride
                                                     dnnl::memory::dims{0},  // dilation
                                                     dnnl::memory::dims{0},  // paddingL
                                                     dnnl::memory::dims{0},  // paddingR
                                                     attr);
}

primitive_desc createPrimitiveDesc(const dnnl::engine& engine,
                                   const dnnl::memory::desc& inputDesc,
                                   const dnnl::memory::desc& weightDesc,
                                   const dnnl::memory::desc& biasDesc,
                                   const dnnl::memory::desc& outputDesc,
                                   const dnnl::primitive_attr& attr,
                                   const std::vector<impl_desc_type>& implPriorities) {
    auto prim_desc = createDescriptorInternal(inputDesc, weightDesc, biasDesc, outputDesc, attr, engine);
    auto first_desc = dnnl::convolution_forward::primitive_desc(prim_desc.get());

    for (auto preferredImplType : implPriorities) {
        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, preferredImplType);

        if (found)
            return std::move(prim_desc);
    }

    return std::move(first_desc);
}

void DnnlConvolutionPrimitive::execute(const dnnl_primitive_args& primArgs) const {
    m_prim.execute(m_stream, primArgs);
}

}  // namespace intel_cpu
}  // namespace ov
