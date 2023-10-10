// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/dnnl/dnnl_utils.hpp"
#include "nodes/executors/executor.hpp"

namespace ov {
namespace intel_cpu {

// @todo generalize caching for dnnl backend
struct DnnlConvKey {
    // @todo shouldn't we have a key representing onednn specific data types only?
    const DnnlMemoryDescCPtr src;
    const DnnlMemoryDescCPtr wei;
    const DnnlMemoryDescCPtr bias;
    const DnnlMemoryDescCPtr dst;

    const dnnl::primitive_attr attr;

    size_t hash() const;
    bool operator==(const DnnlConvKey& rhs) const;
};

// @todo executor is not complete and covers only 1x1 fallback case for fullyconnected node
class DnnlConvolutionPrimitive {
public:
    DnnlConvolutionPrimitive(const DnnlConvKey& key,
                             const dnnl::engine& engine,
                             const std::vector<impl_desc_type>& implPriorities);

    void execute(const dnnl_primitive_args& primArgs) const;

    const DnnlMemoryDescPtr srcDesc() const {
        return m_srcDesc;
    }

    const DnnlMemoryDescPtr dstDesc() const {
        return m_dstDesc;
    }

    const DnnlMemoryDescPtr weightsDesc() const {
        return m_weiDesc;
    }

    const DnnlMemoryDescPtr scratchPadDesc() const {
        return m_scratchPadDesc;
    }

    impl_desc_type implType() const {
        return m_implType;
    }

    static std::shared_ptr<DnnlConvolutionPrimitive> create(const MemoryDescArgs& descs,
                                                            const ConvAttrs& attrs,
                                                            const ExecutorContext::CPtr context,
                                                            const DnnlShapeAgnosticDataPtr& shapeAgnosticData);

    static MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                                          const DnnlMemoryDescPtr dstWeightDesc,
                                          const MemoryCPtr weightsMem,
                                          const ExecutorContext::CPtr context) {
        const auto& weiDesc = srcWeightDesc->getDnnlDesc();
        const auto reshapedWeiDesc = weiDesc.reshape(dstWeightDesc->getDnnlDesc().get_dims());

        return utils::prepareWeightsMemory(DnnlExtensionUtils::makeDescriptor(reshapedWeiDesc),
                                           dstWeightDesc,
                                           weightsMem,
                                           context);
    }

private:
    dnnl::stream m_stream;
    dnnl::primitive_desc m_primDesc;
    impl_desc_type m_implType;
    DnnlMemoryDescPtr m_srcDesc;
    DnnlMemoryDescPtr m_weiDesc;
    DnnlMemoryDescPtr m_dstDesc;
    DnnlMemoryDescPtr m_scratchPadDesc;
    dnnl::primitive m_prim;
};

using DnnlConvExecutorPtr = std::shared_ptr<DnnlConvolutionPrimitive>;

}  // namespace intel_cpu
}  // namespace ov
