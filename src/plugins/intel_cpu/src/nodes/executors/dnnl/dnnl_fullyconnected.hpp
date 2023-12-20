// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_fullyconnected_primitive.hpp"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/memory_arguments.hpp"

namespace ov {
namespace intel_cpu {

template <typename ExecutorT, typename Attrs, typename ShapeAgnosticData>
class DefaultInstantiator {
public:
    std::shared_ptr<ExecutorT> operator()(const MemoryDescArgs& descs,
                                          const Attrs& attrs,
                                          const ExecutorContext::CPtr context,
                                          const std::shared_ptr<ShapeAgnosticData> shapeAgnosticData) {
        return ExecutorT::create(descs, attrs, context, shapeAgnosticData);
    }
};

template <typename Primitive,
          typename Attrs,
          typename ShapeAgnosticData,
          typename Instantiator = DefaultInstantiator<Primitive, Attrs, ShapeAgnosticData>>
class DnnlFCExecutor : public Executor {
public:
    using PrimitivePtr = std::shared_ptr<Primitive>;
    DnnlFCExecutor(const executor::Config<Attrs>& key,
                   const MemoryArgs& memory,
                   const ExecutorContext::CPtr context,
                   const bool cacheWeights)
        : m_attrs(key.attrs),
          m_context(context),
          m_shapeAgnosticData(DnnlFCPrimitive::createShapeAgnosticData(key, memory, context, cacheWeights)),
          m_primArgs(m_shapeAgnosticData->primAttrs.dnnlArgs) {}
    void update(const MemoryDescArgs& descs, const MemoryArgs& memory) override {
        const auto primitive = createPrimitive(descs);
        updateMemory(m_primitive, primitive, descs, memory);
        m_primitive = primitive;
    }

    void execute(const MemoryArgs& memory) override {
        if (resetSrcMemoryDataHandle)
            m_primArgs[DNNL_ARG_SRC].set_data_handle(memory.at(ARG_SRC)->getData());
        if (resetDstMemoryDataHandle)
            m_primArgs[DNNL_ARG_DST].set_data_handle(memory.at(ARG_DST)->getData());
        m_primitive->execute(m_primArgs);
    }

    impl_desc_type implType() const override {
        return m_primitive->implType();
    }

private:
    void updateSrcMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr memory) {
        const auto& primMemDesc = primitive->srcDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_SRC] = memory->getPrimitive();
        } else {
            resetSrcMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_SRC] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), memory->getData());
        }
    }

    void updateDstMemory(const DnnlMemoryDescPtr& memDesc, const PrimitivePtr primitive, const MemoryPtr memory) {
        const auto& primMemDesc = primitive->dstDesc();
        if (memDesc->isCompatible(*primMemDesc)) {
            m_primArgs[DNNL_ARG_DST] = memory->getPrimitive();
        } else {
            resetDstMemoryDataHandle = true;
            // create 2D memory without underlying buffer and reset to the actual memory in scope of 'execute' call
            m_primArgs[DNNL_ARG_DST] =
                dnnl::memory(primMemDesc->getDnnlDesc(), m_context->getEngine(), memory->getData());
        }
    }

    void updateWeightsMemory(const DnnlMemoryDescPtr originalMemDesc,
                             const PrimitivePtr currentPrimitive,
                             const PrimitivePtr newPrimitive,
                             const MemoryPtr memory) {
        const auto newPrimMemDesc = newPrimitive->weightsDesc();
        if (currentPrimitive && currentPrimitive->weightsDesc()->isCompatible(*newPrimMemDesc))
            return;

        const auto weiMemory = newPrimitive->prepareWeightsMemory(originalMemDesc, newPrimMemDesc, memory, m_context);
        m_primArgs[DNNL_ARG_WEIGHTS] = weiMemory->getPrimitive();
    }

    void updateBiasMemory(const MemoryPtr memory) {
        m_primArgs[DNNL_ARG_BIAS] = memory->getPrimitive();
    }

    void updateScratchPadMem(const PrimitivePtr currentPrimitive, const PrimitivePtr newPrimitive) {
        const auto newPrimMemDesc = newPrimitive->scratchPadDesc();
        // @todo should we compare dnnl::memory::desc directly to avoid any overhead?
        if (currentPrimitive && currentPrimitive->scratchPadDesc()->isCompatible(*newPrimMemDesc))
            return;

        m_scratchPadMemory = m_context->getScratchPad()->createScratchPadMem(newPrimMemDesc);
        m_primArgs[DNNL_ARG_SCRATCHPAD] = m_scratchPadMemory->getPrimitive();
    }

    void updateMemory(const PrimitivePtr currentPrimitive,
                      const PrimitivePtr newPrimitive,
                      const MemoryDescArgs& descs,
                      const MemoryArgs& memory) {
        const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.src[0]);
        const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.src[1]);
        const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(descs.dst[0]);

        updateSrcMemory(srcDesc, newPrimitive, memory.at(ARG_SRC));
        updateDstMemory(dstDesc, newPrimitive, memory.at(ARG_DST));
        updateWeightsMemory(weiDesc, currentPrimitive, newPrimitive, memory.at(ARG_WEI));
        updateBiasMemory(memory.at(ARG_BIAS));
        updateScratchPadMem(currentPrimitive, newPrimitive);
    }

    PrimitivePtr createPrimitive(const MemoryDescArgs& descs) {
        return Instantiator{}(descs, m_attrs, m_context, m_shapeAgnosticData);
    }

    const Attrs& m_attrs;
    const ExecutorContext::CPtr m_context;
    const std::shared_ptr<ShapeAgnosticData> m_shapeAgnosticData;
    dnnl_primitive_args& m_primArgs;
    bool resetSrcMemoryDataHandle = false;
    bool resetDstMemoryDataHandle = false;
    MemoryPtr m_scratchPadMemory;
    PrimitivePtr m_primitive;
};

}  // namespace intel_cpu
}  // namespace ov
