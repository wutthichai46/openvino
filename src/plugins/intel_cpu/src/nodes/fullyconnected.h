// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

#include <chrono>
#include <memory>
#include <numeric>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    ~FullyConnected() {
        if (!collectCounters)
            return;

        std::cout << "Result," << getName();
        for (const auto& s : spend) {
            const int total = std::accumulate(s.begin(), s.end(), 0);
            const int average = total / s.size();
            std::cout << "," << executor->implType() << "," << average << "," << total;
        }
        std::cout << '\n';
    }

    void getSupportedDescriptors() override{};
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    size_t descInputNumbers() override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;

    ov::element::Type getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeExecutedInInt8() const override;
    // @todo ideally attributes should be filled in constructor and never changed
    // this require to get rid of graph optimizer and to only use ngraph transformation engine
    void keepWeightsNonTransposed(bool weightsNonTransposed) {
        this->attrs.weightsNonTransposed = weightsNonTransposed;
    }

    void fuseDecompressionMultiply(const MemoryCPtr& memory);
    void fuseDecompressionSubtract(const MemoryCPtr& memory);
    // defines which of the inputs are constant
    constexpr static int mask() {
        return 0 + (1 << 1);
    }

private:
    std::string errorPrefix;
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;

    ExecutorPtr executor = nullptr;

    ExecutorPtr createExecutor();
    void fuseDecompressionConstant(const MemoryCPtr& memory, MemoryCPtr& decompressionValuesPtr);

    FCAttrs attrs;
    PostOps postOps;
    ExecutorFactoryNewPtr<FCAttrs, node::FullyConnected> factory;
    MemoryArgs memory;
    MemoryDescArgs descriptors;
    MemoryPtr emptyMemory;
    const bool collectCounters = std::getenv("COLLECT_COUNTERS");
    int took();
    std::array<std::vector<int>, 5> spend;
    std::chrono::high_resolution_clock::time_point begin;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
