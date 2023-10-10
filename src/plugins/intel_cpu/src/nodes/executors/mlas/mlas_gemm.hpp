// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "nodes/executors/fullyconnected_config.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

class MlasGemmExecutor : public Executor {
public:
    MlasGemmExecutor(const FCConfig& key, const MemoryArgs& memory, const ExecutorContext::CPtr context);

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::gemm_mlas;
    }

    // offloads some execution information preparation from the exec call
    // void prepare(const MemoryDescAttrs& descs) {
    void update(const MemoryDescArgs& descs, const MemoryArgs& memory) override;

    static bool isSupported(const FCConfig& key);

private:
    const MemoryCPtr packedWeights;
    int64_t M, N, K;
};

using MlasGemmExecutorPtr = std::shared_ptr<MlasGemmExecutor>;

}  // namespace intel_cpu
}  // namespace ov
