// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_translation.hpp"

#include <iterator>

#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"
#include "precision_matcher.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

InOutTypes getTypeConfiguration(const TypeMapping& mapping, const MemoryDescArgs& descriptors) {
    std::vector<ov::element::Type> inputs;
    const auto& src = descriptors.src;
    std::transform(src.begin(), src.end(), std::back_inserter(inputs), [](const MemoryDescPtr& desc) {
        return desc->getPrecision();
    });

    const auto output = descriptors.dst[0]->getPrecision();

    for (const auto& entry : mapping) {
        const auto& inOutPattern = entry.first;
        const auto& inPattern = inOutPattern.first;
        const auto& outPattern = inOutPattern.second;
        if (!match(inPattern, inputs) || !match(outPattern, output))
            continue;

        const auto& translator = entry.second;
        return translator(inputs, output);
    }

    // @todo should we throw instead and require ANY -> FP32 entry in the container?
    return {std::vector<ov::element::Type>(inputs.size(), ov::element::f32), ov::element::f32};
}

}  // namespace intel_cpu
}  // namespace ov
