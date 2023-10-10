// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <functional>
#include <vector>

#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/precision_matcher.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/precision_support.h"

namespace ov {
namespace intel_cpu {

struct out {
    ov::element::Type operator()(const std::vector<ov::element::Type>& inputs,
                                 const ov::element::Type output,
                                 const size_t idx) const {
        (void)inputs;
        (void)idx;
        return output;
    }
};

template <size_t bypassId>
struct in {
    ov::element::Type operator()(const std::vector<ov::element::Type>& inputs,
                                 const ov::element::Type output,
                                 const size_t idx) const {
        (void)output;
        (void)idx;
        assert(bypassId < inputs.size());
        return inputs[bypassId];
    }
};

struct bypass {
    ov::element::Type operator()(const std::vector<ov::element::Type>& inputs,
                                 const ov::element::Type output,
                                 const size_t idx) const {
        (void)output;
        (void)idx;
        return inputs[idx];
    }
};

template <ov::element::Type_t type>
struct just {
    ov::element::Type operator()(const std::vector<ov::element::Type>& inputs,
                                 const ov::element::Type output,
                                 const size_t idx) const {
        // ignore all the inputs
        (void)inputs;
        (void)output;
        (void)idx;
        return type;
    }
};

template <>
struct just<TypeMaskAlias::fxx> {
    ov::element::Type operator()(const std::vector<ov::element::Type>& inputs,
                                 const ov::element::Type output,
                                 const size_t idx) const {
        // ignore all the inputs
        (void)inputs;
        (void)output;
        (void)idx;
        return defaultFloatPrecision();
    }
};

template <typename SrcPolicy, typename WeiPolicy, typename BiasPolicy, typename DstPolicy>
struct PortsTranslation {
    std::pair<std::vector<ov::element::Type>, ov::element::Type> operator()(
        const std::vector<ov::element::Type>& inputs,
        const ov::element::Type output) const {
        return {
            {SrcPolicy{}(inputs, output, 0), WeiPolicy{}(inputs, output, 1), BiasPolicy{}(inputs, output, 2)},
            DstPolicy{}
            (inputs, output, 0)
        };
    }
};

// @todo vectors can be replaced with arrays, since we know the size beforehand
//       pros: should be more efficient and safe
//       cons: more template instances (binary size) of the translation utility functions
using InOutTypes = std::pair<std::vector<ov::element::Type>, ov::element::Type>;
using PortsConfigurationImpl = std::function<InOutTypes(const std::vector<ov::element::Type>&, ov::element::Type)>;
using InOutTypeMask = std::pair<std::vector<TypeMask>, TypeMask>;
using TypeMapping = std::vector<std::pair<InOutTypeMask, PortsConfigurationImpl>>;

template <typename SrcPolicy, typename WeiPolicy, typename BiasPolicy, typename DstPolicy>
using pt = PortsTranslation<SrcPolicy, WeiPolicy, BiasPolicy, DstPolicy>;

template <typename Policy>
using everyone = PortsTranslation<Policy, Policy, Policy, Policy>;

InOutTypes getTypeConfiguration(const TypeMapping& mapping, const MemoryDescArgs& descriptors);

}  // namespace intel_cpu
}  // namespace ov
