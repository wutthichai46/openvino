// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "precision_matcher.hpp"

#include <cassert>

#include "openvino/core/type/element_type.hpp"

using namespace ov::element;

namespace ov {
namespace intel_cpu {

bool match(const TypeMask pattern, const Type value) {
    return pattern & value;
}

bool match(const std::vector<TypeMask>& patterns, const std::vector<Type>& values) {
    assert(values.size() <= patterns.size());

    return std::equal(values.begin(), values.end(), patterns.begin(), [](const Type value, const TypeMask pattern) {
        return pattern & value;
    });
}

}  // namespace intel_cpu
}  // namespace ov
