// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "hash_builder.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @todo only attributes necessary for 1x1 convlution as fullyconnected fallback
 * are currently listed
 */
struct ConvAttrs {
    bool withBias;
    size_t hash(size_t seed) const {
        using namespace hash;
        return Builder(seed).combine(withBias).generate();
    }

    bool operator==(const ConvAttrs& rhs) const {
        bool result = true;
        result = result && withBias == rhs.withBias;
        return result;
    }
};

using ConvConfig = executor::Config<ConvAttrs>;

}  // namespace intel_cpu
}  // namespace ov
