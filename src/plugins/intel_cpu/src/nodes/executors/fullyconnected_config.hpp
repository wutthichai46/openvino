// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cpu_memory.h"
#include "executor_config.hpp"
#include "hash_builder.hpp"

namespace ov {
namespace intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    // @todo probably we don't want with bias flag, since this information is already
    // a part of src memory descs
    bool withBias = false;
    // @todo probably we don't want with bias flag, since this information is already
    // a part of weight memory desc
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    // @todo only memory descriptors should be a part of attributes
    // actual memory should be passed into "execute" or "prepareMemory" calls
    std::vector<float> dequantizationScales;
    // @todo should be memory desc or vector dims
    MemoryCPtr decompressionSubtractPtr;
    MemoryCPtr decompressionMultiplyPtr;

    size_t hash(size_t seed) const {
        using namespace hash;
        seed = Builder(seed)
            .combine(withBias)
            .combine(weightsNonTransposed)
            .combine(sparseWeights)
            .combine(dequantizationScales.size())
            .generate();
        if (decompressionSubtractPtr)
            seed = combine(seed, decompressionSubtractPtr->getShape().getStaticDims());
        if (decompressionMultiplyPtr)
            seed = combine(seed, decompressionMultiplyPtr->getShape().getStaticDims());
        return seed;
    }

    bool operator==(const FCAttrs& rhs) const {
        bool result = true;
        result = result && withBias == rhs.withBias;
        result = result && weightsNonTransposed == rhs.weightsNonTransposed;
        result = result && sparseWeights == rhs.sparseWeights;
        result = result && decompressionMultiplyPtr && rhs.decompressionMultiplyPtr && decompressionMultiplyPtr->getShape().getStaticDims() ==
                               rhs.decompressionMultiplyPtr->getShape().getStaticDims();
        result = result && decompressionSubtractPtr && rhs.decompressionMultiplyPtr && decompressionMultiplyPtr->getShape().getStaticDims() ==
                               rhs.decompressionSubtractPtr->getShape().getStaticDims();
        result = result && dequantizationScales.size() == rhs.dequantizationScales.size();
        return result;
    }
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace intel_cpu
}  // namespace ov
