// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_types.h"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {

template<typename Key, size_t idx = 0>
ov::element::Type srcType(const Key& key) {
    return key.descs.src[idx]->getPrecision();
}

template<typename Key>
ov::element::Type weiType(const Key& key) {
    return srcType<Key, 1>(key);
}

template<typename Key>
ov::element::Type biaType(const Key& key) {
    return srcType<Key, 2>(key);
}

template<typename Key, size_t idx = 0>
ov::element::Type dstType(const Key& key) {
    return key.descs.dst[idx]->getPrecision();
}

template<typename Key, size_t idx = 0>
const VectorDims& srcDims(const Key& key) {
    return key.descs.src[idx]->getShape().getDims();
}

template<typename Key>
const VectorDims& weiDims(const Key& key) {
    return srcDims<Key, 1>(key);
}

template<typename Key, size_t idx = 0>
size_t srcRank(const Key& key) {
    return key.descs.src[idx]->getShape().getRank();
}

template<typename Key, size_t idx = 0>
size_t weiRank(const Key& key) {
    return srcRank<Key, 1>(key);
}

template<typename Key, size_t idx = 0>
size_t srcMemSize(const Key& key) {
    return key.descs.src[idx]->getCurrentMemSize();
}

template<typename Key>
size_t weiMemSize(const Key& key) {
    return srcMemSize<Key, 1>(key);
}

}   // namespace intel_cpu
}   // namespace ov
