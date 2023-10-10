// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef CPU_DEBUG_CAPS
#pragma once

#include <ostream>
#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

namespace executor {
template<typename Attrs> struct Config;
}

struct FCAttrs;

std::ostream & operator<<(std::ostream & os, const FCAttrs& attrs);
std::ostream & operator<<(std::ostream & os, const PostOps& postOps);

template<typename Attrs>
std::ostream & operator<<(std::ostream & os, const executor::Config<Attrs>& key) {
    for (const auto& desc : key.descs.src)
        os << *desc << ";";
    for (const auto& desc : key.descs.dst)
        os << *desc << ";";

    os << key.postOps;
    os << key.attrs;

    return os;
}

}   // namespace intel_cpu
}   // namespace ov

#endif // CPU_DEBUG_CAPS
