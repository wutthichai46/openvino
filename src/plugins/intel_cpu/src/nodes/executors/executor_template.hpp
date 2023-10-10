// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "convolution_config.hpp"
#include "executor.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

// @todo do we need this intermediate base class at all?
template <typename Attributes>
class ExecutorTemplate : public Executor {
public:
    ExecutorTemplate(const ExecutorContext::CPtr context, const Attributes& attrs) : attrs(attrs), context(context) {}

    // @todo ideal executor should only have "execute" interface
    // memory preparation can be extracted into some "ExecutorDescriptor" abstraction
    virtual ~ExecutorTemplate() = default;

protected:
    const Attributes attrs;
    const ExecutorContext::CPtr context;
};

template <class T>
using ExecutorTemplatePtr = std::shared_ptr<ExecutorTemplate<T>>;
template <class T>
using ExecutorTemplateCPtr = std::shared_ptr<const ExecutorTemplate<T>>;

}  // namespace intel_cpu
}  // namespace ov
