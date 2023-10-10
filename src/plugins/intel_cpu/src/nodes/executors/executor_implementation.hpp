// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"

namespace ov {
namespace intel_cpu {

// @todo Consider alternative of using template arguments instead of std::functions
template <typename Attrs>
class ExecutorImplementation {
public:
    ExecutorImplementation(
        const char* name,
        const ExecutorType type,
        const OperationType operationType,
        const ShapeTolerance shapeRelation,
        std::function<bool(const executor::Config<Attrs>&)> isSupported,
        std::function<std::pair<bool, executor::Config<Attrs>>(const executor::Config<Attrs>&)> isFullyCompliant,
        std::function<bool(const executor::Config<Attrs>&)> isShapeSuitable,
        std::function<ExecutorPtr(const executor::Config<Attrs>&,
                                  const MemoryArgs& memory,
                                  const ExecutorContext::CPtr context)> create)
        : m_name(name),
          m_type(type),
          m_operationType(operationType),
          m_shapeRelation(shapeRelation),
          m_isSupported(std::move(isSupported)),
          m_isFullyCompliant(std::move(isFullyCompliant)),
          m_isShapeSuitable(std::move(isShapeSuitable)),
          m_create(std::move(create)) {}

    bool isSupported(const executor::Config<Attrs>& key) const {
        if (m_isSupported) {
            return m_isSupported(key);
        }

        return false;
    }

    std::pair<bool, executor::Config<Attrs>> isFullyCompliant(const executor::Config<Attrs>& key) const {
        if (m_isFullyCompliant) {
            return m_isFullyCompliant(key);
        }

        return {false, key};
    }

    bool isShapeSuitable(const executor::Config<Attrs>& key) const {
        if (m_isShapeSuitable) {
            return m_isShapeSuitable(key);
        }

        return false;
    }

    ExecutorPtr doCreate(const executor::Config<Attrs>& key,
                         const MemoryArgs& memory,
                         const ExecutorContext::CPtr context) const {
        // @todo require configure to be defined
        if (m_create)
            return m_create(key, memory, context);
        return nullptr;
    }

    bool isShapeAgnostic() const {
        return m_shapeRelation == ShapeTolerance::Agnostic;
    }

    const char* name() const {
        return m_name;
    }

    const ExecutorType type() const {
        return m_type;
    }

    const OperationType operationType() const {
        return m_operationType;
    }

private:
    const char* m_name;
    const ExecutorType m_type;
    const OperationType m_operationType;
    const ShapeTolerance m_shapeRelation;
    // @todo create aliases for std::functions
    const std::function<bool(const executor::Config<Attrs>&)> m_isSupported = {};
    const std::function<std::pair<bool, executor::Config<Attrs>>(const executor::Config<Attrs>&)> m_isFullyCompliant = {};
    const std::function<bool(const executor::Config<Attrs>&)> m_isShapeSuitable = {};
    const std::function<
        ExecutorPtr(const executor::Config<Attrs>&, const MemoryArgs& memory, const ExecutorContext::CPtr context)>
        m_create = {};
};

template <typename Attrs>
using ExecutorImplementationPtr = std::shared_ptr<ExecutorImplementation<Attrs>>;
}  // namespace intel_cpu
}  // namespace ov
