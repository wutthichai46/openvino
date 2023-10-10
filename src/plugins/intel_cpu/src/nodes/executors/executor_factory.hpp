// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/fullyconnected_implementations.h"
#include "nodes/executors/graph_emitter.hpp"
#include "nodes/executors/printers.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace intel_cpu {
using namespace executor;

template <typename Attrs, typename NodeT>
static ExecutorPtr fallback(const executor::Config<Attrs>& key,
                            const executor::Config<Attrs>& actualKey,
                            const MemoryArgs& memory,
                            const ExecutorContext::CPtr context,
                            const std::string& name) {
    constexpr int mask = NodeT::mask();
    GraphEmitter<Attrs> graphEmitter(key.descs, key.attrs, key.postOps, memory, mask, context, name);

    const auto& graphExecutor = graphEmitter.createGraph(actualKey.descs, actualKey.attrs, actualKey.postOps, context)
                                    .ensureAttrsMatch()
                                    .ensureSrcDescsMatch()
                                    .ensureDstDescsMatch()
                                    .ensurePostOpsMatch()
                                    .emit();

    return graphExecutor;
}

template <typename Attrs, typename NodeT>
class ExecutorFactoryNew {
public:
    ExecutorFactoryNew(const ExecutorContext::CPtr context) : context(context) {}
    ~ExecutorFactoryNew() {
        if (!collectCounters)
            return;

        std::cout << "Factory,";
        for (const auto& s : spend) {
            const int total = std::accumulate(s.begin(), s.end(), 0);
            const int average = total / s.size();
            std::cout << "," << average << "," << total << ",";
        }
    }

    MemoryDescArgs preconfigureMemoryDescriptors(const executor::Config<Attrs>& key) {
        DEBUG_LOG("Preconfiguring memory descriptors");

        const auto& impl = select(key);
        const auto& res = impl.isFullyCompliant(key);
        const bool fullyCompliant = res.first;

        if (fullyCompliant) {
            return key.descs;
        }

        const auto& actualKey = res.second;
        return actualKey.descs;
    }

    void filter(const executor::Config<Attrs>& key, const std::string& implementationPriority = {}) {
        const auto& implementations = getImplementations<Attrs>();

        for (const auto& implementation : implementations) {
            DEBUG_LOG("Processing implementation: ", implementation.name());
            if (!implementationPriority.empty() && implementation.name() != implementationPriority) {
                DEBUG_LOG("Implementation: ",
                          implementation.name(),
                          " does not match priority: ",
                          implementationPriority);
                continue;
            }

            if (!implementation.isSupported(key)) {
                DEBUG_LOG("Implementation is not supported: ", implementation.name());
                continue;
            }

            suitableImplementations.push_back(std::ref(implementation));

            // implementation is supported and it is shape agnostic, there is no way
            // an implementation with a lower priority will be chosen
            if (implementation.isShapeAgnostic()) {
                DEBUG_LOG("Implementation is shape agnostic: ",
                          implementation.name(),
                          ". Stop processing implementations");
                break;
            }
        }

        OPENVINO_ASSERT(!suitableImplementations.empty(), "No implementations supports the provided key");
    }

    void preconfigure(const executor::Config<Attrs>& key, const MemoryArgs& memory) {
        DEBUG_LOG("Preconfiguring factory");
        const auto& impl = select(key);
        const auto& res = impl.isFullyCompliant(key);
        const bool fullyCompliant = res.first;

        if (fullyCompliant) {
            DEBUG_LOG("Executor ", impl.name(), " is fully compliant with the required key");
            (void)create(impl, key, memory, context);
        }

        const auto& actualKey = res.second;
        (void)create(impl, actualKey, memory, context);
    }

    ExecutorPtr make(const executor::Config<Attrs>& key, const MemoryArgs& memory) {
        const auto& impl = select(key);
        const auto& res = impl.isFullyCompliant(key);
        const auto& actualKey = res.second;
        const bool fullyCompliant = res.first;

        if (fullyCompliant) {
            DEBUG_LOG("Executor implementation  ", impl.name(), " is fully compliant with the required key: ", key);
            const auto executor = create(impl, key, memory, context);
            DEBUG_LOG("Updating executor with the relevant memory");
            executor->update(key.descs, memory);
            return executor;
        }

        DEBUG_LOG("Falling back to graph executor for ", impl.name(), ". Original key: ", key, " new key:", actualKey);
        return fallback<Attrs, NodeT>(key, actualKey, memory, context, impl.name());
    }

private:
    const ExecutorImplementation<Attrs>& select(const executor::Config<Attrs>& key) {
        const auto selectedImplementation =
            std::find_if(suitableImplementations.begin(),
                         suitableImplementations.end(),
                         [&key](const std::reference_wrapper<const ExecutorImplementation<Attrs>> implementation) {
                             return implementation.get().isShapeAgnostic() || implementation.get().isShapeSuitable(key);
                         });
        OPENVINO_ASSERT(selectedImplementation != suitableImplementations.end(), "Failed to selected an implemetation");

        return *selectedImplementation;
    }

    ExecutorPtr create(const ExecutorImplementation<Attrs>& impl,
                       const executor::Config<Attrs>& key,
                       const MemoryArgs& memory,
                       const ExecutorContext::CPtr context) {
        DEBUG_LOG("Configuring implementation: ", impl.name());
        const auto& factoryId = std::make_pair(impl.type(), impl.operationType());
        auto factoryIt = factories.find(factoryId);
        if (factoryIt == factories.end()) {
            factoryIt = factories.insert(std::make_pair(factoryId, impl.doCreate(key, memory, context))).first;
        }

        return factoryIt->second;
    }

    const ExecutorContext::CPtr context;
    std::vector<std::reference_wrapper<const ExecutorImplementation<Attrs>>> suitableImplementations;
    // @todo unordered_map?
    std::map<std::pair<ExecutorType, OperationType>, ExecutorPtr> factories;

    int took() {
        const auto& now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - begin).count();
    }
    std::array<std::vector<int>, 5> spend;
    const bool collectCounters = std::getenv("COLLECT_COUNTERS");
    std::chrono::high_resolution_clock::time_point begin;
};

template <typename Attrs, typename NodeT>
using ExecutorFactoryNewPtr = std::shared_ptr<ExecutorFactoryNew<Attrs, NodeT>>;

template <typename Attrs, typename NodeT>
using ExecutorFactoryNewCPtr = std::shared_ptr<const ExecutorFactoryNew<Attrs, NodeT>>;

}  // namespace intel_cpu
}  // namespace ov
