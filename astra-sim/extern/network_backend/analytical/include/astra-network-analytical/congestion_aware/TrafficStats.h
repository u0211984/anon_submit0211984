/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

struct TrafficContext {
    int64_t operation_id = -1;
    int64_t dataset_id = -1;
    std::string collective_type;
    uint64_t collective_size_bytes = 0;
    int64_t stream_id = -1;
    int64_t phase_index = -1;
    std::string phase_type;
    uint64_t activity_multiplier = 1;

    [[nodiscard]] bool has_operation() const noexcept {
        return operation_id >= 0;
    }
};

struct TrafficCounter {
    uint64_t bytes = 0;
    uint64_t chunks = 0;
    uint64_t busy_time_ns = 0;
};

struct TrafficOperation {
    int64_t operation_id = -1;
    int64_t dataset_id = -1;
    std::string collective_type;
    uint64_t collective_size_bytes = 0;
};

struct ReductionMemoryCounter {
    uint64_t processed_bytes = 0;
    uint64_t memory_access_bytes = 0;
    uint64_t events = 0;
};

class TrafficStats {
  public:
    using LinkKey = std::tuple<int, DeviceId, DeviceId>;
    using DimensionKey = int;
    using OperationLinkKey = std::tuple<int64_t, int, DeviceId, DeviceId>;
    using OperationDimensionKey = std::tuple<int64_t, int>;
    using ReductionMemoryKey = std::tuple<DeviceId, int64_t, int>;

    static void reset() noexcept;
    static void register_operation(const TrafficContext& context) noexcept;
    static void record_link(int dimension_index,
                            DeviceId src_device_id,
                            DeviceId dst_device_id,
                            ChunkSize chunk_size,
                            EventTime busy_time_ns,
                            const TrafficContext& context) noexcept;
    static void record_reduction_memory(DeviceId device_id,
                                        int64_t operation_id,
                                        int dimension_index,
                                        uint64_t processed_bytes,
                                        uint64_t memory_access_bytes,
                                        uint64_t events) noexcept;
    static void write_json(const std::string& output_path, const std::string& run_scope);

    static std::vector<std::tuple<int, DeviceId, DeviceId, TrafficCounter>> link_snapshot();
    static std::vector<std::tuple<int, TrafficCounter>> dimension_snapshot();
    static std::vector<std::tuple<int64_t, int, DeviceId, DeviceId, TrafficCounter>> operation_link_snapshot();
    static std::vector<std::tuple<int64_t, int, TrafficCounter>> operation_dimension_snapshot();
    static std::vector<TrafficOperation> operation_snapshot();
    static std::vector<std::tuple<DeviceId, int64_t, int, ReductionMemoryCounter>>
    reduction_memory_snapshot();

  private:
    static std::map<LinkKey, TrafficCounter> links;
    static std::map<DimensionKey, TrafficCounter> dimensions;
    static std::map<OperationLinkKey, TrafficCounter> operation_links;
    static std::map<OperationDimensionKey, TrafficCounter> operation_dimensions;
    static std::map<int64_t, TrafficOperation> operations;
    static std::map<ReductionMemoryKey, ReductionMemoryCounter>
        reduction_memory;
};

}  // namespace NetworkAnalyticalCongestionAware
