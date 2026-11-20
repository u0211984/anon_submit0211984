/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/TrafficStats.h"
#include <fstream>
#include <stdexcept>

using namespace NetworkAnalyticalCongestionAware;

std::map<TrafficStats::LinkKey, TrafficCounter> TrafficStats::links;
std::map<TrafficStats::DimensionKey, TrafficCounter> TrafficStats::dimensions;
std::map<TrafficStats::OperationLinkKey, TrafficCounter> TrafficStats::operation_links;
std::map<TrafficStats::OperationDimensionKey, TrafficCounter> TrafficStats::operation_dimensions;
std::map<int64_t, TrafficOperation> TrafficStats::operations;
std::map<TrafficStats::ReductionMemoryKey, ReductionMemoryCounter>
    TrafficStats::reduction_memory;

namespace {

void add_count(TrafficCounter& counter, const ChunkSize chunk_size,
               const EventTime busy_time_ns) noexcept {
    counter.bytes += static_cast<uint64_t>(chunk_size);
    counter.chunks += 1;
    counter.busy_time_ns += static_cast<uint64_t>(busy_time_ns);
}

std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (const char ch : input) {
        switch (ch) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += ch;
                break;
        }
    }
    return out;
}

void write_counter_fields(std::ofstream& out, const TrafficCounter& counter) {
    out << "\"bytes\":" << counter.bytes << ",\"chunks\":" << counter.chunks
        << ",\"busy_time_ns\":" << counter.busy_time_ns;
}

}  // namespace

void TrafficStats::reset() noexcept {
    links.clear();
    dimensions.clear();
    operation_links.clear();
    operation_dimensions.clear();
    operations.clear();
    reduction_memory.clear();
}

void TrafficStats::record_reduction_memory(const DeviceId device_id,
                                           const int64_t operation_id,
                                           const int dimension_index,
                                           const uint64_t processed_bytes,
                                           const uint64_t memory_access_bytes,
                                           const uint64_t events) noexcept {
    auto& counter = reduction_memory[
        std::make_tuple(device_id, operation_id, dimension_index)];
    counter.processed_bytes += processed_bytes;
    counter.memory_access_bytes += memory_access_bytes;
    counter.events += events;
}

void TrafficStats::register_operation(const TrafficContext& context) noexcept {
    if (!context.has_operation()) {
        return;
    }
    auto& operation = operations[context.operation_id];
    operation.operation_id = context.operation_id;
    operation.dataset_id = context.dataset_id;
    operation.collective_type = context.collective_type;
    operation.collective_size_bytes = context.collective_size_bytes;
}

void TrafficStats::record_link(const int dimension_index,
                               const DeviceId src_device_id,
                               const DeviceId dst_device_id,
                               const ChunkSize chunk_size,
                               const EventTime busy_time_ns,
                               const TrafficContext& context) noexcept {
    const auto logical_chunk_size = chunk_size * context.activity_multiplier;
    add_count(links[std::make_tuple(dimension_index, src_device_id, dst_device_id)], logical_chunk_size, busy_time_ns);
    add_count(dimensions[dimension_index], logical_chunk_size, busy_time_ns);

    if (!context.has_operation()) {
        return;
    }
    register_operation(context);
    add_count(operation_links[std::make_tuple(context.operation_id, dimension_index, src_device_id, dst_device_id)],
              logical_chunk_size, busy_time_ns);
    add_count(operation_dimensions[std::make_tuple(context.operation_id, dimension_index)], logical_chunk_size, busy_time_ns);
}

void TrafficStats::write_json(const std::string& output_path, const std::string& run_scope) {
    if (output_path.empty()) {
        return;
    }
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("Unable to open link stats output path: " + output_path);
    }

    out << "{\"schema_version\":2,\"run_scope\":\"" << json_escape(run_scope) << "\"";

    out << ",\"links\":[";
    bool first = true;
    for (const auto& entry : links) {
        if (!first) {
            out << ",";
        }
        first = false;
        const auto& key = entry.first;
        out << "{\"dimension_index\":" << std::get<0>(key)
            << ",\"src_device_id\":" << std::get<1>(key)
            << ",\"dst_device_id\":" << std::get<2>(key) << ",";
        write_counter_fields(out, entry.second);
        out << "}";
    }
    out << "]";

    out << ",\"reduction_memory_activity\":[";
    first = true;
    for (const auto& entry : reduction_memory) {
        if (!first) {
            out << ",";
        }
        first = false;
        const auto& key = entry.first;
        const auto& counter = entry.second;
        out << "{\"device_id\":" << std::get<0>(key)
            << ",\"operation_id\":" << std::get<1>(key)
            << ",\"dimension_index\":" << std::get<2>(key)
            << ",\"processed_bytes\":" << counter.processed_bytes
            << ",\"memory_access_bytes\":" << counter.memory_access_bytes
            << ",\"events\":" << counter.events << "}";
    }
    out << "]";

    out << ",\"dimensions\":[";
    first = true;
    for (const auto& entry : dimensions) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "{\"dimension_index\":" << entry.first << ",";
        write_counter_fields(out, entry.second);
        out << "}";
    }
    out << "]";

    out << ",\"operations\":[";
    first = true;
    for (const auto& entry : operations) {
        if (!first) {
            out << ",";
        }
        first = false;
        const auto& operation = entry.second;
        out << "{\"operation_id\":" << operation.operation_id
            << ",\"dataset_id\":" << operation.dataset_id
            << ",\"collective_type\":\"" << json_escape(operation.collective_type) << "\""
            << ",\"collective_size_bytes\":" << operation.collective_size_bytes << "}";
    }
    out << "]";

    out << ",\"operation_links\":[";
    first = true;
    for (const auto& entry : operation_links) {
        if (!first) {
            out << ",";
        }
        first = false;
        const auto& key = entry.first;
        out << "{\"operation_id\":" << std::get<0>(key)
            << ",\"dimension_index\":" << std::get<1>(key)
            << ",\"src_device_id\":" << std::get<2>(key)
            << ",\"dst_device_id\":" << std::get<3>(key) << ",";
        write_counter_fields(out, entry.second);
        out << "}";
    }
    out << "]";

    out << ",\"operation_dimensions\":[";
    first = true;
    for (const auto& entry : operation_dimensions) {
        if (!first) {
            out << ",";
        }
        first = false;
        const auto& key = entry.first;
        out << "{\"operation_id\":" << std::get<0>(key)
            << ",\"dimension_index\":" << std::get<1>(key) << ",";
        write_counter_fields(out, entry.second);
        out << "}";
    }
    out << "]}";
}

std::vector<std::tuple<int, DeviceId, DeviceId, TrafficCounter>> TrafficStats::link_snapshot() {
    std::vector<std::tuple<int, DeviceId, DeviceId, TrafficCounter>> snapshot;
    for (const auto& entry : links) {
        snapshot.emplace_back(std::get<0>(entry.first), std::get<1>(entry.first), std::get<2>(entry.first),
                              entry.second);
    }
    return snapshot;
}

std::vector<std::tuple<int, TrafficCounter>> TrafficStats::dimension_snapshot() {
    std::vector<std::tuple<int, TrafficCounter>> snapshot;
    for (const auto& entry : dimensions) {
        snapshot.emplace_back(entry.first, entry.second);
    }
    return snapshot;
}

std::vector<std::tuple<int64_t, int, DeviceId, DeviceId, TrafficCounter>> TrafficStats::operation_link_snapshot() {
    std::vector<std::tuple<int64_t, int, DeviceId, DeviceId, TrafficCounter>> snapshot;
    for (const auto& entry : operation_links) {
        snapshot.emplace_back(std::get<0>(entry.first), std::get<1>(entry.first), std::get<2>(entry.first),
                              std::get<3>(entry.first), entry.second);
    }
    return snapshot;
}

std::vector<std::tuple<int64_t, int, TrafficCounter>> TrafficStats::operation_dimension_snapshot() {
    std::vector<std::tuple<int64_t, int, TrafficCounter>> snapshot;
    for (const auto& entry : operation_dimensions) {
        snapshot.emplace_back(std::get<0>(entry.first), std::get<1>(entry.first), entry.second);
    }
    return snapshot;
}

std::vector<TrafficOperation> TrafficStats::operation_snapshot() {
    std::vector<TrafficOperation> snapshot;
    for (const auto& entry : operations) {
        snapshot.push_back(entry.second);
    }
    return snapshot;
}

std::vector<std::tuple<DeviceId, int64_t, int, ReductionMemoryCounter>>
TrafficStats::reduction_memory_snapshot() {
    std::vector<std::tuple<DeviceId, int64_t, int, ReductionMemoryCounter>>
        snapshot;
    for (const auto& entry : reduction_memory) {
        snapshot.emplace_back(std::get<0>(entry.first),
                              std::get<1>(entry.first),
                              std::get<2>(entry.first), entry.second);
    }
    return snapshot;
}
