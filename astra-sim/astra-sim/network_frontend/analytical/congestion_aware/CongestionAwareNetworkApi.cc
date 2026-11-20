/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/Chunk.h>
#include <astra-network-analytical/congestion_aware/TrafficStats.h>
#include <cassert>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;

namespace {

std::string collective_type_name(const int collective_type) {
    switch (collective_type) {
        case 0:
            return "ALL_REDUCE";
        case 2:
            return "ALL_GATHER";
        case 4:
            return "SCATTER";
        case 5:
            return "BROADCAST";
        case 6:
            return "ALL_TO_ALL";
        case 7:
            return "REDUCE_SCATTER";
        case 8:
            return "REDUCE_SCATTER_BLOCK";
        default:
            return "";
    }
}

std::string phase_type_name(const int phase_type) {
    switch (static_cast<ComType>(phase_type)) {
        case ComType::Reduce_Scatter:
            return "Reduce_Scatter";
        case ComType::All_Gather:
            return "All_Gather";
        case ComType::All_Reduce:
            return "All_Reduce";
        case ComType::All_to_All:
            return "All_to_All";
        case ComType::All_Reduce_All_to_All:
            return "All_Reduce_All_to_All";
        default:
            return "";
    }
}

TrafficContext traffic_context_from_request(const sim_request* const request) {
    auto context = TrafficContext();
    if (request == nullptr) {
        return context;
    }
    context.operation_id = request->traffic_operation_id;
    context.dataset_id = request->traffic_dataset_id;
    context.collective_type = collective_type_name(request->traffic_collective_type);
    context.collective_size_bytes = request->traffic_collective_size_bytes;
    context.stream_id = request->traffic_stream_id;
    context.phase_index = request->traffic_phase_index;
    context.phase_type = phase_type_name(request->traffic_phase_type);
    context.activity_multiplier = request->traffic_activity_multiplier;
    return context;
}

}  // namespace

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

CongestionAwareNetworkApi::CongestionAwareNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionAwareNetworkApi::sim_send(void* const buffer,
                                        const uint64_t count,
                                        const int type,
                                        const int dst,
                                        const int tag,
                                        sim_request* const request,
                                        void (*msg_handler)(void*),
                                        void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionAwareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // create chunk
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());
    const auto route = topology->route(src, dst);
    const auto traffic_context = traffic_context_from_request(request);
    auto chunk = std::make_unique<Chunk>(
        count, route, CongestionAwareNetworkApi::process_chunk_arrival,
        arg_ptr, traffic_context);

    // initiate transmission from src -> dst.
    topology->send(std::move(chunk));

    // return
    return 0;
}
