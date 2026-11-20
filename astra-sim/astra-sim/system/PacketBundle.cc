/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

using namespace AstraSim;

PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           std::list<MyPacket*> locked_packets,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->locked_packets = locked_packets;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->stream = stream;
    this->transmition = transmition;
    creation_time = Sys::boostedTick();
}

PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->stream = stream;
    this->transmition = transmition;
    creation_time = Sys::boostedTick();
}

void PacketBundle::send_to_MA() {
    sys->memBus->send_from_NPU_to_MA(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::send_to_NPU() {
    sys->memBus->send_from_MA_to_NPU(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::call(EventType event, CallData* data) {
    if (needs_processing == true) {
        needs_processing = false;
        int dimension_index = -1;
        if (sys->scheduler_unit != nullptr && stream != nullptr) {
            const auto dimension_it =
                sys->scheduler_unit->queue_id_to_dimension.find(
                    stream->current_queue_id);
            if (dimension_it !=
                sys->scheduler_unit->queue_id_to_dimension.end()) {
                dimension_index = dimension_it->second;
            }
        }
        int64_t operation_id = -1;
        if (stream != nullptr && stream->dataset != nullptr) {
            operation_id = stream->dataset->traffic_operation_id;
        }
        auto& activity = sys->reduction_memory_activity[
            std::make_tuple(operation_id, dimension_index)];
        const auto activity_multiplier =
            (stream != nullptr && stream->dataset != nullptr)
                ? stream->dataset->traffic_activity_multiplier
                : 1;
        activity.processed_bytes += size * activity_multiplier;
        activity.memory_access_bytes += size * 3 * activity_multiplier;
        activity.events += 1;
        // A fused reduction (NCCL ring) overlaps the transfer slice by slice, so the message
        // is released without a serial local-memory pass. Its memory traffic is recorded above.
        // NCCL v2.27.7 src/device/reduce_scatter.h:13-55 (runRing: send, recvReduceSend, recvReduceCopy) and
        // src/device/all_reduce.h:42-64 (runRing: directRecvReduceDirectSend); each primitive is one genericOp pass
        // (src/device/prims_simple.h:184-299, recvReduceSend at :916-918) that, slice by slice, waits for the
        // received slice, loads it with the local input, reduces and stores/sends it (reduceCopy and postPeer, :266-294).
        if (!fused_reduction) {
            // this->delay[ns], size[bytes] local_mem_bw[bytes/s]
            this->delay = static_cast<uint64_t>(static_cast<double>(size) /
                                                sys->local_mem_bw * 1e9)  // write
                          + static_cast<uint64_t>(static_cast<double>(size) /
                                                  sys->local_mem_bw * 1e9)  // read
                          + static_cast<uint64_t>(static_cast<double>(size) /
                                                  sys->local_mem_bw * 1e9);  // read
            sys->try_register_event(this, EventType::CommProcessingFinished, data,
                                    this->delay);
            return;
        }
        this->delay = 0;
    }
    Tick current = Sys::boostedTick();
    for (auto& packet : locked_packets) {
        packet->ready_time = current;
    }
    stream->call(EventType::General, data);
    delete this;
}
