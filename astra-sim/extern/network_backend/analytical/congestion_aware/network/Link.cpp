/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Link.h"
#include "common/NetworkFunction.h"
#include "congestion_aware/Chunk.h"
#include "congestion_aware/Device.h"
#include "congestion_aware/TrafficStats.h"
#include <cassert>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

// declaring static event_queue
std::shared_ptr<EventQueue> Link::event_queue;

void Link::link_become_free(void* const link_ptr) noexcept {
    assert(link_ptr != nullptr);

    // cast to Link*
    auto* const link = static_cast<Link*>(link_ptr);

    // set link free
    link->set_free();

    // process pending chunks if one exist
    if (link->pending_chunk_exists()) {
        link->process_pending_transmission();
    }
}

void Link::set_event_queue(std::shared_ptr<EventQueue> event_queue_ptr) noexcept {
    assert(event_queue_ptr != nullptr);

    // set the event queue
    Link::event_queue = std::move(event_queue_ptr);
}

Link::Link(const Bandwidth bandwidth,
           const Latency latency,
           const int dimension_index,
           const DeviceId src_device_id,
           const DeviceId dst_device_id) noexcept
    : bandwidth(bandwidth),
      latency(latency),
      dimension_index(dimension_index),
      src_device_id(src_device_id),
      dst_device_id(dst_device_id),
      pending_chunks(),
      busy(false) {
    assert(bandwidth > 0);
    assert(latency >= 0);
    assert(dimension_index >= 0);

    // convert bandwidth from GB/s to B/ns
    bandwidth_Bpns = bw_GBps_to_Bpns(bandwidth);
}

void Link::send(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    if (busy) {
        // link is busy, add to pending chunks
        pending_chunks.push_back(std::move(chunk));
    } else {
        // service this chunk immediately
        schedule_chunk_transmission(std::move(chunk));
    }
}

void Link::process_pending_transmission() noexcept {
    // pending chunk should exist
    assert(pending_chunk_exists());

    // get chunk to process
    auto chunk = std::move(pending_chunks.front());
    pending_chunks.pop_front();

    // service this chunk
    schedule_chunk_transmission(std::move(chunk));
}

bool Link::pending_chunk_exists() const noexcept {
    // check pending chunks is not empty
    return !pending_chunks.empty();
}

void Link::set_busy() noexcept {
    // set busy to true
    busy = true;
}

void Link::set_free() noexcept {
    // set busy to false
    busy = false;
}

EventTime Link::serialization_delay(const ChunkSize chunk_size) const noexcept {
    assert(chunk_size > 0);

    // calculate serialization delay
    const auto delay = static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns;

    // return serialization delay in EventTime type
    return static_cast<EventTime>(delay);
}

EventTime Link::communication_delay(const ChunkSize chunk_size) const noexcept {
    assert(chunk_size > 0);

    // calculate communication delay
    const auto delay = latency + (static_cast<Bandwidth>(chunk_size) / bandwidth_Bpns);

    // return communication delay in EventTime type
    return static_cast<EventTime>(delay);
}

void Link::schedule_chunk_transmission(std::unique_ptr<Chunk> chunk) noexcept {
    assert(chunk != nullptr);

    // link should be free
    assert(!busy);

    // set link busy
    set_busy();

    // get metadata
    const auto chunk_size = chunk->get_size();
    const auto current_time = Link::event_queue->get_current_time();
    const auto serialization_time = serialization_delay(chunk_size);
    TrafficStats::record_link(dimension_index,
                              src_device_id,
                              dst_device_id,
                              chunk_size,
                              serialization_time,
                              chunk->get_traffic_context());

    // Tail of the chunk leaves this link after its serialization, and no earlier than the tail reaches this
    // device (when an upstream link fed it cut-through). At a source device, or with equal hop bandwidths,
    // the bound is inactive and the timing is the original one.
    auto tail_leave_time = current_time + serialization_time;
    auto tail_arrival_time = current_time + communication_delay(chunk_size);
    if (chunk->get_tail_ready_time() > tail_leave_time) {
        tail_leave_time = chunk->get_tail_ready_time();
        tail_arrival_time = tail_leave_time + static_cast<EventTime>(latency);
    }

    // Cut-through at a switch device: the next link starts when the chunk's head arrives (after the link
    // latency), not after the whole chunk has been received. Otherwise (NPU next hop, i.e., the destination or
    // an NPU relay) the chunk arrives after its full serialization (store and forward), as before.
    // NVSwitch forwards NVLink packets of at most 256 B payload through a non-blocking crossbar
    // (NVIDIA, Ishii et al., "NVSwitch and DGX-2", Hot Chips 30, 2018, slides 9-10; NVIDIA NVSwitch Technical
    // Overview, 2018, pp. 2-3).
    auto chunk_arrival_time = tail_arrival_time;
    if (chunk->next_device()->is_cut_through()) {
        chunk->set_tail_ready_time(tail_arrival_time);
        chunk_arrival_time = current_time + static_cast<EventTime>(latency);
    }

    // schedule chunk arrival event
    auto* const chunk_ptr = static_cast<void*>(chunk.release());
    Link::event_queue->schedule_event(chunk_arrival_time, Chunk::chunk_arrived_next_device, chunk_ptr);

    // schedule link free time (the link is held until the chunk's tail has left it)
    const auto link_free_time = tail_leave_time;
    auto* const link_ptr = static_cast<void*>(this);
    Link::event_queue->schedule_event(link_free_time, link_become_free, link_ptr);
}
