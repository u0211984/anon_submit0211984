/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Switch.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

Switch::Switch(const int npus_count,
           const Bandwidth bandwidth,
           const Latency latency,
           const bool bidirectional,
           const bool is_multi_dim,
           const std::vector<std::tuple<int, int, double>>& faulty_links) noexcept
    : bidirectional(bidirectional),
      // npus 0..npus_count-1 plus the switch itself (device npus_count).
      BasicTopology(npus_count, npus_count + 1, bandwidth, latency, is_multi_dim),
      faulty_links(faulty_links) {   // initialize faulty links
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    Switch::basic_topology_type = TopologyBuildingBlock::Switch;

    // set switch id
    switch_id = npus_count;

    // The switch forwards cut-through. NVSwitch forwards NVLink packets of at most 256 B payload through a non-blocking crossbar
    // (NVIDIA, Ishii et al., "NVSwitch and DGX-2", Hot Chips 30, 2018, slides 9-10; NVIDIA NVSwitch Technical
    // Overview, 2018, pp. 2-3).
    devices[switch_id]->set_cut_through();

    // connect every npu to the switch (a star), bidirectional, as route()
    // and get_connection_policies() assume. A faulty link derates the
    // npu<->switch link; a health of 0 keeps the nominal link (as before).
    if (!is_multi_dim) {
        for (auto i = 0; i < npus_count; i++) {
            const double derate = fault_derate(i, switch_id);
            connect(i, switch_id, derate != 0 ? bandwidth * derate : bandwidth, latency, true);
        }
    }
}

Route Switch::route(DeviceId src, DeviceId dest) const noexcept {
    // assert npus are in valid range
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    // construct route
    // start at source, and go to switch, then go to destination
    auto route = Route();
    route.push_back(devices[src]);
    route.push_back(devices[switch_id]);
    route.push_back(devices[dest]);

    return route;
}

std::vector<ConnectionPolicy> Switch::get_connection_policies() const noexcept {
    std::vector<ConnectionPolicy> policies;

    for (auto i = 0; i < npus_count; i++) {
        policies.emplace_back(i, switch_id);
        policies.emplace_back(switch_id, i);
    }

    return policies;
}

double Switch::fault_derate(int src, int dst) const{
    for (const auto& link : faulty_links) {
        int a = std::get<0>(link);
        int b = std::get<1>(link);
        double health = std::get<2>(link);

        // If this link exists and health != 0.0 → it's soft fault
        if ((a == src && b == dst) || (a == dst && b == src)) {
            return health;
        }
    }
    return 1.0;
}

