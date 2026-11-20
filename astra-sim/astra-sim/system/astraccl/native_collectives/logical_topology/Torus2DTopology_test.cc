/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/Torus2DTopology_test.hh"

using namespace std;
using namespace AstraSim;

Torus2DTopology::Torus2DTopology(int id, int total_nodes, int local_dim) {
    // Compute second dimension size
    int vertical_dim = total_nodes / local_dim;

    local_dimension = new RingTopology(
        RingTopology::Dimension::Local,
        id,
        local_dim,
        id % local_dim,
        1
    );

    vertical_dimension = new RingTopology(
        RingTopology::Dimension::Vertical,
        id,
        vertical_dim,
        id / local_dim,
        local_dim
    );
}

Torus2DTopology::~Torus2DTopology() {
    delete local_dimension;
    delete vertical_dimension;
}

int Torus2DTopology::get_num_of_dimensions() {
    return 2;
}

int Torus2DTopology::get_num_of_nodes_in_dimension(int dimension) {
    if (dimension == 0) {
        return local_dimension->get_num_of_nodes_in_dimension(0);
    } else if (dimension == 1) {
        return vertical_dimension->get_num_of_nodes_in_dimension(0);
    }
    return -1;
}

BasicLogicalTopology* Torus2DTopology::get_basic_topology_at_dimension(
    int dimension, ComType type) {
    if (dimension == 0) {
        return local_dimension;
    } else if (dimension == 1) {
        return vertical_dimension;
    }
    return NULL;
}
