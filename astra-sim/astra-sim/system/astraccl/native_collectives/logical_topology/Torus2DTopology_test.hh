/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __TORUS2D_TOPOLOGY_HH__
#define __TORUS2D_TOPOLOGY_HH__

#include "astra-sim/system/astraccl/native_collectives/logical_topology/ComplexLogicalTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"

namespace AstraSim {

class Torus2DTopology : public ComplexLogicalTopology {
 public:
    Torus2DTopology(int id, int total_nodes, int local_dim);
    ~Torus2DTopology();

    int get_num_of_dimensions() override;
    int get_num_of_nodes_in_dimension(int dimension) override;
    BasicLogicalTopology* get_basic_topology_at_dimension(
        int dimension, ComType type) override;

 private:
    RingTopology* local_dimension;
    RingTopology* vertical_dimension;
};

}  // namespace AstraSim

#endif /* __TORUS2D_TOPOLOGY_HH__ */
