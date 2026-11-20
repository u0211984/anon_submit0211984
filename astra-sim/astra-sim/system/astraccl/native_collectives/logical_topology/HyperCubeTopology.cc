/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/HyperCubeTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>

using namespace std;
using namespace AstraSim;

HyperCubeTopology::HyperCubeTopology(Dimension dimension, int id, std::vector<int> NPUs)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::HyperCube) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_hypercube = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_hypercube = -1;

    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_hypercube};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_hypercube));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_hypercube};
    }

    for (int i = 0; i < total_nodes_in_hypercube; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_hypercube = i;
        }
    }

    LoggerFactory::get_logger("system::topology::HyperCubeTopology")
        ->info("custom hypercube, id: {}, dimension: {} total nodes in hypercube: {} "
               "index in hypercube: {} total nodes in hypercube {}",
               id, name, total_nodes_in_hypercube, index_in_hypercube,
               total_nodes_in_hypercube);

    assert(index_in_hypercube >= 0);
}

HyperCubeTopology::HyperCubeTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_hypercube,
                           int index_in_hypercube,
                           int offset)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::HyperCube) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::HyperCubeTopology")
            ->info("hypercube of node 0, id: {} dimension: {} total nodes in hypercube: "
                   "{} index in hypercube: {} offset: {} total nodes in hypercube: {}",
                   id, name, total_nodes_in_hypercube, index_in_hypercube, offset,
                   total_nodes_in_hypercube);
    }
    this->id = id;
    this->total_nodes_in_hypercube = total_nodes_in_hypercube;
    this->index_in_hypercube = index_in_hypercube;
    this->dimension = dimension;
    this->offset = offset;

    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_hypercube};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_hypercube));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_hypercube};
    }

    id_to_index[id] = index_in_hypercube;
    index_to_id[index_in_hypercube] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_hypercube - 1; i++) {
        tmp = get_receiver_homogeneous(tmp, HyperCubeTopology::Direction::Clockwise,
                                       offset);
    }
}


int HyperCubeTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Clockwise) {
        int receiver = node_id + offset;
        if (index == total_nodes_in_hypercube - 1) {
            receiver -= (total_nodes_in_hypercube * offset);
            index = 0;
        } else {
            index++;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::HyperCubeTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_hypercube {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_hypercube, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    } else {
        int receiver = node_id - offset;
        if (index == 0) {
            receiver += (total_nodes_in_hypercube * offset);
            index = total_nodes_in_hypercube - 1;
        } else {
            index--;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::HyperCubeTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_hypercube {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_hypercube, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    }
}

int HyperCubeTopology::get_receiver(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Clockwise) {
        index++;
        if (index == total_nodes_in_hypercube) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_hypercube - 1;
        }
        return index_to_id[index];
    }
}

std::vector<int> HyperCubeTopology::get_receivers(int node_id, HyperCubeTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D hypercube fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_hypercube;
        if (n <= 1) return receivers;
        if (direction == HyperCubeTopology::Direction::Clockwise) {
            int cw_index = (index + 1) % n;
            receivers.push_back(index_to_id.at(cw_index));
        } else {
            int ccw_index = (index - 1 + n) % n;
            receivers.push_back(index_to_id.at(ccw_index));
        }
        return receivers;
    }

    // multi-dimensional case
    const int num_dims = static_cast<int>(dims.size());
    std::vector<int> coords(num_dims);

    int tmp = index;
    for (int d = num_dims - 1; d >= 0; --d) {
        coords[d] = tmp % dims[d];
        tmp /= dims[d];
    }

    if (direction == HyperCubeTopology::Direction::Clockwise) {
        // Clockwise => +1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] + 1) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    } else {
        // Anticlockwise => -1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] - 1 + dims[d]) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    }

    return receivers;
}




int HyperCubeTopology::get_sender(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == HyperCubeTopology::Direction::Anticlockwise) {
        index++;
        if (index == total_nodes_in_hypercube) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_hypercube - 1;
        }
        return index_to_id[index];
    }
}

std::vector<int> HyperCubeTopology::get_senders(int node_id, HyperCubeTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D hypercube fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_hypercube;
        if (n <= 1) return receivers;
        if (direction == HyperCubeTopology::Direction::Anticlockwise) {
            int cw_index = (index + 1) % n;
            receivers.push_back(index_to_id.at(cw_index));
        } else {
            int ccw_index = (index - 1 + n) % n;
            receivers.push_back(index_to_id.at(ccw_index));
        }
        return receivers;
    }

    // multi-dimensional case
    const int num_dims = static_cast<int>(dims.size());
    std::vector<int> coords(num_dims);

    int tmp = index;
    for (int d = num_dims - 1; d >= 0; --d) {
        coords[d] = tmp % dims[d];
        tmp /= dims[d];
    }

    if (direction == HyperCubeTopology::Direction::Anticlockwise) {
        // Clockwise => +1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] + 1) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    } else {
        // Anticlockwise => -1 along each dimension
        for (int d = 0; d < num_dims; ++d) {
            std::vector<int> new_coords = coords;
            new_coords[d] = (new_coords[d] - 1 + dims[d]) % dims[d];
            int new_index = 0;
            for (int k = 0; k < num_dims; ++k) {
                new_index = new_index * dims[k] + new_coords[k];
            }
            receivers.push_back(index_to_id.at(new_index));
        }
    }

    return receivers;
}


int HyperCubeTopology::get_index_in_hypercube() {
    return index_in_hypercube;
}

HyperCubeTopology::Dimension HyperCubeTopology::get_dimension() {
    return dimension;
}

int HyperCubeTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_hypercube();
}

int HyperCubeTopology::get_nodes_in_hypercube() {
    return total_nodes_in_hypercube;
}

bool HyperCubeTopology::is_enabled() {
    assert(offset > 0);
    int tmp_index = index_in_hypercube;
    int tmp_id = id;
    while (tmp_index > 0) {
        tmp_index--;
        tmp_id -= offset;
    }
    if (tmp_id == 0) {
        return true;
    }
    return false;
}
