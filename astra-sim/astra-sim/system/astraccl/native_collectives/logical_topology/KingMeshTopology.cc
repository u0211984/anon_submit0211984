/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/KingMeshTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <iostream>

using namespace std;
using namespace AstraSim;

KingMeshTopology::KingMeshTopology(Dimension dimension, int id, std::vector<int> NPUs)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::KingMesh) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_kmesh = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_kmesh = -1;

    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_kmesh};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_kmesh));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_kmesh};
    }

    for (int i = 0; i < total_nodes_in_kmesh; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_kmesh = i;
        }
    }

    LoggerFactory::get_logger("system::topology::KingMeshTopology")
        ->info("custom kmesh, id: {}, dimension: {} total nodes in kmesh: {} "
               "index in kmesh: {} total nodes in kmesh {}",
               id, name, total_nodes_in_kmesh, index_in_kmesh,
               total_nodes_in_kmesh);

    assert(index_in_kmesh >= 0);
}

KingMeshTopology::KingMeshTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_kmesh,
                           int index_in_kmesh,
                           int offset)
    : BasicLogicalTopology(BasicLogicalTopology::BasicTopology::KingMesh) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::KingMeshTopology")
            ->info("kmesh of node 0, id: {} dimension: {} total nodes in kmesh: "
                   "{} index in kmesh: {} offset: {} total nodes in kmesh: {}",
                   id, name, total_nodes_in_kmesh, index_in_kmesh, offset,
                   total_nodes_in_kmesh);
    }
    this->id = id;
    this->total_nodes_in_kmesh = total_nodes_in_kmesh;
    this->index_in_kmesh = index_in_kmesh;
    this->dimension = dimension;
    this->offset = offset;

    // Initialize dims
    if (dimension == Dimension::Local) {
        dims = {total_nodes_in_kmesh};
    } else if (dimension == Dimension::Horizontal || dimension == Dimension::Vertical) {
        int dim_size = static_cast<int>(std::sqrt(total_nodes_in_kmesh));
        dims = {dim_size, dim_size};
    } else {
        dims = {total_nodes_in_kmesh};
    }

    id_to_index[id] = index_in_kmesh;
    index_to_id[index_in_kmesh] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_kmesh - 1; i++) {
        tmp = get_receiver_homogeneous(tmp, KingMeshTopology::Direction::Clockwise,
                                       offset);
    }
}


int KingMeshTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == KingMeshTopology::Direction::Clockwise) {
        int receiver = node_id + offset;
        if (index == total_nodes_in_kmesh - 1) {
            receiver -= (total_nodes_in_kmesh * offset);
            index = 0;
        } else {
            index++;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::KingMeshTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_kmesh {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_kmesh, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    } else {
        int receiver = node_id - offset;
        if (index == 0) {
            receiver += (total_nodes_in_kmesh * offset);
            index = total_nodes_in_kmesh - 1;
        } else {
            index--;
        }
        if (receiver < 0) {
            LoggerFactory::get_logger("system::topology::KingMeshTopology")
                ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                           "id: {}, offset: {}, index_in_kmesh {} receiver {}",
                           name, id, name, index, node_id, offset,
                           index_in_kmesh, receiver);
        }
        assert(receiver >= 0);
        id_to_index[receiver] = index;
        index_to_id[index] = receiver;
        return receiver;
    }
}

int KingMeshTopology::get_receiver(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == KingMeshTopology::Direction::Clockwise) {
        index++;
        if (index == total_nodes_in_kmesh) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_kmesh - 1;
        }
        return index_to_id[index];
    }
}

std::vector<int> KingMeshTopology::get_receivers(int node_id, KingMeshTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D kmesh fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_kmesh;
        if (n <= 1) return receivers;
        if (direction == KingMeshTopology::Direction::Clockwise) {
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

    if (direction == KingMeshTopology::Direction::Clockwise) {
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




int KingMeshTopology::get_sender(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    if (direction == KingMeshTopology::Direction::Anticlockwise) {
        index++;
        if (index == total_nodes_in_kmesh) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_kmesh - 1;
        }
        return index_to_id[index];
    }
}

std::vector<int> KingMeshTopology::get_senders(int node_id, KingMeshTopology::Direction direction) const {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index.at(node_id);

    std::vector<int> receivers;

    // 1D kmesh fallback
    if (dims.empty() || dims.size() == 1) {
        int n = total_nodes_in_kmesh;
        if (n <= 1) return receivers;
        if (direction == KingMeshTopology::Direction::Anticlockwise) {
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

    if (direction == KingMeshTopology::Direction::Anticlockwise) {
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


int KingMeshTopology::get_index_in_kmesh() {
    return index_in_kmesh;
}

KingMeshTopology::Dimension KingMeshTopology::get_dimension() {
    return dimension;
}

int KingMeshTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_kmesh();
}

int KingMeshTopology::get_nodes_in_kmesh() {
    return total_nodes_in_kmesh;
}

bool KingMeshTopology::is_enabled() {
    assert(offset > 0);
    int tmp_index = index_in_kmesh;
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
