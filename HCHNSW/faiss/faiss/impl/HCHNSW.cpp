
#include <faiss/impl/HCHNSW.h>

#include <cstddef>
#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/prefetch.h>

#include <faiss/impl/platform_macros.h>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

namespace faiss {

/*******************************************************************
 * HCHNSW structure implementation
 ********************************************************************/

HCHNSW::HCHNSW(int ML = 0, int M = 32, int CL = 1, int vector_size)
        : rng(12345) {
    max_level = ML;
    // cross_links = CL;
    level_neighbors.resize(max_level + 1);
    for (int level = 0; level <= max_level; level++) {
        int nn = level == 0 ? M * 2 : M;
        set_nb_neighbors(level, nn);
    }

    // reserve space for some variable for the vectors

    levels.reserve(vector_size + 1);
    offsets.reserve(vector_size + 1);
    offsets.push_back(0);
    neighbors.reserve(vector_size * M * 2);

    leiden_hier_offset.reserve(vector_size);
    leiden_hier_offset.push_back(0);
    leiden_hier_neighbor.reserve(vector_size + 1);

    // cross_offsets.reserve(vector_size);
    // cross_offsets.push_back(0);
    cross_neighbors = std::vector<storage_idx_t>(M, -1);

    first_entry_points_in_level.resize(max_level + 1, -1);
}

void HCHNSW::set_nb_neighbors(int level_no, int n_number) {
    if (level_no >= max_level) {
        max_level = level_no + 1;
    }
    if (level_no >= level_neighbors.size()) {
        level_neighbors.resize(level_no + 1);
    }
    level_neighbors[level_no] = n_number;
}

void HCHNSW::neighbor_range(idx_t no, size_t* begin, size_t* end) const {
    *begin = offsets[no];
    *end = offsets[no + 1];
}

void HCHNSW::add_leiden_hier_links_sequentially(
        idx_t no,
        const storage_idx_t* neighbors,
        size_t n) {
    leiden_hier_neighbor.insert(
            leiden_hier_neighbor.end(), neighbors, neighbors + n);

    int next_offset = leiden_hier_neighbor.size();
    leiden_hier_offset.push_back(next_offset);
}

void HCHNSW::get_level(idx_t no, int* level) const {
    *level = 0;
    if (no < levels.size()) {
        *level = levels[no];
    } else {
        std::cerr << "Error: Index " << no << " out of range in get_level"
                  << std::endl;
    }
}

void HCHNSW::get_max_level_random_entry(idx_t* entry) {
    if (first_entry_points_in_level[max_level] != -1) {
        *entry = first_entry_points_in_level[max_level];
        return;
    } else {
        std::vector<storage_idx_t> max_level_vector;
        max_level_vector.reserve(levels.size());
        for (storage_idx_t i = 0; i < levels.size(); i++) {
            int level = -1;
            get_level(i, &level);
            if (level == max_level)
                max_level_vector.push_back(i);
        }
        int random_index = rng.rand_int(max_level_vector.size());
        first_entry_points_in_level[max_level] = max_level_vector[random_index];
        *entry = first_entry_points_in_level[max_level];
    }
}

void HCHNSW::get_first_entry_points_in_level(
        int level,
        storage_idx_t* entry,
        storage_idx_t input) {
    if (first_entry_points_in_level[level] == -1) {
        first_entry_points_in_level[level] = input;
        *entry = input;
        return;
    } else {
        *entry = first_entry_points_in_level[level];
    }
}

int HCHNSW::get_nb_neighbor(int level_no) const {
    return level_neighbors[level_no];
};

void HCHNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
    leiden_hier_offset.clear();
    leiden_hier_neighbor.clear();
    // cross_links = -1;
    // cross_offsets.clear();
    cross_neighbors.clear();
}

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void HCHNSW::shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistFarther>& input,
        std::vector<NodeDistFarther>& output,
        int max_size,
        bool keep_max_size_level0 = false) {
    // This prevents number of neighbors at
    // level 0 from being shrunk to less than 2 * M.
    // This is essential in making sure
    // `faiss::gpu::GpuIndexCagra::copyFrom(IndexHNSWCagra*)` is functional
    std::vector<NodeDistFarther> outsiders;

    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }

        if (good) { // should be added
            output.push_back(v1);
            if (output.size() >= max_size) {
                return;
            }
        } else if (keep_max_size_level0) {
            outsiders.push_back(v1);
        }
    }
    size_t idx = 0;
    while (keep_max_size_level0 && (output.size() < max_size) &&
           (idx < outsiders.size())) {
        output.push_back(outsiders[idx++]);
    }
}

namespace {

using storage_idx_t = HCHNSW::storage_idx_t;
using NodeDistCloser = HCHNSW::NodeDistCloser;
using NodeDistFarther = HCHNSW::NodeDistFarther;

/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& resultSet1,
        int max_size,
        bool keep_max_size_level0 = false) {
    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    HCHNSW::shrink_neighbor_list(
            qdis, resultSet, returnlist, max_size, keep_max_size_level0);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }
}

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(
        HCHNSW& hchnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest,
        int level,
        bool keep_max_size_level0 = false) {
    size_t begin, end;
    hchnsw.neighbor_range(src, &begin, &end);
    if (hchnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while (i > begin) {
            if (hchnsw.neighbors[i - 1] != -1)
                break;
            i--;
        }
        hchnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hchnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    shrink_neighbor_list(qdis, resultSet, end - begin, keep_max_size_level0);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hchnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while (i < end) {
        hchnsw.neighbors[i++] = -1;
    }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(
        HCHNSW& hchnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        int level,
        VisitedTable& vt) {
    // selects a version
    const bool reference_version = true;

    // top is nearest candidate
    // because we want to keep the queue sorted by distance and we start from
    // the nearest vector, we use NodeDistFarther here.
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hchnsw.neighbor_range(currNode, &begin, &end);

        // select a version, based on a flag
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                storage_idx_t nodeId = hchnsw.neighbors[i];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId)) // visited
                    continue;
                vt.set(nodeId);

                float dis = qdis(nodeId);
                NodeDistFarther evE1(dis, nodeId);

                if (results.size() < hchnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, nodeId);
                    candidates.emplace(dis, nodeId);
                    if (results.size() > hchnsw.efConstruction) {
                        results.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                if (results.size() < hchnsw.efConstruction ||
                    results.top().d > dis) {
                    results.emplace(dis, idx);
                    candidates.emplace(dis, idx);
                    if (results.size() > hchnsw.efConstruction) {
                        results.pop();
                    }
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nodeId = hchnsw.neighbors[j];
                if (nodeId < 0)
                    break;
                if (vt.get(nodeId)) {
                    continue;
                }
                vt.set(nodeId);

                buffered_ids[n_buffered] = nodeId;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (size_t icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }
    }

    vt.advance();
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
HCHNSWStats greedy_update_nearest(
        const HCHNSW& hchnsw,
        DistanceComputer& qdis,
        int level,
        storage_idx_t& nearest,
        float& d_nearest) {
    // selects a version
    const bool reference_version = true;

    HCHNSWStats stats;

    for (;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hchnsw.neighbor_range(nearest, &begin, &end);

        size_t ndis = 0;

        // select a version, based on a flag
        if (reference_version) {
            // a reference version
            for (size_t i = begin; i < end; i++) {
                storage_idx_t v = hchnsw.neighbors[i];
                if (v < 0)
                    break;
                ndis += 1;
                float dis = qdis(v);
                if (dis < d_nearest) {
                    nearest = v;
                    d_nearest = dis;
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            auto update_with_candidate = [&](const storage_idx_t idx,
                                             const float dis) {
                if (dis < d_nearest) {
                    nearest = idx;
                    d_nearest = dis;
                }
            };

            int n_buffered = 0;
            storage_idx_t buffered_ids[4];

            for (size_t j = begin; j < end; j++) {
                storage_idx_t v = hchnsw.neighbors[j];
                if (v < 0)
                    break;
                ndis += 1;

                buffered_ids[n_buffered] = v;
                n_buffered += 1;

                if (n_buffered == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            buffered_ids[0],
                            buffered_ids[1],
                            buffered_ids[2],
                            buffered_ids[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        update_with_candidate(buffered_ids[id4], dis[id4]);
                    }

                    n_buffered = 0;
                }
            }

            // process leftovers
            for (size_t icnt = 0; icnt < n_buffered; icnt++) {
                float dis = qdis(buffered_ids[icnt]);
                update_with_candidate(buffered_ids[icnt], dis);
            }
        }

        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
}

} // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void HCHNSW::add_links_level_starting_from(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        int level,
        omp_lock_t* locks,
        VisitedTable& vt,
        bool keep_max_size_level = false) {
    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            *this, ptdis, link_targets, nearest, d_nearest, level, vt);

    // but we can afford only this many neighbors
    int M = get_nb_neighbor(level);

    ::faiss::shrink_neighbor_list(ptdis, link_targets, M, keep_max_size_level);

    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id, level, keep_max_size_level);
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id, level, keep_max_size_level);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

void HCHNSW::add_with_locks_level(
        DistanceComputer& ptdis,
        int pt_level,
        int pt_id,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt,
        bool keep_max_size_level = false) {
    //  greedy search on upper levels

    storage_idx_t nearest;

    // TODO: without para nearest
#pragma omp critical
    {
        idx_t max_entry;
        get_max_level_random_entry(&max_entry);
        nearest = max_entry;
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for (; level > pt_level; level--) {
        // 1. find the nearest vector (clnv) in current level
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);

        // 2. find the linked vector of clnv in the next level
        // and update the nearest vector and distance

        // 2.1 find the linked vector of clnv in the next level
        storage_idx_t next_level_entry = cross_neighbors[nearest];

        if (level == (pt_level + 1)) {
            // 3. in the pt+1 level, need to update the cross level and the
            // nearest ;

            if (next_level_entry == -1) {
                // 3.1 pt+1 vector do not have cross neighbor, create one and
                // update the nearest in ptlevel
                cross_neighbors[nearest] = pt_id;
                // use the first entry as the nearest
                get_first_entry_points_in_level(
                        pt_level, &next_level_entry, pt_id);
            } else {
                // 3.1 pt+1 have cross neighbor
                // check and update the cross link
                float dis_n_q = ptdis(nearest);
                float dis_n_next_level_entry =
                        ptdis.symmetric_dis(nearest, next_level_entry);
                if (dis_n_q < dis_n_next_level_entry) {
                    cross_neighbors[nearest] = pt_id;
                }
            }
        }
        // 2.2 update the nearest in the next level
        d_nearest = ptdis(next_level_entry);
        nearest = next_level_entry;
    }

    // find the true nearest entry in pt_level
    greedy_update_nearest(*this, ptdis, pt_level, nearest, d_nearest);

    // only add links in the pt_level
    add_links_level_starting_from(
            ptdis,
            pt_id,
            nearest,
            d_nearest,
            pt_level,
            locks.data(),
            vt,
            keep_max_size_level);
    omp_unset_lock(&locks[pt_id]);
} // namespace faiss


/**************************************************************
 * MinimaxHeap
 **************************************************************/

void HCHNSW::MinimaxHeap::push(storage_idx_t i, float v) {
    if (k == n) {
        if (v >= dis[0])
            return;
        if (ids[0] != -1) {
            --nvalid;
        }
        faiss::heap_pop<HC>(k--, dis.data(), ids.data());
    }
    faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
    ++nvalid;
}

float HCHNSW::MinimaxHeap::max() const {
    return dis[0];
}

int HCHNSW::MinimaxHeap::size() const {
    return nvalid;
}

void HCHNSW::MinimaxHeap::clear() {
    nvalid = k = 0;
}

// baseline non-vectorized version
int HCHNSW::MinimaxHeap::pop_min(float* vmin_out) {
    assert(k > 0);
    // returns min. This is an O(n) operation
    int i = k - 1;
    while (i >= 0) {
        if (ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = dis[i];
    i--;
    while (i >= 0) {
        if (ids[i] != -1 && dis[i] < vmin) {
            vmin = dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = ids[imin];
    ids[imin] = -1;
    --nvalid;

    return ret;
}

int HCHNSW::MinimaxHeap::count_below(float thresh) {
    int n_below = 0;
    for (int i = 0; i < k; i++) {
        if (dis[i] < thresh) {
            n_below++;
        }
    }

    return n_below;
}
} // namespace faiss
