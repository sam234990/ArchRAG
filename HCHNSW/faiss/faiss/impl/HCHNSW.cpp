#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HCHNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/prefetch.h>

#include <cstddef>
#include <string>

#ifdef __AVX2__
#include <immintrin.h>

#include <limits>
#include <type_traits>
#endif

namespace faiss {

/*******************************************************************
 * HCHNSW structure implementation
 ********************************************************************/

HCHNSW::HCHNSW(int ML, int M, int CL, int VS) : rng(12345) {
    max_level = ML;
    // cross_links = CL;
    level_neighbors.resize(max_level + 1);
    for (int level = 0; level <= max_level; level++) {
        int nn = level == 0 ? M * 2 : M;
        set_nb_neighbors(level, nn);
    }

    // reserve space for some variable for the vectors
    // if (VS <= 0) {
    //     std::cerr << "Error: vector_size should be greater than 0" <<
    //     std::endl;
    // }
    vector_size = VS;
    levels.reserve(VS + 1);
    offsets.reserve(VS + 1);

    leiden_hier_offset.reserve(VS);
    leiden_hier_offset.push_back(0);
    leiden_hier_neighbor.reserve(VS + 1);

    // cross_offsets.reserve(vector_size);
    // cross_offsets.push_back(0);
    cross_neighbors = std::vector<storage_idx_t>(VS, -1);

    first_entry_points_in_level.resize(max_level + 1, -1);
}

void HCHNSW::set_nb_neighbors(int level_no, int n_number) {
    if (level_no > max_level) {
        std::cerr << "Error: level_no should be less than max_level"
                  << std::endl;
    }
    level_neighbors[level_no] = n_number;
}

void HCHNSW::neighbor_range(idx_t no, size_t* begin, size_t* end) const {
    *begin = offsets[no];
    *end = offsets[no + 1];
}

void HCHNSW::set_level(size_t size, const idx_t* level) {
    if (levels.size() < size) {
        levels.resize(size);
    }
    std::copy(level, level + size, levels.begin());

    offsets.reserve(size + 1);
    offsets.push_back(0);
    // levels initial is zero
    for (storage_idx_t i = 0; i < size; i++) {
        int number_neighbor = level_neighbors[levels[i]];
        offsets.push_back(offsets[i] + number_neighbor);
    }
    neighbors.resize(offsets.back(), -1);
}

void HCHNSW::add_leiden_hier_links_sequentially(
        idx_t no,
        const storage_idx_t* leiden_neighbors,
        size_t n) {
    int no_level = levels[no];
    for (size_t i = 0; i < n; i++) {
        int neighbor_level = levels[leiden_neighbors[i]];
        if ((neighbor_level + 1) != no_level) {
            std::cerr << "Error: neighbor " << leiden_neighbors[i]
                      << " is not the parent of " << no << std::endl;
        }
    }
    leiden_hier_neighbor.insert(
            leiden_hier_neighbor.end(), leiden_neighbors, leiden_neighbors + n);

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
        // TODO: error with none vector initial
        std::vector<storage_idx_t> max_level_vector;
        max_level_vector.reserve(levels.size());
        for (storage_idx_t i = 0; i < levels.size(); i++) {
            if (levels[i] == max_level)
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
        if (input == -1) {
            std::vector<storage_idx_t> level_vector;
            level_vector.reserve(levels.size());
            for (storage_idx_t i = 0; i < levels.size(); i++) {
                int level = -1;
                get_level(i, &level);
                if (level == max_level)
                    level_vector.push_back(i);
            }
            int random_index = rng.rand_int(level_vector.size());
            input = level_vector[random_index];
        }
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
        bool keep_max_size_level0) {
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
        bool keep_max_size_level0) {
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
        bool keep_max_size_level) {
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
        bool keep_max_size_level) {
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

void HCHNSW::add_remain_cross_link(
        DistanceComputer& ptdis,
        storage_idx_t pt_id,
        int level) {
    if (cross_neighbors[pt_id] != -1) {
        // the cross neighbor of this vector is already set
        return;
    }
    storage_idx_t next_level_entry = -1;
    get_first_entry_points_in_level(level - 1, &next_level_entry, -1);
    if (next_level_entry == -1) {
        // no entry in the next level
        std::cerr << "Error: no entry in the " << level - 1
                  << " level in add_remain_cross_link" << std::endl;
    }
    float d_nearest = ptdis(next_level_entry);
    greedy_update_nearest(*this, ptdis, level - 1, next_level_entry, d_nearest);
    cross_neighbors[pt_id] = next_level_entry;
}

/**************************************************************
 * HCHNSW search
 **************************************************************/

namespace {

using MinimaxHeap = HCHNSW::MinimaxHeap;
using Node = HCHNSW::Node;
using C = HCHNSW::C;

int search_from_candidates(
        const HCHNSW& hchnsw,
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HCHNSWStats& stats,
        int level,
        int nres_in = 0,
        const SearchParametersHCHNSW* params = nullptr) {
    // selects a version
    const bool reference_version = true;

    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
    bool do_dis_check = params ? params->check_relative_distance
                               : hchnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hchnsw.efSearch;
    const IDSelector* sel = params ? params->sel : nullptr;

    C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hchnsw.neighbor_range(v0, &begin, &end);

        // select a version, based on a flag
        if (reference_version) {
            // a reference version
            for (size_t j = begin; j < end; j++) {
                int v1 = hchnsw.neighbors[j];
                if (v1 < 0)
                    break;
                if (vt.get(v1)) {
                    continue;
                }
                vt.set(v1);
                ndis++;
                float d = qdis(v1);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            nres += 1;
                        }
                    }
                }

                candidates.push(v1, d);
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t j = begin; j < end; j++) {
                int v1 = hchnsw.neighbors[j];
                if (v1 < 0)
                    break;

                prefetch_L2(vt.visited.data() + v1);
                jmax += 1;
            }

            int counter = 0;
            size_t saved_j[4];

            ndis += jmax - begin;
            threshold = res.threshold;

            auto add_to_heap = [&](const size_t idx, const float dis) {
                if (!sel || sel->is_member(idx)) {
                    if (dis < threshold) {
                        if (res.add_result(dis, idx)) {
                            threshold = res.threshold;
                            nres += 1;
                        }
                    }
                }
                candidates.push(idx, dis);
            };

            for (size_t j = begin; j < jmax; j++) {
                int v1 = hchnsw.neighbors[j];

                bool vget = vt.get(v1);
                vt.set(v1);
                saved_j[counter] = v1;
                counter += vget ? 0 : 1;

                if (counter == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            saved_j[0],
                            saved_j[1],
                            saved_j[2],
                            saved_j[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        add_to_heap(saved_j[id4], dis[id4]);
                    }

                    counter = 0;
                }
            }

            for (size_t icnt = 0; icnt < counter; icnt++) {
                float dis = qdis(saved_j[icnt]);
                add_to_heap(saved_j[icnt], dis);
            }
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

std::priority_queue<HCHNSW::Node> search_from_candidate_unbounded(
        const HCHNSW& hchnsw,
        const Node& node,
        DistanceComputer& qdis,
        int ef,
        VisitedTable* vt,
        HCHNSWStats& stats) {
    // selects a version
    const bool reference_version = false;

    int ndis = 0;
    std::priority_queue<Node> top_candidates;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hchnsw.neighbor_range(v0, &begin, &end);

        if (reference_version) {
            // reference version
            for (size_t j = begin; j < end; ++j) {
                int v1 = hchnsw.neighbors[j];

                if (v1 < 0) {
                    break;
                }
                if (vt->get(v1)) {
                    continue;
                }

                vt->set(v1);

                float d1 = qdis(v1);
                ++ndis;

                if (top_candidates.top().first > d1 ||
                    top_candidates.size() < ef) {
                    candidates.emplace(d1, v1);
                    top_candidates.emplace(d1, v1);

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            }
        } else {
            // a faster version

            // the following version processes 4 neighbors at a time
            size_t jmax = begin;
            for (size_t j = begin; j < end; j++) {
                int v1 = hchnsw.neighbors[j];
                if (v1 < 0)
                    break;

                prefetch_L2(vt->visited.data() + v1);
                jmax += 1;
            }

            int counter = 0;
            size_t saved_j[4];

            ndis += jmax - begin;

            auto add_to_heap = [&](const size_t idx, const float dis) {
                if (top_candidates.top().first > dis ||
                    top_candidates.size() < ef) {
                    candidates.emplace(dis, idx);
                    top_candidates.emplace(dis, idx);

                    if (top_candidates.size() > ef) {
                        top_candidates.pop();
                    }
                }
            };

            for (size_t j = begin; j < jmax; j++) {
                int v1 = hchnsw.neighbors[j];

                bool vget = vt->get(v1);
                vt->set(v1);
                saved_j[counter] = v1;
                counter += vget ? 0 : 1;

                if (counter == 4) {
                    float dis[4];
                    qdis.distances_batch_4(
                            saved_j[0],
                            saved_j[1],
                            saved_j[2],
                            saved_j[3],
                            dis[0],
                            dis[1],
                            dis[2],
                            dis[3]);

                    for (size_t id4 = 0; id4 < 4; id4++) {
                        add_to_heap(saved_j[id4], dis[id4]);
                    }

                    counter = 0;
                }
            }

            for (size_t icnt = 0; icnt < counter; icnt++) {
                float dis = qdis(saved_j[icnt]);
                add_to_heap(saved_j[icnt], dis);
            }
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

// just used as a lower bound for the minmaxheap, but it is set for heap search
int extract_k_from_ResultHandler(ResultHandler<C>& res) {
    using RH = HeapBlockResultHandler<C>;
    if (auto hres = dynamic_cast<RH::SingleResultHandler*>(&res)) {
        return hres->k;
    }
    return 1;
}

} // namespace

HCHNSWStats HCHNSW::search(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        VisitedTable& vt,
        const SearchParametersHCHNSW* params) const {
    HCHNSWStats stats;
    if (first_entry_points_in_level[max_level] == -1) {
        return stats;
    }
    if (params->search_level == -1) {
        std::cerr << "Error: search_level is not set in search" << std::endl;
    }

    int k = extract_k_from_ResultHandler(res);

    bool bounded_queue =
            params ? params->bounded_queue : this->search_bounded_queue;

    //  greedy search on upper levels
    storage_idx_t nearest = first_entry_points_in_level[max_level];
    float d_nearest = qdis(nearest);

    if (params->entry_point != -1) {
        storage_idx_t entry_point = params->entry_point;
        if (levels[entry_point] != params->search_level + 1) {
            std::cerr << "Error: entry point " << entry_point
                      << " is not in the level above search level" << std::endl;
        }
        storage_idx_t next_level_entry = cross_neighbors[entry_point];
        nearest = next_level_entry;
        d_nearest = qdis(nearest);
    } else {
        for (int level = max_level; level > params->search_level; level--) {
            // 1. find the nearest vector (clnv) in current level
            HCHNSWStats local_stats = greedy_update_nearest(
                    *this, qdis, level, nearest, d_nearest);
            stats.combine(local_stats);

            // 2. find the linked vector of clnv in the next level
            storage_idx_t next_level_entry = cross_neighbors[nearest];
            d_nearest = qdis(next_level_entry);
            nearest = next_level_entry;
        }
    }

    // find the true nearest entry in pt_level
    HCHNSWStats local_stats = greedy_update_nearest(
            *this, qdis, params->search_level, nearest, d_nearest);
    stats.combine(local_stats);

    int ef = std::max(params ? params->efSearch : efSearch, k);
    if (bounded_queue) {
        MinimaxHeap candidates(ef);
        candidates.push(nearest, d_nearest);

        search_from_candidates(
                *this, qdis, res, candidates, vt, stats, 0, 0, params);

    } else {
        std::priority_queue<Node> top_candidates =
                search_from_candidate_unbounded(
                        *this, Node(d_nearest, nearest), qdis, ef, &vt, stats);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        while (!top_candidates.empty()) {
            float d;
            storage_idx_t label;
            std::tie(d, label) = top_candidates.top();
            res.add_result(d, label);
            top_candidates.pop();
        }
    }
    vt.advance();

    return stats;
}

// this function should be not used in HCHNSW
void HCHNSW::search_level_n(
        DistanceComputer& qdis,
        ResultHandler<C>& res,
        int search_level,
        idx_t nprobe,
        const storage_idx_t* nearest_i,
        const float* nearest_d,
        int search_type,
        HCHNSWStats& search_stats,
        VisitedTable& vt,
        const SearchParametersHCHNSW* params) const {
    const HCHNSW& hchnsw = *this;
    auto efSearch = params ? params->efSearch : hchnsw.efSearch;
    int k = extract_k_from_ResultHandler(res);

    if (search_type == 1) {
        int nres = 0;

        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;

            if (vt.get(cj))
                continue;

            int candidates_size = std::max(efSearch, k);
            MinimaxHeap candidates(candidates_size);

            candidates.push(cj, nearest_d[j]);

            nres = search_from_candidates(
                    hchnsw,
                    qdis,
                    res,
                    candidates,
                    vt,
                    search_stats,
                    0,
                    nres,
                    params);
            nres = std::min(nres, candidates_size);
        }
    } else if (search_type == 2) {
        int candidates_size = std::max(efSearch, int(k));
        candidates_size = std::max(candidates_size, int(nprobe));

        MinimaxHeap candidates(candidates_size);
        for (int j = 0; j < nprobe; j++) {
            storage_idx_t cj = nearest_i[j];

            if (cj < 0)
                break;
            candidates.push(cj, nearest_d[j]);
        }

        search_from_candidates(
                hchnsw,
                qdis,
                res,
                candidates,
                vt,
                search_stats,
                search_level,
                0,
                params);
    }
};

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
