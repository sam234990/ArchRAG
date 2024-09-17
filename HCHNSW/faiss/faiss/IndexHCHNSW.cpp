/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHCHNSW.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>

#include <sys/stat.h>
#include <sys/types.h>
#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

using MinimaxHeap = HCHNSW::MinimaxHeap;
using storage_idx_t = HCHNSW::storage_idx_t;
using NodeDistFarther = HCHNSW::NodeDistFarther;

HCHNSWStats hchnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

// add new n vector(n, x) to the existing index with ntotoal vectors(n0)
void hchnsw_add_vertices(
        IndexHCHNSW& index_hchnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose) {
    size_t d = index_hchnsw.d;
    HCHNSW& hchnsw = index_hchnsw.hchnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hchnsw_add_vertices: adding %zd elements on top of %zd ",
               n,
               n0);
    }

    if (n == 0) {
        return;
    }

    int max_level = hchnsw.max_level;

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hchnsw.levels[pt_id];
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hchnsw.levels[pt_id];
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hchnsw.d * hchnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hchnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hchnsw.add_with_locks_level(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hchnsw.keep_max_size_level);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }

            // check the cross links in pt_level+1 after add in pt_level
            // i1-- the start of the pt+1 level and the end of the pt_level
            if (pt_level != (hist.size() - 1)) {
                int start = i1;
                int end = i1 + hist[pt_level + 1];
                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hchnsw.storage));
                for (int i = start; i < end; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);
                    hchnsw.add_remain_cross_link(*dis, pt_id, pt_level + 1);
                }
            }

            i1 = i0;
        }
        if (index_hchnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHCHNSW implementation
 **************************************************************/

IndexHCHNSW::IndexHCHNSW(
        int d,
        int ML,
        int M,
        int CL,
        int vector_size,
        MetricType metric)
        : Index(d, metric), hchnsw(ML, M, CL, vector_size) {}

IndexHCHNSW::IndexHCHNSW(Index* storage, int ML, int M, int CL, int vector_size)
        : Index(storage->d, storage->metric_type),
          hchnsw(ML, M, CL, vector_size),
          storage(storage) {}

IndexHCHNSW::~IndexHCHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHCHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHCHNSWFlat (or variants) instead of IndexHCHNSW directly");
    // hchnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hchnsw_search(
        const IndexHCHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHCHNSWFlat (or variants) "
            "instead of IndexHCHNSW directly");
    const SearchParametersHCHNSW* params = nullptr;
    const HCHNSW& hchnsw = index->hchnsw;

    int efSearch = hchnsw.efSearch;
    int search_level = -1;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHCHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
        search_level = params->search_level;
    }
    if (search_level == -1) {
        std::cerr << "search_level is not set, using default value 0"
                  << std::endl;
        search_level = 0;
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hchnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

// #pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

// #pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HCHNSWStats stats = hchnsw.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hchnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHCHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HCHNSW::C>;
    RH bres(n, distances, labels, k);

    hchnsw_search(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHCHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HCHNSW::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    hchnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

// add new n vector to the existing index with ntotoal vectors
void IndexHCHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHCHNSWFlat (or variants) instead of IndexHCHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hchnsw_add_vertices(*this, n0, n, x, verbose);
}

void IndexHCHNSW::construct_leiden_edge(
        const std::vector<int>& offset,
        const storage_idx_t* edges,
        size_t nedge) {
    for (int i = 0; i < offset.size(); i++) {
        int number_of_neighbors = offset[i + 1] - offset[i];
        // 获取对应的邻居列表
        const storage_idx_t* neighbor =
                reinterpret_cast<const storage_idx_t*>(&edges[offset[i]]);
        hchnsw.add_leiden_hier_links_sequentially(
                i, neighbor, number_of_neighbors);
    }
};

void IndexHCHNSW::set_vector_level(const std::vector<int>& level) {
    if (hchnsw.levels.size() < level.size()) {
        hchnsw.levels.resize(level.size());
    }
    for (int i = 0; i < level.size(); i++) {
        hchnsw.levels[i] = level[i];
    }
}

void IndexHCHNSW::reset() {
    hchnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHCHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHCHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hchnsw.neighbor_range(i, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hchnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HCHNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hchnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hchnsw.neighbors[j] = -1;
            }
        }
    }
}

DistanceComputer* IndexHCHNSW::get_distance_computer() const {
    return storage->get_distance_computer();
}

/**************************************************************
 * IndexHCHNSWFlat implementation
 **************************************************************/

IndexHCHNSWFlat::IndexHCHNSWFlat() {
    is_trained = true;
}

IndexHCHNSWFlat::IndexHCHNSWFlat(
        int d,
        int ML,
        int M,
        int CL,
        int vector_size,
        MetricType metric)
        : IndexHCHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  ML,
                  M,
                  CL,
                  vector_size) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHCHNSWPQ implementation
 **************************************************************/

IndexHCHNSWPQ::IndexHCHNSWPQ() = default;

IndexHCHNSWPQ::IndexHCHNSWPQ(
        int d,
        int ML,
        int M,
        int CL,
        int vector_size,
        int pq_m,
        int pq_nbits,
        MetricType metric)
        : IndexHCHNSW(
                  new IndexPQ(d, pq_m, pq_nbits, metric),
                  ML,
                  M,
                  CL,
                  vector_size) {
    own_fields = true;
    is_trained = false;
}

void IndexHCHNSWPQ::train(idx_t n, const float* x) {
    IndexHCHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHCHNSWSQ implementation
 **************************************************************/

IndexHCHNSWSQ::IndexHCHNSWSQ(
        int d,
        int ML,
        int M,
        int CL,
        int vector_size,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric)
        : IndexHCHNSW(
                  new IndexScalarQuantizer(d, qtype, metric),
                  ML,
                  M,
                  CL,
                  vector_size) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHCHNSWSQ::IndexHCHNSWSQ() = default;

} // namespace faiss
