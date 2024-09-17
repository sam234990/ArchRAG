/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/HCHNSW.h>
#include <faiss/utils/utils.h>

namespace faiss {

struct IndexHCHNSW;

/** The HCHNSW index is a normal random-access index with a HCHNSW
 * link structure built on top.
 * This code is constructed based on the IndexHNSW
 * */

struct IndexHCHNSW : Index {
    typedef HCHNSW::storage_idx_t storage_idx_t;

    // the link structure
    HCHNSW hchnsw;

    // the sequential storage
    bool own_fields = false;
    Index* storage = nullptr;

    // When set to false, level 0 in the knn graph is not initialized.
    // This option is used by GpuIndexCagra::copyTo(IndexHCHNSWCagra*)
    // as level 0 knn graph is copied over from the index built by
    // GpuIndexCagra.
    bool init_level0 = true;

    // When set to true, all neighbors in level 0 are filled up
    // to the maximum size allowed (2 * M). This option is used by
    // IndexHHNSWCagra to create a full base layer graph that is
    // used when GpuIndexCagra::copyFrom(IndexHCHNSWCagra*) is invoked.
    bool keep_max_size_level = true;

    explicit IndexHCHNSW(
            int d = 0,
            int ML = 0,
            int M = 32,
            int CL = 1,
            int vector_size = 0,
            MetricType metric = METRIC_L2);
    explicit IndexHCHNSW(
            Index* storage,
            int ML,
            int M = 32,
            int CL = 1,
            int vector_size = 0);

    ~IndexHCHNSW() override;

    void add(idx_t n, const float* x) override;

    void construct_leiden_edge(
            const std::vector<int>& offset,
            const storage_idx_t* edges,
            size_t nedge);

    void set_vector_level(const std::vector<int>& level);

    /// Trains the storage if needed
    void train(idx_t n, const float* x) override;

    /// entry point for search
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;

    void reset() override;

    void shrink_level_0_neighbors(int size);

    DistanceComputer* get_distance_computer() const override;
};

/*
explicit IndexHCHNSW(
            int d = 0,
            int ML = 0,
            int M = 32,
            int CL = 1,
            int vector_size = 0,
            MetricType metric = METRIC_L2);
    explicit IndexHCHNSW(Index* storage, int ML, int M = 32, int CL = 1);
    */

/** Flat index topped with with a HNSW structure to access elements
 *  more efficiently.
 */

struct IndexHCHNSWFlat : IndexHCHNSW {
    IndexHCHNSWFlat();
    IndexHCHNSWFlat(
            int d = 0,
            int ML = 0,
            int M = 32,
            int CL = 1,
            int vector_size = 0,
            MetricType metric = METRIC_L2);
};

/** PQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHCHNSWPQ : IndexHCHNSW {
    IndexHCHNSWPQ();
    IndexHCHNSWPQ(
            int d,
            int ML,
            int M,
            int CL,
            int vector_size,
            int pq_m,
            int pq_nbits = 8,
            MetricType metric = METRIC_L2);
    void train(idx_t n, const float* x) override;
};

/** SQ index topped with with a HNSW structure to access elements
 *  more efficiently.
 */
struct IndexHCHNSWSQ : IndexHCHNSW {
    IndexHCHNSWSQ();
    IndexHCHNSWSQ(
            int d,
            int ML,
            int M,
            int CL,
            int vector_size,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);
};

} // namespace faiss
