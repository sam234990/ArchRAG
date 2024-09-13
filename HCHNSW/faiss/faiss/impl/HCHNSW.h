
#pragma once

#include <queue>
#include <unordered_set>
#include <vector>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

namespace faiss {

struct SearchParametersHCHNSW : SearchParameters {
    int efSearch = 16;
    bool check_relative_distance = true;
    bool bounded_queue = true;

    ~SearchParametersHCHNSW() {}
};

struct HCHNSW {
    /// internal storage of vectors (32 bits: this is expensive)
    using storage_idx_t = int32_t;

    // for now we do only these distances
    using C = CMax<float, int64_t>;

    // use for the heap or queue <distance, node_id>
    typedef std::pair<float, storage_idx_t> Node;
    /** Heap structure that allows fast
     */
    struct MinimaxHeap {
        int n;
        int k;
        int nvalid;

        std::vector<storage_idx_t> ids;
        std::vector<float> dis;
        typedef faiss::CMax<float, storage_idx_t> HC;

        explicit MinimaxHeap(int n) : n(n), k(0), nvalid(0), ids(n), dis(n) {}

        void push(storage_idx_t i, float v);

        float max() const;

        int size() const;

        void clear();

        int pop_min(float* vmin_out = nullptr);

        int count_below(float thresh);
    };

    /// to sort pairs of (id, distance) from nearest to fathest or the reverse
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    /// the level of each vector
    std::vector<int> levels;

    /// the neighbors of each vector in each level
    std::vector<int> level_neighbors;

    /// offsets[i] is the offset in the neighbors array where vector i is stored
    /// size ntotal + 1
    std::vector<size_t> offsets;

    /// neighbors[offsets[i]:offsets[i+1]] is the list of neighbors of vector i
    /// for all levels. this is where all storage goes.
    std::vector<storage_idx_t> neighbors;

    std::vector<size_t> leiden_hier_offset;
    std::vector<storage_idx_t> leiden_hier_neighbor;

    /// cross_neighbors[cross_offsets[i]:cross_offsets[i+1]] is the list of
    /// cross edge neighbors of vector i
    // std::vector<size_t> cross_offsets;
    std::vector<storage_idx_t> cross_neighbors;

    /// entry point in the search structure (one of the points with maximum
    /// level
    storage_idx_t entry_point = -1;

    std::vector<storage_idx_t> first_entry_points_in_level;

    faiss::RandomGenerator rng;

    /// maximum level
    int max_level = -1;

    /// expansion factor at construction time
    int efConstruction = 40;

    /// expansion factor at search time
    int efSearch = 16;

    /// number of cross-links per vector
    // int cross_links = -1;

    /// during search: do we check whether the next best distance is good
    /// enough?
    bool check_relative_distance = true;

    /// use bounded queue during exploration
    bool search_bounded_queue = true;

    /// set number of neighbors for a given level
    void set_nb_neighbors(int level_no, int n_number);

    int get_nb_neighbor(int level_no) const;

    void neighbor_range(idx_t no, size_t* begin, size_t* end) const;

    explicit HCHNSW(int ML = 0, int M = 32, int CL = 1, int vector_size);

    void add_leiden_hier_links_sequentially(
            idx_t no,
            const storage_idx_t* neighbors,
            size_t n);

    void get_level(idx_t no, int* level) const;

    void get_max_level_random_entry(idx_t* entry);

    void get_first_entry_points_in_level(
            int level,
            storage_idx_t* entry,
            storage_idx_t input) ;

    void add_links_level_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t* locks,
            VisitedTable& vt,
            bool keep_max_size_level = false);

    void add_with_locks_level(
            DistanceComputer& ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt,
            bool keep_max_size_level = false);


    // TODO: Implement the following functions
    HCHNSWStats search(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            VisitedTable& vt,
            const SearchParametersHCHNSW* params = nullptr) const;

    // search only in the level n
    void search_level_n(
            DistanceComputer& qdis,
            ResultHandler<C>& res,
            int search_level,
            idx_t nprobe,
            const storage_idx_t* nearest_i,
            const float* nearest_d,
            int search_type,
            HCHNSWStats& search_stats,
            VisitedTable& vt,
            const SearchParametersHCHNSW* params = nullptr) const;

    void reset();

    static void shrink_neighbor_list(
            DistanceComputer& qdis,
            std::priority_queue<NodeDistFarther>& input,
            std::vector<NodeDistFarther>& output,
            int max_size,
            bool keep_max_size_level0 = false);
};

struct HCHNSWStats {
    size_t n1 = 0; /// number of vectors searched

    size_t n2 =
            0; // number of queries for which the candidate list is exhausted
    size_t ndis = 0;  /// number of distances computed
    size_t nhops = 0; /// number of hops aka number of edges traversed

    void reset() {
        n1 = n2 = 0;
        ndis = 0;
        nhops = 0;
    }

    void combine(const HCHNSWStats& other) {
        n1 += other.n1;
        n2 += other.n2;
        ndis += other.ndis;
        nhops += other.nhops;
    }
};

// global var that collects them all
FAISS_API extern HCHNSWStats hchnsw_stats;

} // namespace faiss