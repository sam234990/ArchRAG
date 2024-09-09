
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

HCHNSW::HCHNSW(int ML = 0, int M = 32, int CL = 1) : rng(12345) {
    max_level = ML;
    cross_links = CL;
    level_neighbors.resize(max_level + 1);
    for (int level = 0; level <= max_level; level++) {
        int nn = level == 0 ? M * 2 : M;
        set_nb_neighbors(level, nn);
    }
}

void HCHNSW::set_nb_neighbors(int level_no, int n) {
        if (level_no >= max_level) {
        max_level = level_no + 1;
    }
    if (level_no >= level_neighbors.size()) {
        level_neighbors.resize(level_no + 1);
    }
    level_neighbors[level_no] = n;
}

void HCHNSW::neighbor_range(id_t no, size_t* begin, size_t* end) const {
    *begin = offsets[no];
    *end = offsets[no + 1];
}

} // namespace faiss
