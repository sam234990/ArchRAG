#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/IndexHCHNSW.h>
#include <faiss/IndexHNSW.h>

using idx_t = faiss::idx_t;

void cout_result(
        int nq,
        int k,
        idx_t* I,
        float* D,
        std::vector<idx_t> original_indices_level2);

int main() {
    int d = 64;    // dimension
    int nb = 1000; // database size
    int nq = 100;  // nb of queries
    int max_level = 4;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    std::vector<int> level(nb, -1);
    std::vector<int> level_count(max_level + 1, 0);
    std::lognormal_distribution<> distrib_log(0.0, 1.0); // 参数可以调整

    for (int i = 0; i < nb; i++) {
        // 生成对数正态分布的随机数，并将其转换到 0 到 max_level 的范围
        double random_value = distrib_log(rng);
        int assigned_level = std::min(
                max_level,
                static_cast<int>(std::log(random_value + 1) / std::log(2.0)));

        assigned_level = std::max(0, std::min(assigned_level, max_level));

        // 将 level 赋值
        level[i] = assigned_level;
        level_count[assigned_level]++;
    }

    // 输出每个 level 的点的数量
    for (int lvl = 0; lvl <= max_level; lvl++) {
        std::cout << "Level " << lvl << ": " << level_count[lvl] << " points"
                  << std::endl;
    }

    int k = 4;
    int search_level = 2;

    // Count how many points are at the desired level (level 2)
    int count_level2 = 0;
    for (int i = 0; i < nb; i++) {
        if (level[i] == search_level) {
            count_level2++;
        }
    }

    // Allocate memory for xb_level2
    float* xb_level2 = new float[d * count_level2];
    // Vector to store original indices of vectors at level 2
    std::vector<idx_t> original_indices_level2;

    // Copy vectors at level 2 into xb_level2
    int idx = 0;
    for (int i = 0; i < nb; i++) {
        if (level[i] == search_level) {
            // Copy the vector xb[d * i : d * (i + 1)] to xb_level2
            std::copy(xb + d * i, xb + d * (i + 1), xb_level2 + d * idx);
            // Record the original index
            original_indices_level2.push_back(i);
            idx++;
        }
    }

    std::cout << "Total points at level " << search_level << ": "
              << count_level2 << std::endl;

    faiss::IndexHCHNSWFlat index(d, max_level, 32, 1, nb);
    index.set_vector_level(level);
    index.add(nb, xb);

    faiss::IndexHNSWFlat index_hnsw(d, 32);
    index_hnsw.add(count_level2, xb_level2);

    { // search xq
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        faiss::SearchParametersHCHNSW* params =
                new faiss::SearchParametersHCHNSW();
        params->search_level = 2;
        // TODO: Search algorithm has bugs
        index.search(nq, xq, k, D, I, params);
        cout_result(nq, k, I, D, std::vector<idx_t>());

        delete[] I;
        delete[] D;

        idx_t* I_2 = new idx_t[k * nq];
        float* D_2 = new float[k * nq];
        index_hnsw.search(nq, xq, k, D_2, I_2);
        cout_result(nq, k, I_2, D_2, original_indices_level2);

        delete[] I_2;
        delete[] D_2;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}

void cout_result(
        int nq,
        int k,
        idx_t* I,
        float* D,
        std::vector<idx_t> original_indices_level2) {
    bool ori = false;
    if (original_indices_level2.size() > 0)
        ori = true;
    printf("I=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            if (ori)
                printf("%5zd ", original_indices_level2[I[i * k + j]]);
            else
                printf("%5zd ", I[i * k + j]);
        }
        printf("\n");
    }

    printf("D=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++)
            printf("%5f ", D[i * k + j]);
        printf("\n");
    }
}