import numpy as np
import math
import faiss

max_level = 4
search_level = 3


def make_test_data():
    dim = 128
    num_elements = 10000
    num_queries = 100

    # 生成数据库和查询数据
    data = np.float32(np.random.random((num_elements, dim)))
    data[:, 0] += np.arange(num_elements) / 1000.0  # 按照索引对第一维进行微调
    data = np.ascontiguousarray(data)  # 确保数据是 C-contiguous 的

    queries = np.float32(np.random.random((num_queries, dim)))
    queries[:, 0] += np.arange(num_queries) / 1000.0  # 同样处理查询数据
    queries = np.ascontiguousarray(queries)  # 确保 queries 是 C-contiguous 的

    # 分配层级信息 (使用对数正态分布)
    levels = np.zeros(num_elements, dtype=int)
    level_counts = np.zeros(max_level + 1, dtype=int)

    # 使用对数正态分布分配层级
    random_values = np.random.lognormal(mean=0.0, sigma=1.0, size=num_elements)
    for i in range(num_elements):
        assigned_level = min(
            max_level, int(math.log(random_values[i] + 1) / math.log(2.0))
        )
        levels[i] = max(0, assigned_level)
        level_counts[assigned_level] += 1

    # 输出每个层级的点数
    for lvl in range(max_level + 1):
        print(f"Level {lvl}: {level_counts[lvl]} points")

    # 筛选出在 search_level 的元素
    selected_level_mask = levels == search_level
    data_level2 = data[selected_level_mask]
    original_indices_level2 = np.where(selected_level_mask)[0]

    print(f"Total points at level {search_level}: {len(data_level2)}")

    # 返回生成的数据、查询数据、维度、以及层级信息
    M = 32
    efSearch = 100
    efConstruction = 100

    return (
        data,
        queries,
        data_level2,
        original_indices_level2,
        dim,
        M,
        efSearch,
        efConstruction,
        levels,
    )


def test_hnsw_hchnsw(
    data,
    queries,
    data_level2,
    original_indices_level2,
    dim,
    M,
    efSearch,
    efConstruction,
    levels,
):
    # build hnsw
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(data_level2)  # 确保数据符合 Faiss 的要求
    
    hnsw_save_path = "./hnsw.index"
    # 保存索引到文件
    faiss.write_index(index, hnsw_save_path)
    
    index_hnsw = faiss.read_index(hnsw_save_path)
    # query hnsw
    top_k = 10
    distance, preds = index_hnsw.search(queries, k=top_k)
    for i in range(3):
        print(
            f"Query number: {i} Top {top_k} results: {[original_indices_level2[p] for p in preds[i]]}"
        )
        print(f"Distances: {distance[i]}")
        print()

    

    index_2 = faiss.IndexHCHNSWFlat(dim, max_level, M, 1, len(data))
    index_2.set_vector_level(levels)

    index_2.hchnsw.efConstruction = efConstruction
    index_2.hchnsw.efSearch = efSearch
    print("Adding data to HCHNSW index")

    index_2.add(data)
    
    hchnsw_save_path = "./hchnsw.index"
    # 保存索引到文件
    faiss.write_index(index_2, hchnsw_save_path)
    
    index_hchnsw = faiss.read_index(hchnsw_save_path)
    

    params = faiss.SearchParametersHCHNSW()
    params.search_level = search_level
    distance, preds = index_hchnsw.search(queries, k=top_k, params=params)
    for i in range(3):
        print(f"Query number: {i} Top {top_k} results: {preds[i]}")
        print(f"Distances: {distance[i]}")
        print()


if __name__ == "__main__":
    (
        data,
        queries,
        data_level2,
        original_indices_level2,
        dim,
        M,
        efSearch,
        efConstruction,
        levels,
    ) = make_test_data()

    test_hnsw_hchnsw(
        data=data,
        queries=queries,
        data_level2=data_level2,
        original_indices_level2=original_indices_level2,
        dim=dim,
        M=M,
        efSearch=efSearch,
        efConstruction=efConstruction,
        levels=levels,
    )
