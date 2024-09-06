import numpy as np
import faiss


def make_test_data():
    dim = 128
    num_elements = 10000
    num_queries = 100

    data = np.float32(np.random.random((num_elements, dim)))
    data = np.ascontiguousarray(data)  # 确保数据是 C-contiguous 的
    ids = np.arange(num_elements)
    print(data.shape)
    print(type(data))

    queries = np.float32(np.random.random((num_queries, dim)))
    queries = np.ascontiguousarray(queries)  # 同样确保 queries 是 C-contiguous 的
    M = 32
    efSearch = 100  # number of entry points (neighbors) we use on each layer
    efConstruction = (
        100  # number of entry points used on each layer during construction
    )
    return data, queries, dim, M, efSearch, efConstruction


def test_ori_hnsw(data, queries, dim, M, efSearch, efConstruction):
    # build hnsw
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = efConstruction
    index.hnsw.efSearch = efSearch
    index.add(data)  # 确保数据符合 Faiss 的要求

    # query hnsw
    top_k = 10
    distance, preds = index.search(queries, k=top_k)
    for i in range(3):
        print(f"Top {top_k} results: {preds[i]}")
        print(f"Distances: {distance[i]}")
        print()


if __name__ == "__main__":
    data, queries, dim, M, efSearch, efConstruction = make_test_data()
    test_ori_hnsw(
        data=data,
        queries=queries,
        dim=dim,
        M=M,
        efSearch=efSearch,
        efConstruction=efConstruction,
    )
