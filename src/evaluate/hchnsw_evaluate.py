import faiss
import time
import numpy as np
import math
import pandas as pd
import os

max_level = 10
num_element = int(1e7)
# dim = 128
# dim = 1024
dim = 3072
num_queries = 200
M = 32
# M = 16
efSearch = 100
efConstruction = 50
# topks = [1, 2, 3, 4, 5, 10, 15, 20]
topks = [1, 2, 5, 10]


def check_hnswflat_default_metric():
    d = 128
    index = faiss.IndexHNSWFlat(d, 32)
    metric = index.metric_type
    if metric == faiss.METRIC_L2:
        print("IndexHNSWFlat L2")
    elif metric == faiss.METRIC_INNER_PRODUCT:
        print("IndexHNSWFlat Inner Product")
    else:
        print(f"IndexHNSWFlat used metric: {metric}")


def make_test_data(
    max_level, num_element, dim, num_queries, min_top_nodes=10, fluctuation=0.05
):
    np.random.seed(42)
    level_counts = [int(num_element)]

    for lvl in range(1, max_level + 1):
        prev_cnt = level_counts[-1]
        divisor = 3 if np.random.rand() < 0.5 else 4
        base_cnt = prev_cnt / divisor
        fluct = base_cnt * fluctuation
        cnt = int(base_cnt + np.random.uniform(-fluct, fluct))
        cnt = max(min_top_nodes, cnt)
        level_counts.append(cnt)

    # 修正最上层节点数
    level_counts[-1] = max(level_counts[-1], min_top_nodes)

    # 打印每层节点数
    for lvl, cnt in enumerate(level_counts):
        print(f"Level {lvl}: {cnt} points")

    levels = []
    datas = []
    # 为每层生成向量和level标签
    for lvl, cnt in enumerate(level_counts):
        datas.append(np.float32(np.random.random((cnt, dim))))
        levels.extend([lvl] * cnt)

    data = np.ascontiguousarray(np.vstack(datas))
    levels = np.array(levels, dtype=int)

    # 创建查询向量
    queries = np.float32(np.random.random((num_queries, dim)))
    queries[:, 0] += np.arange(num_queries) / 1000.0
    queries = np.ascontiguousarray(queries)

    print(f"Total points: {len(data)}")
    print(f"Total queries: {len(queries)}")
    print(f"Data shape: {data.shape}")
    print(f"Queries shape: {queries.shape}")
    return data, levels, queries


def compute_ground_truth(data, queries, levels):
    # compute each query ground truth, exact top 20 nearest neighbors in each level.

    result_ground_truth = []

    for lvl in range(max_level + 1):
        print(f"compute ground truth for level {lvl}")
        # get the level data
        level_data = data[levels == lvl]
        # create faiss index
        index = faiss.IndexFlatL2(level_data.shape[1])
        # add data to index
        index.add(level_data)
        # search top 20 nearest neighbors
        D, I = index.search(queries, 20)
        # save ground truth (indices are relative to level_data)
        ground_truth = np.array(I)
        result_ground_truth.append(ground_truth)
    # result_ground_truth is a list of arrays, each (num_queries, 20)
    print(f"ground truth per level: {[gt.shape for gt in result_ground_truth]}")
    return result_ground_truth


def load_test_dataset(save_dataset_dir):
    data_path = os.path.join(save_dataset_dir, "data.npy")
    levels_path = os.path.join(save_dataset_dir, "levels.npy")
    queries_path = os.path.join(save_dataset_dir, "queries.npy")
    gt_path = os.path.join(save_dataset_dir, "ground_truth_list.npy")
    if os.path.exists(data_path):
        data = np.load(data_path)
        levels = np.load(levels_path)
        queries = np.load(queries_path)
        ground_truth = np.load(gt_path, allow_pickle=True)

        print(
            f"ground truth shape: {np.array(ground_truth).shape if isinstance(ground_truth, list) else ground_truth.shape}"
        )

        # output level counts
        level_counts = np.zeros(max_level + 1, dtype=int)
        for lvl in range(max_level + 1):
            level_counts[lvl] = np.sum(levels == lvl)
            print(f"Level {lvl}: {level_counts[lvl]} points")

        print("finish load data")
        level_offset = []
        for lvl in range(max_level + 1):
            level_offset.append(np.sum(level_counts[:lvl]))
        print(f"level offset: {level_offset}")
        return data, levels, queries, ground_truth, level_offset
    else:
        # create directory
        print(f"data not found, create directory{save_dataset_dir}")
        os.makedirs(save_dataset_dir, exist_ok=True)
        data, levels, queries = make_test_data(max_level, num_element, dim, num_queries)
        # compute ground truth
        ground_truth = compute_ground_truth(data, queries, levels)

        # save data
        np.save(data_path, data)
        np.save(queries_path, queries)
        np.save(levels_path, levels)
        np.save(gt_path, ground_truth, allow_pickle=True)
        # output level counts
        level_counts = np.zeros(max_level + 1, dtype=int)
        for lvl in range(max_level + 1):
            level_counts[lvl] = np.sum(levels == lvl)
            print(f"Level {lvl}: {level_counts[lvl]} points")

        level_offset = []
        for lvl in range(max_level + 1):
            level_offset.append(np.sum(level_counts[:lvl]))
        print(f"level offset: {level_offset}")
        return data, levels, queries, ground_truth, level_offset


def compute_ann_baseline_index(data, levels, save_path):

    index_path = os.path.join(
        save_path, f"baseline_index_{M}", f"level_{max_level}.index"
    )
    baseline_index_list = []
    if os.path.exists(index_path):
        # load index
        for lvl in range(max_level + 1):
            index = faiss.read_index(
                os.path.join(save_path, f"baseline_index_{M}", f"level_{lvl}.index")
            )
            baseline_index_list.append(index)
        index_path = os.path.join(save_path, f"baseline_index_{M}")
        print(f"baseline index loaded from {index_path}")
    else:
        index_time_list = []

        # create directory
        os.makedirs(os.path.join(save_path, f"baseline_index_{M}"), exist_ok=True)

        for lvl in range(max_level + 1):
            print(f"compute baseline index for level {lvl}")
            time_start = time.time()
            # get the level data
            level_data = data[levels == lvl]
            # create faiss index
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = efSearch
            print(f"hnsw level: {index.hnsw.max_level}")

            # add data to index
            index.add(level_data)

            print(f"hnsw level: {index.hnsw.max_level}")

            # save index
            faiss.write_index(
                index,
                os.path.join(save_path, f"baseline_index_{M}", f"level_{lvl}.index"),
            )

            baseline_index_list.append(index)
            time_end = time.time()
            index_time_list.append(time_end - time_start)
            print(f"index {lvl} time: {time_end - time_start:.2f}s")

        index_all_time = sum(index_time_list)
        print(f"total index time: {index_all_time:.2f}s")

        print(f"index saved to {save_path}")

    return baseline_index_list


def compute_hchnsw_index(data, levels, save_path):

    hchnsw_save_path = os.path.join(save_path, f"hchnsw_{M}", "hchnsw.index")
    if os.path.exists(hchnsw_save_path):
        # load index
        index = faiss.read_index(hchnsw_save_path)
        print(f"HCHNSW index loaded from {hchnsw_save_path}")
        return index
    else:
        # create directory
        os.makedirs(os.path.join(save_path, f"hchnsw_{M}"), exist_ok=True)

    start_time = time.time()
    # create hchnsw index
    index = faiss.IndexHCHNSWFlat(dim, max_level, M, 1, len(data))
    index.set_vector_level(levels)
    index.hchnsw.efConstruction = efConstruction
    index.hchnsw.efSearch = efSearch

    # add data to index
    index.add(data)

    # save index
    faiss.write_index(index, hchnsw_save_path)

    end_time = time.time()
    print(f"index time: {end_time - start_time:.2f}s")
    print(f"hchnsw index saved to {save_path}")
    return index


def compute_recall(pred, gt, topk):
    # compute recall
    # pred is a 2D array of shape (num_queries, topk)
    # gt is a 2D array of shape (num_queries, topk)
    # pred and gt are both indices
    # compute recall
    recall = np.mean(np.isin(pred, gt[:, :topk], assume_unique=True))
    return recall


def test_baseline_index(baseline_index_list, queries, ground_truth: list):
    # test baseline index
    time_list = []
    recall_list = []
    per_level_time = {}
    for topk in topks:
        print(f"test baseline index for topk {topk}")
        topk_time_list = []
        topk_recall_list = []
        # for each level, search topk nearest neighbors
        for lvl in range(max_level + 1):
            # print(f"test baseline index for level {lvl}")
            index = baseline_index_list[lvl]
            index.hnsw.efSearch = 100
            query_time = 0
            search_res = []
            for i in range(queries.shape[0]):
                query = np.expand_dims(queries[i], axis=0)
                time_start = time.time()
                # search top k nearest neighbors
                D, I = index.search(query, topk)
                time_end = time.time()
                query_time += time_end - time_start

                # append search answer into search res to compute recall
                I = I.flatten()
                search_res.append(I)

            topk_time_list.append(query_time)

            # compute recall
            search_res = np.array(search_res)
            ground_truth_lvl = ground_truth[lvl]
            recall = compute_recall(search_res, ground_truth_lvl, topk)
            topk_recall_list.append(recall)

        per_level_time[topk] = topk_time_list
        topk_all_time = sum(topk_time_list)
        topk_avg_recall = np.mean(topk_recall_list)
        print(f"topk {topk} time: {topk_all_time:.2f}s")
        print(f"topk {topk} recall: {topk_avg_recall:.4f}")
        time_list.append(topk_all_time)
        recall_list.append(topk_avg_recall)

    print(f"total time: {sum(time_list):.2f}s")
    print(f"total recall: {np.mean(recall_list):.4f}")
    return time_list, recall_list, per_level_time


def test_hchnsw_index(
    hchnsw_index, queries, ground_truth, level_offset, fast_query: bool = False
):
    # test hchnsw index
    time_list = []
    recall_list = []
    per_level_time = {}

    for topk in topks:
        print(f"test hchnsw index for topk {topk}")
        topk_recall_list = []
        query_time = 0
        all_level_search_res = []
        per_level_query_time = [0.0 for _ in range(max_level + 1)]  # 新增

        for i in range(queries.shape[0]):
            query = np.expand_dims(queries[i], axis=0)
            entry = -1
            per_query_level_results = []
            for search_l in range(max_level, -1, -1):
                params = faiss.SearchParametersHCHNSW()
                params.search_level = search_l
                if fast_query:
                    params.entry_point = entry
                else:
                    params.entry_point = -1

                hchnsw_index.hchnsw.efSearch = 20

                time_start = time.time()
                D, I = hchnsw_index.search(query, k=topk, params=params)
                time_end = time.time()
                query_time += time_end - time_start
                per_level_query_time[search_l] += time_end - time_start  # 统计每层时间

                I = I.flatten()
                per_query_level_results.append(I)
                entry = int(I[0])

            per_query_level_results = per_query_level_results[::-1]
            all_level_search_res.append(per_query_level_results)

        all_level_search_res = np.array(all_level_search_res)

        all_level_search_res = np.transpose(all_level_search_res, (1, 0, 2))

        for search_l in range(max_level, -1, -1):
            query_res = all_level_search_res[search_l]
            offset = level_offset[search_l]
            # compute recall
            ground_truth_lvl = ground_truth[search_l]
            # add offset to each value in ground_truth_lvl
            # ground_truth_lvl shape: (num_queries, topk)
            ground_truth_lvl_offset = ground_truth_lvl + offset

            recall = compute_recall(query_res, ground_truth_lvl_offset, topk)
            topk_recall_list.append(recall)

        topk_avg_recall = np.mean(topk_recall_list)
        print(f"topk {topk} time: {query_time:.4f}s")
        print(f"topk {topk} recall: {topk_avg_recall:.4f}")
        time_list.append(query_time)
        recall_list.append(topk_avg_recall)
        per_level_time[topk] = per_level_query_time

    print(f"total time: {sum(time_list):.4f}s")
    print(f"total recall: {np.mean(recall_list):.4f}")
    return time_list, recall_list, per_level_time


if __name__ == "__main__":
    check_hnswflat_default_metric()

    # save_dir = "/mnt/data/wangshu/hcarag/syndata/10M"
    save_dir = f"/mnt/data/wangshu/hcarag/syndata/10M-{dim}"
    print(save_dir)

    data, levels, queries, ground_truth, level_offset = load_test_dataset(save_dir)

    baseline_index_list = compute_ann_baseline_index(data, levels, save_dir)
    hchnsw_index = compute_hchnsw_index(data, levels, save_dir)

    # test baseline index
    base_time, base_recall, base_per_level_time = test_baseline_index(
        baseline_index_list, queries, ground_truth
    )
    
    for k in range(len(base_recall)):
        print(f"topk {topks[k]}:")
        print(
            f"baseline recall: {base_recall[k]:.4f}, baseline time: {base_time[k]:.2f}s"
        )
        for lvl in range(max_level + 1):
            print(
                f"level {lvl} baseline time: {base_per_level_time[topks[k]][lvl]:.4f}s"
            )
    
    
    # # test hchnsw index
    # hchnsw_time, hchnsw_recall, hchnsw_per_level_time = test_hchnsw_index(
    #     hchnsw_index, queries, ground_truth, level_offset, fast_query=False
    # )

    # hchnsw_fast_time, hchnsw_fast_recall, hchnsw_fast_per_level_time = (
    #     test_hchnsw_index(
    #         hchnsw_index, queries, ground_truth, level_offset, fast_query=True
    #     )
    # )
    # # print results
    # for k in range(len(base_recall)):
    #     print(f"topk {topks[k]}:")
    #     print(
    #         f"baseline recall: {base_recall[k]:.4f}, hchnsw recall: {hchnsw_recall[k]:.4f}, hchnsw fast recall: {hchnsw_fast_recall[k]:.4f}"
    #     )
    #     print(
    #         f"baseline time: {base_time[k]:.2f}s, hchnsw time: {hchnsw_time[k]:.2f}s, hchnsw fast time: {hchnsw_fast_time[k]:.2f}s"
    #     )
    #     for lvl in range(max_level + 1):
    #         print(
    #             f"level {lvl} baseline time: {base_per_level_time[topks[k]][lvl]:.4f}s, hchnsw fast time: {hchnsw_fast_per_level_time[topks[k]][lvl]:.4f}s"
    #         )
