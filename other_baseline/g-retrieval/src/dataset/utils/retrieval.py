import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import pandas as pd


def retrieval_via_pcst(
    graph, q_emb, textual_nodes, textual_edges, topk=3, topk_e=3, cost_e=0.5
):
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = (
            textual_nodes.to_csv(index=False)
            + "\n"
            + textual_edges.to_csv(index=False, columns=["src", "edge_attr", "dst"])
        )
        graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            num_nodes=graph.num_nodes,
        )
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    vertices, edges = pcst_fast(
        edges, prizes, costs, root, num_clusters, pruning, verbosity_level
    )

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(
        np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()])
    )

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = (
        n.to_csv(index=False)
        + "\n"
        + e.to_csv(index=False, columns=["src", "edge_attr", "dst"])
    )

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes)
    )

    return data, desc


def retrieval_via_pcst_hnsw(
    I_e,
    I_r,
    entity_df,
    relation_df: pd.DataFrame,
    topk=3,
    topk_e=3,
    cost_e=0.5,
):
    c = 0.01
    root = -1  # unrooted
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0

    # 初始化 n_prizes
    n_prizes = np.zeros(len(entity_df))

    if topk > 0:
        # 使用搜索到的索引 I_e 构建前 topk 的 n_prizes
        topk_n_indices = I_e  # I_e[0] 是最近的 topk 节点的索引

        # 设置前 topk 节点的奖赏值为从 topk 到 1 递减
        n_prizes[topk_n_indices] = np.arange(topk, 0, -1)
    else:
        # 如果 topk <= 0，则所有奖赏为 0
        n_prizes = np.zeros(len(entity_df))

    # 获取 relation_df 中的 embedding_idx
    embedding_idx = relation_df["embedding_idx"].values
    if topk_e > 0:
        e_prizes = np.zeros(len(relation_df))
        for i, idx in enumerate(I_r):
            e_prizes[embedding_idx == idx] = topk_e - i

        unique_e_prizes = np.unique(e_prizes)
        topk_e = min(topk_e, len(unique_e_prizes))
        topk_e_values = np.sort(unique_e_prizes)[::-1][:topk_e]
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / np.sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        cost_e = min(cost_e, np.max(e_prizes) * (1 - c / 2))
    else:
        e_prizes = np.zeros(len(relation_df))

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, row in enumerate(relation_df.itertuples(index=False)):
        prize_e = e_prizes[i]
        src = row.head_pcst_idx
        dst = row.tail_pcst_idx
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = len(entity_df) + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    vertices, edges = pcst_fast(
        edges, prizes, costs, root, num_clusters, pruning, verbosity_level
    )

    selected_nodes = vertices[vertices < len(entity_df)]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= len(entity_df)]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= len(entity_df)]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    selected_edges = np.array(selected_edges)

    relation_df_select = relation_df.iloc[selected_edges]
    selected_nodes = np.unique(
        np.concatenate(
            [
                selected_nodes,
                relation_df_select["head_pcst_idx"].values,
                relation_df_select["tail_pcst_idx"].values,
            ]
        )
    )

    entity_df_select = entity_df.iloc[selected_nodes]
    # entity_df_select = entity_df_select[
    #     ["human_readable_id", "name", "node_description"]
    # ]

    entity_df_select = entity_df_select[["human_readable_id", "name"]]
    relation_df_select = relation_df_select[["head_id", "description", "tail_id"]]

    entity_df_select.columns = ["node_id", "node_attr"]
    relation_df_select.columns = ["src", "edge_attr", "dst"]

    desc = (
        entity_df_select.to_csv(index=False)
        + "\n"
        + relation_df_select.to_csv(index=False)
    )

    return desc
