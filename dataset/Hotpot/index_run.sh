base_path="/mnt/data/wangshu/hcarag/HotpotQA/hcarag"
relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"
output_dir="/mnt/data/wangshu/hcarag/HotpotQA/hcarag/hc_index_8b4k"
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=10
max_cluster_size=15
entity_second_embedding=True
engine="llama3.1:8b4k"

num_workers=32

log_file="./log_index/index_${engine}-token_usage.log"
python_file="/home/wangshu/rag/hier_graph_rag/src/index.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

nohup python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
    --entity_filename $entity_filename --output_dir $output_dir --wx_weight $wx_weight --m_du_scale $m_du_scale --max_level $max_level \
    --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
    --entity_second_embedding $entity_second_embedding \
    --engine $engine --num_workers $num_workers \
    >$log_file 2>&1 &
