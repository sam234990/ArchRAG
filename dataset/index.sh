base_path="archrag" # input dataset path 
relationship_filename="create_final_relationships.parquet"
entity_filename="create_final_entities.parquet"
output_dir="archrag_index" # output index path
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=10
max_cluster_size=15
entity_second_embedding=True
api_key="" #TODO
api_base="" #TODO
engine="llama3.1:8b4k" # llm engine


augment_graph=True
cluster_method="weighted_leiden"
output_dir="" #TODO
num_workers=10

log_file="./index.log"
python_file="src/index.py"

export CUDA_VISIBLE_DEVICES=7

nohup python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
    --entity_filename $entity_filename --output_dir $output_dir --wx_weight $wx_weight --m_du_scale $m_du_scale --max_level $max_level \
    --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
    --entity_second_embedding $entity_second_embedding \
    --engine $engine --num_workers $num_workers \
    --augment_graph $augment_graph --cluster_method $cluster_method \
    > $log_file 2>&1 &
echo "log file: $log_file"
