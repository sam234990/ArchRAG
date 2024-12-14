test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

relationship_filename="create_final_relationships.parquet"
entity_filename="embeded_entities.parquet"
wx_weight=0.8
m_du_scale=1
max_level=6
min_clusters=10
max_cluster_size=15
entity_second_embedding=True
engine="llama3.1:8b4k"
num_workers=10

python_file="/home/wangshu/rag/hier_graph_rag/src/index.py"

# 最大并行任务数
max_jobs=3
current_jobs=0

# for i in {198..1101}; do
for i in {0..10}; do
    echo "root_path: ${i}"

    base_path="${train_base_path}/${i}/hcarag"
    output_dir="${train_base_path}/${i}//hcarag/hc_index_8b"
    log_file="/home/wangshu/rag/hier_graph_rag/dataset/narrativeqa/log_index/${i}_index_3_${engine}.log"
    
    python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
        --entity_filename $entity_filename --output_dir $output_dir --wx_weight $wx_weight \
        --m_du_scale $m_du_scale --max_level $max_level \
        --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
        --entity_second_embedding $entity_second_embedding \
        --engine $engine --num_workers $num_workers \
        >$log_file 2>&1 &

    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
    echo "log file: $i finish"
done

# 等待所有后台任务完成
wait


# 重置当前任务计数
# current_jobs=0


# 处理 test 数据集
# for i in {0..354}; do
# for i in {142..354}; do
#     echo "root_path: ${i}"
#     base_path="${test_base_path}/${i}/hcarag"
#     output_dir="${test_base_path}/${i}//hcarag/hc_index_8b"
#     log_file="/home/wangshu/rag/hier_graph_rag/dataset/narrativeqa/log_index/test_${i}_index_3_${engine}.log"
    
#     python -u $python_file --base_path $base_path --relationship_filename $relationship_filename \
#         --entity_filename $entity_filename --output_dir $output_dir --wx_weight $wx_weight --m_du_scale $m_du_scale --max_level $max_level \
#         --min_clusters $min_clusters --max_cluster_size $max_cluster_size \
#         --entity_second_embedding $entity_second_embedding \
#         --engine $engine --num_workers $num_workers \
#         >$log_file 2>&1 &
    
#     # 增加当前任务计数
#     current_jobs=$((current_jobs + 1))

#     # 如果达到最大并行任务数，等待一个任务完成
#     if [ $current_jobs -ge $max_jobs ]; then
#         wait -n
#         current_jobs=$((current_jobs - 1))
#     fi
#     echo "log file: $i finish"
# done

# # 等待所有后台任务完成
# wait
