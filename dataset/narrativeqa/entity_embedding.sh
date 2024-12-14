test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

entity_filename="create_final_entities.parquet"

python_file="/home/wangshu/rag/hier_graph_rag/dataset/narrativeqa/entity_embedding.py"

# 最大并行任务数
max_jobs=10
current_jobs=0

for i in {0..1101}; do
# for i in {0..5}; do
    echo "root_path: ${i}"

    input="${train_base_path}/${i}/hcarag/${entity_filename}"
    output="${train_base_path}/${i}/hcarag/embeded_entities.parquet"
    log_file="./log_index/embedding_${i}.log"
    python -u $python_file --entity_path $input --entity_save_path $output \
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
current_jobs=0

# 处理 test 数据集
for i in {0..354}; do
    echo "root_path: ${i}"
    input="${test_base_path}/${i}/hcarag/${entity_filename}"
    output="${test_base_path}/${i}/hcarag/embeded_entities.parquet"
    log_file="./log_index/test_embedding_${i}.log"
    python -u $python_file --entity_path $input --entity_save_path $output \
        >$log_file 2>&1 &
    
    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
done

# 等待所有后台任务完成
wait
