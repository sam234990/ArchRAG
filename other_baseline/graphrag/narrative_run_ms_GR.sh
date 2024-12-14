dataset="narrativeqa_train"
community_level=2

# method="docqa"
method="localdocqa"

export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/other_baseline/graphrag/:$PYTHONPATH"

test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

# 最大并行任务数
max_jobs=3
current_jobs=0

for i in {0..1101}; do
# for i in {0..3}; do
    echo "root_path: ${i}"
    dataset_root="${train_base_path}/${i}/"
    log_file="./eval_narrative/train_msgr_${i}-${community_level}-${method}-toke_usage.log"
    max_concurrent=8
    data="${train_base_path}/${i}/output/"
    nohup python -m graphrag.query --method $method --dataset_name $dataset \
        --community_level $community_level --data $data \
        --root $dataset_root --max_concurrent $max_concurrent > $log_file 2>&1 &
    echo "Running on $dataset, log file: $log_file"

    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
    echo "root_path: ${root_path} finish"

done
# 等待所有后台任务完成
wait



# 重置当前任务计数
current_jobs=0
dataset="narrativeqa_test"


for i in {0..354}; do
    echo "root_path: ${i}"
    dataset_root="${test_base_path}/${i}/"
    log_file="./eval_narrative/test-msgr_${i}-${community_level}-${method}-toke_usage.log"
    max_concurrent=8
    data="${test_base_path}/${i}/output/"
    nohup python -m graphrag.query --method $method --dataset_name $dataset \
        --community_level $community_level --data $data \
        --root $dataset_root --max_concurrent $max_concurrent > $log_file 2>&1 &
    echo "Running on $dataset, log file: $log_file"

    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
    echo "root_path: ${root_path} finish"

done
# 等待所有后台任务完成
wait
