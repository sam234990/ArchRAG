engine="llama3.1:8b4k"
topk=2
eval_mode="DocQA"
num_workers=4

# 最大并行任务数
max_jobs=8
current_jobs=0
log_base_dir="/home/wangshu/rag/hier_graph_rag/other_baseline/villa-rag/narrative_eval_res"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

dataset_name="narrativeqa_train"

# for i in {0..4}; do
for i in {159..1101}; do

    log_file="${log_base_dir}/train-${i}-${dataset_name}-${topk}-${engine}.log"
    python_file="./villa_rag.py"

    nohup python -u $python_file --dataset_name $dataset_name --topk $topk \
        --eval_mode $eval_mode --engine $engine --num_workers $num_workers \
        --doc_idx $i \
        >$log_file 2>&1 &

    echo "log file: $log_file"
    
    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
    echo "root_path: ${i} finish"
done

# 重置当前任务计数
current_jobs=0
dataset_name="narrativeqa_test"

for i in {0..354}; do
    log_file="${log_base_dir}/test-${i}-${dataset_name}-${topk}-${engine}.log"
    python_file="./villa_rag.py"

    nohup python -u $python_file --dataset_name $dataset_name --topk $topk \
        --eval_mode $eval_mode --engine $engine --num_workers $num_workers \
        --doc_idx $i \
        >$log_file 2>&1 &

    echo "log file: $log_file"
    
    # 增加当前任务计数
    current_jobs=$((current_jobs + 1))

    # 如果达到最大并行任务数，等待一个任务完成
    if [ $current_jobs -ge $max_jobs ]; then
        wait -n
        current_jobs=$((current_jobs - 1))
    fi
    echo "root_path: ${i} finish"
done
