eval_mode="DocQA"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/:$PYTHONPATH"

relationship_filename="create_final_relationships.parquet"
entity_filename="embeded_entities.parquet"

python_file="/home/wangshu/rag/hier_graph_rag/src/evaluate/test_qa.py"

test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

strategy="global"
k_each_level=4
k_final=15
topk_e=5
all_k_inference=15
# generate_strategy="direct"
generate_strategy="mr"
response_type="QA"

temperature=0.1
only_entity=False
involve_llm_res=False
topk_chunk=0
num_workers=4

# 最大并行任务数
max_jobs=8
current_jobs=0

log_file_dir="./eval/evaluate_t${temperature}_${strategy}_${k_each_level}_${k_final}_${topk_e}_${all_k_inference}_${generate_strategy}_${response_type}_${involve_llm_res}_${topk_chunk}/"
if [ ! -d $log_file_dir ]; then
    mkdir $log_file_dir
fi

for i in {0..1101}; do
# for i in {0..10}; do
    echo "$i"

    base_path="${train_base_path}/${i}/hcarag"
    output_dir="${train_base_path}/${i}//hcarag/hc_index_8b"

    dataset_name="narrativeqa_train"
    log_file="${log_file_dir}/train_${i}_eval.log"

    python -u $python_file --strategy $strategy --k_each_level $k_each_level \
        --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
        --generate_strategy $generate_strategy --response_type $response_type \
        --temperature $temperature --eval_mode $eval_mode \
        --relationship_filename $relationship_filename --entity_filename $entity_filename \
        --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
        --only_entity $only_entity --num_workers $num_workers \
        --involve_llm_res $involve_llm_res --topk_chunk $topk_chunk \
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
    echo "log file: $i finish"
done

# 等待所有后台任务完成
wait

# 重置当前任务计数
current_jobs=0

dataset_name="narrativeqa_test"

for i in {0..354}; do

    base_path="${test_base_path}/${i}/hcarag"
    output_dir="${test_base_path}/${i}//hcarag/hc_index_8b"

    dataset_name="narrativeqa_test"
    log_file="${log_file_dir}/test_${i}_eval.log"

    nohup python -u $python_file --strategy $strategy --k_each_level $k_each_level \
        --k_final $k_final --all_k_inference $all_k_inference --topk_e $topk_e \
        --generate_strategy $generate_strategy --response_type $response_type \
        --temperature $temperature --eval_mode $eval_mode \
        --relationship_filename $relationship_filename --entity_filename $entity_filename \
        --output_dir $output_dir --base_path $base_path --dataset_name $dataset_name \
        --only_entity $only_entity --num_workers $num_workers \
        --involve_llm_res $involve_llm_res --topk_chunk $topk_chunk \
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
    echo "log file: $i finish"
done

# 等待所有后台任务完成
wait
