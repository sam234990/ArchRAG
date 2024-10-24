python_file="./zero-cot.py"
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

datasets=(hotpot multihop webq narrativeqa mintaka)
# datasets=("hotpot")
strategies=("cot" "zero-shot")

for dataset_name in ${datasets[@]}; do
    for strategy in ${strategies[@]}; do
        if [ $dataset_name == "mintaka" ] || [ $dataset_name == "webq" ]; then
            eval_mode="KGQA"
        else
            eval_mode="DocQA"
        fi

        log_file="./eval_res/inference_only-${dataset_name}-${strategy}-${eval_mode}.log"
        python_file="./zero-cot.py"
        (
            python -u $python_file --dataset_name $dataset_name --strategy $strategy \
                --eval_mode $eval_mode \
                >$log_file 2>&1
            wait
            echo "log file: $log_file"
        )
    done
done
