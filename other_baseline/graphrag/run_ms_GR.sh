# dataset="multihop"
dataset="hotpot"

# dataset_root="/mnt/data/wangshu/hcarag/MultiHop-RAG/graphrag_io"
dataset_root="/mnt/data/wangshu/hcarag/HotpotQA/graphrag_io"
log_file="./eval/msgr_${dataset}-toke_usage.log"

max_concurrent=20

export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/other_baseline/graphrag/:$PYTHONPATH"

nohup python -m graphrag.query --method docqa --dataset_name $dataset \
    --root $dataset_root --max_concurrent $max_concurrent > $log_file 2>&1 &
echo "Running on $dataset, log file: $log_file"