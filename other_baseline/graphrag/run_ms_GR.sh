dataset="multihop"
# dataset="hotpot"
community_level=2

dataset_root="/mnt/data/wangshu/hcarag/MultiHop-RAG/graphrag_io"
# dataset_root="/mnt/data/wangshu/hcarag/HotpotQA/graphrag_io"
log_file="./eval/msgr_${dataset}-${community_level}-toke_usage.log"

max_concurrent=4

export PYTHONPATH="/home/wangshu/rag/hier_graph_rag/other_baseline/graphrag/:$PYTHONPATH"

nohup python -m graphrag.query --method docqa --dataset_name $dataset \
    --community_level $community_level \
    --root $dataset_root --max_concurrent $max_concurrent > $log_file 2>&1 &
echo "Running on $dataset, log file: $log_file"