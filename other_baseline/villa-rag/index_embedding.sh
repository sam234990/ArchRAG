python_file="./rag_index.py"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/hier_graph_rag:$PYTHONPATH"

base_path="/mnt/data/wangshu/hcarag/narrativeqa/data"
train_base="${base_path}/train"
test_base="${base_path}/test"

for i in {0..1101}; do
    echo "root_path: ${i}"
    dataset_path="${train_base}/${i}/qa_dataset/corpus_chunk.json"
    save_path="${train_base}/${i}/qa_dataset/rag_corpus_chunk.index"
    python -u $python_file --dataset_path $dataset_path --save_path $save_path
    wait
done

for i in {0..354}; do
    echo "root_path: ${i}"
    dataset_path="${test_base}/${i}/qa_dataset/corpus_chunk.json"
    save_path="${test_base}/${i}/qa_dataset/rag_corpus_chunk.index"
    python -u $python_file --dataset_path $dataset_path --save_path $save_path
    wait
done
