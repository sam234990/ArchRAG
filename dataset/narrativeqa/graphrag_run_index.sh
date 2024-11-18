test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

# 设置PYTHONPATH
export PYTHONPATH="/home/wangshu/rag/4_graphrag/graphrag/:$PYTHONPATH"

# 最大并行任务数
max_jobs=4
current_jobs=0

# for i in {530..1101}; do
#     echo "root_path: ${i}"
#     root_path="${train_base_path}/${i}"
#     cd /home/wangshu/rag/4_graphrag/graphrag/
#     graphrag index --root $root_path > "${root_path}/index.log" 2>&1 &

#     # 增加当前任务计数
#     current_jobs=$((current_jobs + 1))

#     # 如果达到最大并行任务数，等待一个任务完成
#     if [ $current_jobs -ge $max_jobs ]; then
#         wait -n
#         current_jobs=$((current_jobs - 1))
#     fi
#     echo "root_path: ${root_path} finish"

# done

# # 等待所有后台任务完成
# wait

# # 重置当前任务计数
current_jobs=0

# 处理 test 数据集
for i in {0..354}; do
    root_path="${test_base_path}/${i}"
    cd /home/wangshu/rag/4_graphrag/graphrag/
    graphrag init --root $root_path > "${root_path}/init.log" 2>&1 &
    echo "root_path: ${root_path} finish"
    
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
