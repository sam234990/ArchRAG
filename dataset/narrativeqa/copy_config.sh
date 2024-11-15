test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"

base_setting="/mnt/data/wangshu/hcarag/narrativeqa/data/settings.yaml"

for i in {0..354}; do
    root_path="${test_base_path}/${i}"
    cp $base_setting "${root_path}/settings.yaml"
    echo "${root_path} copy finish"
done

for i in {0..1101}; do
    root_path="${train_base_path}/${i}"
    cp $base_setting "${root_path}/settings.yaml"
    echo "${root_path} copy finish"
done
