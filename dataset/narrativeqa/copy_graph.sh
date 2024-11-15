test_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/test"
train_base_path="/mnt/data/wangshu/hcarag/narrativeqa/data/train"


for i in {0..354}; do
    output_path="${test_base_path}/${i}/output/"
    hcarag_path="${test_base_path}/${i}/hcarag/"

    ln -s $output_path"/create_final_entities.parquet" $hcarag_path"/create_final_entities.parquet"
    ln -s $output_path"/create_final_relationships.parquet" $hcarag_path"/create_final_relationships.parquet"
    echo "${output_path} link finish"
done

for i in {11..1101}; do
    output_path="${train_base_path}/${i}/output/"
    hcarag_path="${train_base_path}/${i}/hcarag/"

    ln -s $output_path"/create_final_entities.parquet" $hcarag_path"/create_final_entities.parquet"
    ln -s $output_path"/create_final_relationships.parquet" $hcarag_path"/create_final_relationships.parquet"
    echo "${output_path} link finish"
done
