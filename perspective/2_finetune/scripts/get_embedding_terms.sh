for dataset in banking77
do
    for scale in small
    do
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python ./perspective/2_finetune/get_embedding.py \
            --model_name hkunlp/instructor-large \
            --scale $scale \
            --task_name $dataset \
            --data_path ../../datasets/${dataset}/${scale}_terms.jsonl \
            --result_file ../../datasets/${dataset}/${scale}_embeds_terms.hdf5 \
            --measure
    done
done