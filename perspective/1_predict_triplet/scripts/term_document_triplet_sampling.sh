# ===== instructor-large =====
scale=small
for dataset in banking77
do
    for max_query in 1024
    do
        for embed in instructor
        do
            terms_feat_path=./datasets/${dataset}/${scale}_embeds_terms.hdf5
            docs_feat_path=./datasets/${dataset}/${scale}_embeds.hdf5
            python term_document_triplet_sampling.py \
                --terms_path ./datasets/${dataset}/${scale}.jsonl \
                --docs_path ./datasets/${dataset}/${scale}.jsonl \
                --terms_feat_path $terms_feat_path \
                --docs_feat_path $docs_feat_path \
                --dataset $dataset \
                --embed_method $embed \
                --max_query $max_query \
                --filter_first_prop 0.0 \
                --large_ent_prop 0.2 \
                --out_dir coclustering_sampled_triplet_results \
                --max_distance 67 \
                --scale $scale \
                --shuffle_inds \
                --seed 100
        done
    done
done