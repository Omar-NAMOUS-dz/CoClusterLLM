#!/bin/bash

bash perspective/2_finetune/scripts/get_embedding.sh
bash perspective/2_finetune/scripts/get_embedding_terms.sh

bash perspective/1_predict_triplet/scripts/term_document_triplet_sampling.sh

bash perspective/1_predict_triplet/scripts/predict_triplet.sh