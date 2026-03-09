python3 -m spacy download en_core_web_sm
python3 preprocess/get_terms.py --dataset "banking77" --size "small" --max_doc_freq 0.9