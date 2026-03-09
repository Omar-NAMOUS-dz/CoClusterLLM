import argparse
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import json


def preprocess(dataset, size="small", max_doc_freq=0.9):
    data = []
    path = f"./datasets/{dataset}/{size}.jsonl" 
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(data[0])

    documents = [document["input"] for document in data]
    processed_docs = preprocess_documents(documents)
    terms = extract_terms(processed_docs, max_doc_freq=max_doc_freq)

    print(terms)

    terms_jsonl = [{"task": dataset, "input": t, "label": ""} for t in terms]

    dist_path = f"./datasets/{dataset}/{size}_terms.jsonl" 

    with open(dist_path, "w", encoding="utf-8") as f:
        for obj in terms_jsonl:
            f.write(json.dumps(obj) + "\n")


def preprocess_documents(documents):
    nlp = spacy.load("en_core_web_sm")
    
    processed_docs = []

    for doc in documents:

        doc = doc.lower()
        doc = re.sub(r"[^a-z\s]", " ", doc)

        spacy_doc = nlp(doc)

        tokens = [
            token.lemma_
            for token in spacy_doc
            if not token.is_stop and token.is_alpha
        ]

        processed_docs.append(" ".join(tokens))

    return processed_docs


def extract_terms(processed_docs, max_doc_freq=0.9):

    vectorizer = CountVectorizer(
        max_df=max_doc_freq
    )

    X = vectorizer.fit_transform(processed_docs)
    terms = vectorizer.get_feature_names_out()
    return terms

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="banking77", type=str)
    parser.add_argument("--size", default="small", type=str)
    parser.add_argument("--max_doc_freq", default=0.9, type=float)
    args = parser.parse_args()

    preprocess(args.dataset, args.size, args.max_doc_freq)

