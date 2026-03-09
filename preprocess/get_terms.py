import argparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import json

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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

    return terms


def preprocess_documents(documents):
    processed_docs = []

    for doc in documents:
        # lowercase
        doc = doc.lower()

        # remove punctuation / special characters
        doc = re.sub(r"[^a-z\s]", " ", doc)

        # tokenize
        tokens = word_tokenize(doc)

        # remove stopwords + lemmatize
        tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words
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

