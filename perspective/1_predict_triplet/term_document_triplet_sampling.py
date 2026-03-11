import os
import json
import random
import h5py
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

def load_data(args):
    # data_path = f"../../{args.dataset}/{args.scale}.jsonl"
    docs = [json.loads(l) for l in open(args.docs_path, 'r')]
    terms = [json.loads(l) for l in open(args.terms_path, 'r')]
    return docs, terms

def load_feat(args):
    docs_feat_path = args.docs_feat_path
    terms_feat_path = args.terms_feat_path

    with h5py.File(docs_feat_path, 'r') as f:
        X_docs = f['embeds']
        X_docs = np.asarray(X_docs)

    with h5py.File(terms_feat_path, 'r') as f:
        X_terms = f['embeds']
        X_terms = np.asarray(X_terms)

    return X_docs, X_terms
    
def entropy(vals):
    vals = np.asarray(vals)
    vals /= vals.sum()
    return - (vals * np.log(vals)).sum()

def generate(args):
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = f"{args.out_dir}/{args.dataset}_embed={args.embed_method}_s={args.scale}_m={args.max_query}_d={round(args.max_distance, 1)}{'_f=' + str(round(args.filter_first_prop, 2)) if args.filter_first_prop != 0 else ''}{'_l=' + str(round(args.large_ent_prop, 2)) if args.large_ent_prop != 0.2 else ''}{'_p=' + str(round(args.close_cluster_prop, 3)) if args.close_cluster_prop != 0.02 else ''}{'_sf' if args.shuffle_inds else ''}_choice_seed={args.seed}.json"
    print(save_path)
    random.seed(args.seed)
    np.random.seed(args.seed)
    docs, terms = load_data(args)
    docs_inp = [d['input'] for d in docs]
    terms_inp = [d['input'] for d in terms]
    # for analyzing purpose only
    docs_labels = [d['label'] for d in docs]
    terms_labels = [d['label'] for d in terms]
    X_docs, X_terms = load_feat(args)

    X_docs = StandardScaler().fit_transform(X_docs)
    X_terms = StandardScaler().fit_transform(X_terms)

    if args.scale == "small":
        clustering_docs = AgglomerativeClustering(n_clusters=None, distance_threshold=args.max_distance).fit(X_docs)
        clustering_terms = AgglomerativeClustering(n_clusters=None, distance_threshold=args.max_distance).fit(X_terms)
    elif args.scale == "large":
        clustering_docs = MiniBatchKMeans(n_clusters=100, random_state=args.seed).fit(X_docs)
        clustering_terms = MiniBatchKMeans(n_clusters=100, random_state=args.seed).fit(X_terms)
    
    preds_docs  = clustering_docs .labels_
    preds_terms  = clustering_terms .labels_

    n_clusters_docs = len(set(preds_docs))
    n_clusters_terms = len(set(preds_terms))

    print("Estimated number of clusters for documents: %d" % n_clusters_docs)
    print("Estimated number of clusters for terms: %d" % n_clusters_terms)

    class_member_inds_docs = {}
    for i in range(n_clusters_docs):
        class_member_mask = preds_docs == i
        class_member_inds_docs[i] = np.where(class_member_mask)[0]

    class_member_inds_terms = {}
    for i in range(n_clusters_terms):
        class_member_mask = preds_terms == i
        class_member_inds_terms[i] = np.where(class_member_mask)[0]
    
    
    triplets = []
    while len(triplets) < args.max_query:
        # sample 1 term and 2 documents
        term_cluster = 0
        cluster1, cluster2 = random.sample(range(n_clusters_docs), 2)
        idx = random.choice(class_member_inds_terms[term_cluster])
        choice1 = random.choice(class_member_inds_docs[cluster1])
        choice2 = random.choice(class_member_inds_docs[cluster2])
        if (idx, choice1, choice2) not in triplets \
            and choice1 != choice2:
            triplets.append((idx, choice1, choice2))
            if len(triplets) >= args.max_query:
                break
        term_cluster += 1
        if term_cluster >= n_clusters_terms: term_cluster = 0

    result = []
    for trip in triplets:
        result.append({
            "input": "Query: " + terms_inp[trip[0]] + "\nChoice 1: " + docs_inp[trip[1]] + "\nChoice 2: " + docs_inp[trip[2]] + "\nChoice",
            "output": "",
            "options": [" 1", " 2"],
            "task": args.dataset,
            "query_idx": int(trip[0]),
            "choice1_idx": int(trip[2]),
            "choice2_idx": int(trip[1]),
        })

    while len(triplets) < 2 * args.max_query:
        # sample 1 document and 2 terms
        docs_cluster = 0
        cluster1, cluster2 = random.sample(range(n_clusters_terms), 2)
        idx = random.choice(class_member_inds_docs[docs_cluster])
        choice1 = random.choice(class_member_inds_terms[cluster1])
        choice2 = random.choice(class_member_inds_terms[cluster2])
        if (idx, choice1, choice2) not in triplets \
            and choice1 != choice2:
            triplets.append((idx, choice1, choice2))
            if len(triplets) >= 2 * args.max_query:
                break
        docs_cluster += 1
        if docs_cluster >= n_clusters_docs: docs_cluster = 0
    
    for trip in triplets:
        result.append({
            "input": "Query: " + docs_inp[trip[0]] + "\nChoice 1: " + terms_inp[trip[1]] + "\nChoice 2: " + terms_inp[trip[2]] + "\nChoice",
            "output": "",
            "options": [" 1", " 2"],
            "task": args.dataset,
            "query_idx": int(trip[0]),
            "choice1_idx": int(trip[2]),
            "choice2_idx": int(trip[1]),
        })
    

    print("Total number: ", len(result))
    with open(save_path, 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--docs_path", type=str, required=True)
    parser.add_argument("--terms_path", type=str, required=True)
    parser.add_argument("--docs_feat_path", type=str, required=True)
    parser.add_argument("--terms_feat_path", type=str, required=True)
    parser.add_argument("--embed_method", type=str, default='instructor')
    parser.add_argument("--scale", type=str, default="small")
    parser.add_argument("--max_query", type=int, default=256)
    parser.add_argument("--large_ent_prop", type=float, default=0.20)
    parser.add_argument("--filter_first_prop", type=float, default=0.)
    parser.add_argument("--close_cluster_prop", type=float, default=0.02)
    parser.add_argument("--max_distance", type=float, default=67)
    parser.add_argument("--shuffle_inds", action="store_true")
    parser.add_argument("--out_dir", default="links", type=str)
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()
    generate(args)