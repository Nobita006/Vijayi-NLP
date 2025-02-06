#!/usr/bin/env python3
"""
Improved NLP Pipeline for Extracting and Categorizing Tasks from Unstructured Text
using the Elbow Method for Cluster Selection.

Improvements include:
    - Reading input text from a file.
    - Using the elbow method to help decide the number of clusters.
    - Pretty-printing output in JSON format.
    - Modular structure with clear functions and command-line argument parsing.
    - Adjusting the max clusters to not exceed the number of task samples.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.cluster import KMeans
from gensim import corpora, models

# Load the spaCy model (with medium-sized vectors)
nlp = spacy.load('en_core_web_md')


##############################################
# Utility Functions
##############################################
def load_text_from_file(filepath):
    """Read text from the provided file path."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        return text.strip()
    except Exception as e:
        raise IOError(f"Error reading file '{filepath}': {e}")


def pretty_print_tasks(tasks, title="Tasks:"):
    """Pretty-print the list of task dictionaries."""
    print(f"\n{title}")
    print(json.dumps(tasks, indent=4))


##############################################
# Preprocessing Functions
##############################################
def clean_text(text):
    """Clean the input text by stripping whitespace."""
    return text.strip()


##############################################
# Task Identification and Extraction
##############################################
def is_task_sentence(sentence):
    """
    Identify if a sentence likely represents a task using heuristics:
      - Sentence starts with a verb in base form.
      - Contains common task-related keywords/phrases.
    """
    sent_text = sentence.text.strip()
    sent_lower = sent_text.lower()

    # Define keywords that may indicate a task.
    task_keywords = ['has to', 'need to', 'needs to', 'should', 'must', 'please', "don't forget"]

    # Heuristic 1: Check if the first token is a verb in base form.
    first_token = sentence[0]
    if first_token.pos_ == 'VERB' and first_token.tag_ == 'VB':
        return True

    # Heuristic 2: Look for task-indicating phrases in the sentence.
    for keyword in task_keywords:
        if keyword in sent_lower:
            return True

    return False


def extract_deadline(sentence):
    """
    Extract deadline information from a sentence by combining adjacent
    entities labeled as TIME or DATE.
    """
    deadline_tokens = []
    for ent in sentence.ents:
        if ent.label_ in ["TIME", "DATE"]:
            deadline_tokens.append(ent.text)
    if deadline_tokens:
        return " ".join(deadline_tokens)
    return None


def extract_task_details(sentence):
    """
    Extract details from a task sentence:
        - The full task text.
        - The performer (the first PERSON entity encountered, if any).
        - The deadline (if any).
    """
    task_text = sentence.text.strip()
    performer = None

    # Look for the first PERSON entity.
    for ent in sentence.ents:
        if ent.label_ == "PERSON":
            performer = ent.text
            break

    deadline = extract_deadline(sentence)

    return task_text, performer, deadline


def process_text(text):
    """
    Process the text: clean it, segment it into sentences,
    and extract sentences that likely represent tasks.
    Returns a list of dictionaries with task details.
    """
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)
    tasks = []

    for sent in doc.sents:
        if is_task_sentence(sent):
            task_text, performer, deadline = extract_task_details(sent)
            tasks.append({
                "task": task_text,
                "performer": performer,
                "deadline": deadline
            })
    return tasks


##############################################
# Clustering and Categorization Functions
##############################################
def cluster_tasks(tasks, num_clusters):
    """
    Cluster the tasks based on their sentence embeddings using KMeans.
    Adds a 'cluster' field to each task.
    Returns:
        - Updated tasks.
        - The fitted KMeans model.
        - The array of task vectors.
    """
    task_vectors = []
    for task in tasks:
        doc = nlp(task["task"])
        task_vectors.append(doc.vector)
    X = np.array(task_vectors)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    for i, task in enumerate(tasks):
        task["cluster"] = int(labels[i])
    return tasks, kmeans, X


def label_clusters_with_lda(tasks, num_topics=1):
    """
    For each cluster, perform topic modeling (using LDA) on the task texts
    to derive a category label from the dominant topic.
    Adds a 'category' field to each task.
    """
    # Group tasks by cluster.
    cluster_tasks_dict = {}
    for task in tasks:
        cluster = task["cluster"]
        cluster_tasks_dict.setdefault(cluster, []).append(task["task"])

    cluster_labels = {}
    for cluster, sentences in cluster_tasks_dict.items():
        texts = []
        for sentence in sentences:
            doc = nlp(sentence)
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if token.is_alpha and token.text.lower() not in STOP_WORDS
            ]
            texts.append(tokens)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        if len(dictionary) == 0:
            cluster_labels[cluster] = "General"
            continue

        lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)
        topic_terms = lda_model.show_topic(0, topn=3)
        label = " ".join([word for word, prob in topic_terms])
        cluster_labels[cluster] = label

    for task in tasks:
        task["category"] = cluster_labels[task["cluster"]]
    return tasks, cluster_labels


def determine_optimal_clusters(task_vectors, min_clusters=2, max_clusters=10):
    """
    Use the elbow method to display a plot of inertia for different numbers
    of clusters. The function adjusts max_clusters if there are fewer samples.
    """
    n_samples = task_vectors.shape[0]
    
    if n_samples < min_clusters:
        print(f"Warning: Number of samples ({n_samples}) is less than the minimum clusters ({min_clusters}).")
        min_clusters = n_samples
    if n_samples < max_clusters:
        max_clusters = n_samples

    inertias = []
    cluster_range = range(min_clusters, max_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(task_vectors)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(list(cluster_range), inertias, marker="o")
    plt.title("Elbow Method For Optimal Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.xticks(list(cluster_range))
    plt.grid(True)
    plt.show()

    print("Elbow Method Inertia Values:")
    for k, inertia in zip(cluster_range, inertias):
        print(f"Clusters: {k}, Inertia: {inertia}")


##############################################
# Main Function with Argument Parsing
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Extract and categorize tasks from unstructured text using the Elbow Method for cluster selection."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input text file (e.g., input.txt)."
    )
    parser.add_argument(
        "--min_clusters", type=int, default=2,
        help="Minimum number of clusters to try for the elbow method (default: 2)."
    )
    parser.add_argument(
        "--max_clusters", type=int, default=10,
        help="Maximum number of clusters to try for the elbow method (default: 10)."
    )
    args = parser.parse_args()

    try:
        text = load_text_from_file(args.input)
    except IOError as e:
        print(e)
        return

    print("Input Text:")
    print(text)

    # Process the text to extract tasks.
    tasks = process_text(text)
    pretty_print_tasks(tasks, title="Extracted Tasks:")

    if not tasks:
        print("No tasks found in the input.")
        return

    # Compute task vectors for clustering.
    task_vectors = []
    for task in tasks:
        doc = nlp(task["task"])
        task_vectors.append(doc.vector)
    task_vectors = np.array(task_vectors)

    print("\nDetermining the optimal number of clusters using the Elbow Method...")
    determine_optimal_clusters(task_vectors, min_clusters=args.min_clusters, max_clusters=args.max_clusters)

    # After reviewing the elbow plot, prompt the user to input the desired number of clusters.
    num_clusters = int(input("Based on the elbow plot, enter the desired number of clusters: "))

    tasks, kmeans, _ = cluster_tasks(tasks, num_clusters)
    tasks, cluster_labels = label_clusters_with_lda(tasks, num_topics=1)

    pretty_print_tasks(tasks, title="Tasks with Categories:")


if __name__ == "__main__":
    main()
