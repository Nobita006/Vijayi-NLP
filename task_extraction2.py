#!/usr/bin/env python3
"""
Improved NLP Pipeline for Extracting and Categorizing Tasks from Unstructured Text

Improvements made:
    - Takes input text from a file specified via command-line argument.
    - Uses argparse for configurable parameters.
    - Uses pretty-printing for readable output.
    - Contains improved heuristics with additional dependency parsing hints.
    - Has modular structure with clear function definitions.
"""

import re
import json
import string
import argparse
import numpy as np
from pprint import pprint

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.cluster import KMeans
from gensim import corpora, models

# Load the spaCy model with medium-sized word vectors.
nlp = spacy.load('en_core_web_md')


##############################################
# Utility Functions
##############################################
def load_text_from_file(filepath):
    """Read text from a given file."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        return text.strip()
    except Exception as e:
        raise IOError(f"Error reading {filepath}: {e}")


def pretty_print_tasks(tasks, title="Tasks:"):
    """Pretty print a list of task dictionaries."""
    print(f"\n{title}")
    print(json.dumps(tasks, indent=4))


##############################################
# Preprocessing
##############################################
def clean_text(text):
    """
    Clean the text by stripping whitespace and removing extraneous punctuation.
    (Further cleaning can be added if necessary.)
    """
    return text.strip()


##############################################
# Task Identification & Extraction
##############################################
def is_task_sentence(sentence):
    """
    Identify if a sentence is likely to be a task using heuristics:
      - Check if the sentence starts with an imperative verb.
      - Check for common task-related keywords/phrases.
    """
    sent_text = sentence.text.strip()
    sent_lower = sent_text.lower()

    # Keywords indicating a task
    task_keywords = ['has to', 'need to', 'needs to', 'should', 'must', 'please', "don't forget"]

    # Heuristic 1: Check if the sentence begins with a verb in base form.
    first_token = sentence[0]
    if first_token.pos_ == 'VERB' and first_token.tag_ == 'VB':
        return True

    # Heuristic 2: Look for task-indicating phrases.
    for keyword in task_keywords:
        if keyword in sent_lower:
            return True

    # Additional heuristic: Use dependency parsing to check if the root is imperative.
    # (If the root verb's mood is "IMP", that can indicate an imperative.)
    if sentence.root.tag_ in ('VB', 'VBP') and sentence.root.dep_ == 'ROOT':
        if any(child.dep_ == 'aux' and child.lower_ in ['do', 'please'] for child in sentence.root.children):
            return True

    return False


def extract_deadline(sentence):
    """
    Extract deadline information by combining adjacent DATE or TIME entities.
    For example, in "5 pm today", both tokens might be recognized separately.
    """
    deadline = None
    deadline_tokens = []
    for ent in sentence.ents:
        if ent.label_ in ["TIME", "DATE"]:
            deadline_tokens.append(ent.text)
    if deadline_tokens:
        # Join tokens with a space.
        deadline = " ".join(deadline_tokens)
    return deadline


def extract_task_details(sentence):
    """
    Extract the task details from a sentence:
        - The task text.
        - The performer (if a PERSON entity is found).
        - The deadline using improved extraction.
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
    Process the text: clean it, split into sentences, and extract tasks.
    Returns a list of dictionaries, each containing a task and its details.
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
# Task Categorization
##############################################
def cluster_tasks(tasks, num_clusters=3):
    """
    Cluster tasks based on their sentence embeddings using KMeans.
    Each task gets assigned a cluster label.
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
    return tasks, kmeans


def label_clusters_with_lda(tasks, num_topics=1):
    """
    For each cluster, perform topic modeling (LDA) on the tokenized task texts
    to derive a category label from the top topic words.
    """
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
        label = " ".join([word for word, _ in topic_terms])
        cluster_labels[cluster] = label

    for task in tasks:
        task["category"] = cluster_labels[task["cluster"]]
    return tasks, cluster_labels


##############################################
# Main function with Argument Parsing
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Extract and categorize tasks from unstructured text."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input text file (e.g., input.txt)."
    )
    parser.add_argument(
        "--clusters", "-c", type=int, default=3,
        help="Number of clusters to use for categorizing tasks (default: 3)."
    )
    args = parser.parse_args()

    try:
        text = load_text_from_file(args.input)
    except IOError as e:
        print(e)
        return

    print("Input Text:")
    print(text)

    # Process the text and extract tasks.
    tasks = process_text(text)
    pretty_print_tasks(tasks, title="Extracted Tasks:")

    if tasks:
        tasks, kmeans = cluster_tasks(tasks, num_clusters=args.clusters)
        tasks, cluster_labels = label_clusters_with_lda(tasks, num_topics=1)
        pretty_print_tasks(tasks, title="Tasks with Categories:")
    else:
        print("No tasks found in the input.")


if __name__ == "__main__":
    main()
