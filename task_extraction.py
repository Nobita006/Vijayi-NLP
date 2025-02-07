#!/usr/bin/env python3
"""
Improved NLP Pipeline for Extracting and Categorizing Tasks from Unstructured Text

Guidelines Implemented:
1. Preprocessing:
   • Clean the text by removing stop words, punctuation, and irrelevant metadata.
   • Tokenize sentences and perform POS tagging to identify actionable verbs (e.g., "schedule," "discuss," "review").
2. Task Identification:
   • Use heuristics to flag sentences as tasks. Examples include sentences that start with imperative verbs,
     contain common task phrases, or include actionable verbs (e.g., schedule, discuss, review).
3. Categorization:
   • Use word embeddings (from spaCy’s medium model) with KMeans to cluster tasks.
   • Dynamically define categories using topic modelling (LDA).
4. Output:
   • Generate a structured list of tasks with categories.
   • For tasks that include additional information, list who is to perform the task and when it is due.
"""

import argparse
import json
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim import corpora, models

# Load the spaCy medium English model (includes word vectors, tokenization, POS tagging, NER)
nlp = spacy.load('en_core_web_md')

# Define a set of actionable verbs (lemmas) as examples
ACTIONABLE_VERBS = {"schedule", "discuss", "review", "buy", "clean", "submit", "call", "forget"}


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
    """Pretty-print the list of task dictionaries in JSON format."""
    print(f"\n{title}")
    print(json.dumps(tasks, indent=4))


##############################################
# Preprocessing Functions
##############################################
def preprocess_text(text):
    """
    Preprocess the input text by:
      • Removing irrelevant metadata (e.g., content inside square brackets).
      • Lowercasing the text.
      • Removing punctuation.
      • Removing stop words.
      • Tokenizing the text into sentences and performing POS tagging.
    
    Returns:
      cleaned_text: The processed text with punctuation, stop words, and metadata removed.
      sentences_with_tags: A list of sentences; each sentence is represented as a list of tuples (token, POS, tag).
    """
    # Remove irrelevant metadata (e.g., content inside square brackets)
    text_no_metadata = re.sub(r'\[.*?\]', '', text)
    
    # Tokenize into sentences and get POS tags (using the original text so punctuation helps sentence splitting)
    doc = nlp(text_no_metadata)
    sentences_with_tags = []
    for sent in doc.sents:
        tokens_info = [(token.text, token.pos_, token.tag_) for token in sent]
        sentences_with_tags.append(tokens_info)
    
    # Create a cleaned version of the text for analysis:
    # Lowercase the text
    text_lower = text_no_metadata.lower()
    # Remove punctuation
    text_no_punct = text_lower.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words (using spaCy tokenization)
    doc_clean = nlp(text_no_punct)
    cleaned_tokens = [token.text for token in doc_clean if token.text not in STOP_WORDS]
    cleaned_text = " ".join(cleaned_tokens)
    
    return cleaned_text, sentences_with_tags


##############################################
# Task Identification and Extraction
##############################################
def is_task_sentence(sentence):
    """
    Identify if a sentence likely represents a task using several heuristics:
      • Check if the sentence starts with a verb in base form.
      • Look for common task-related phrases (e.g., "has to", "need to", "should", "must", "please", "don't forget").
      • Check if any token's lemma is in a predefined set of actionable verbs.
    """
    sent_text = sentence.text.strip()
    sent_lower = sent_text.lower()

    # Heuristic: common task-related phrases
    task_keywords = ['has to', 'need to', 'needs to', 'should', 'must', 'please', "don't forget"]

    # Heuristic 1: Check if the first token is a verb in base form (imperative).
    first_token = sentence[0]
    if first_token.pos_ == 'VERB' and first_token.tag_ == 'VB':
        return True

    # Heuristic 2: Check for task-indicating phrases in the sentence.
    for keyword in task_keywords:
        if keyword in sent_lower:
            return True

    # Heuristic 3: Check if any token (verb) is an actionable verb.
    for token in sentence:
        if token.pos_ == 'VERB' and token.lemma_.lower() in ACTIONABLE_VERBS:
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
      • The full task text.
      • The performer (the first PERSON entity encountered, if any).
      • The deadline (if any).
    """
    task_text = sentence.text.strip()
    performer = None

    # Look for the first PERSON entity as the performer.
    for ent in sentence.ents:
        if ent.label_ == "PERSON":
            performer = ent.text
            break

    deadline = extract_deadline(sentence)

    return task_text, performer, deadline


def process_text(text):
    """
    Process the original text (with punctuation preserved for sentence boundaries)
    to extract sentences that likely represent tasks.
    
    Returns a list of dictionaries with task details.
    """
    doc = nlp(text)
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
    For each cluster, perform topic modeling (LDA) on the task texts to derive
    a category label from the dominant topic. Adds a 'category' field to each task.
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


def determine_optimal_clusters_elbow(task_vectors, min_clusters=2, max_clusters=10):
    """
    Use the elbow method to plot inertia for a range of cluster numbers.
    The function adjusts max_clusters if there are fewer samples.
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


def determine_optimal_clusters_silhouette(task_vectors, min_clusters=2, max_clusters=10):
    """
    Compute the average silhouette score for different numbers of clusters
    and return the number of clusters with the highest score.
    Adjust max_clusters to be at most n_samples - 1.
    """
    n_samples = task_vectors.shape[0]
    if n_samples < min_clusters:
        min_clusters = n_samples
    max_clusters = min(max_clusters, n_samples - 1)
    silhouette_scores = []
    cluster_range = range(min_clusters, max_clusters + 1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(task_vectors)
        score = silhouette_score(task_vectors, labels)
        silhouette_scores.append(score)
        print(f"Clusters: {k}, Silhouette Score: {score:.3f}")
    optimal_k = cluster_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")
    return optimal_k


##############################################
# Main Function with Argument Parsing
##############################################
def main():
    parser = argparse.ArgumentParser(
        description="Extract and categorize tasks from unstructured text using advanced preprocessing and clustering."
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to the input text file (e.g., input.txt).")
    parser.add_argument("--min_clusters", type=int, default=2,
                        help="Minimum number of clusters to try (default: 2).")
    parser.add_argument("--max_clusters", type=int, default=10,
                        help="Maximum number of clusters to try (default: 10).")
    parser.add_argument("--method", type=str, default="silhouette",
                        choices=["elbow", "silhouette"],
                        help="Method to determine optimal clusters: 'elbow' (manual) or 'silhouette' (automatic). Default is silhouette.")
    args = parser.parse_args()

    # Load the original text
    try:
        original_text = load_text_from_file(args.input)
    except IOError as e:
        print(e)
        return

    print("Original Input Text:")
    print(original_text)

    # Preprocessing: Clean the text and get tokenization with POS tags.
    cleaned_text, sentences_with_tags = preprocess_text(original_text)
    print("\nCleaned Text (stop words, punctuation, and metadata removed):")
    print(cleaned_text)
    print("\nTokenized Sentences with POS Tags:")
    for i, sent in enumerate(sentences_with_tags):
        print(f"Sentence {i+1}: {sent}")

    # Process the original text (with punctuation preserved) to extract task sentences.
    tasks = process_text(original_text)
    pretty_print_tasks(tasks, title="Extracted Tasks:")

    if not tasks:
        print("No tasks found in the input.")
        return

    # Compute task vectors (using spaCy embeddings) for clustering.
    task_vectors = []
    for task in tasks:
        doc = nlp(task["task"])
        task_vectors.append(doc.vector)
    task_vectors = np.array(task_vectors)

    # Determine the number of clusters using the chosen method.
    if args.method == "elbow":
        print("\nDetermining the optimal number of clusters using the Elbow Method...")
        determine_optimal_clusters_elbow(task_vectors, min_clusters=args.min_clusters, max_clusters=args.max_clusters)
        num_clusters = int(input("Based on the elbow plot, enter the desired number of clusters: "))
    else:
        print("\nDetermining the optimal number of clusters using the Silhouette Score...")
        num_clusters = determine_optimal_clusters_silhouette(task_vectors, min_clusters=args.min_clusters, max_clusters=args.max_clusters)

    # Cluster tasks and assign dynamic categories using LDA.
    tasks, kmeans, _ = cluster_tasks(tasks, num_clusters)
    tasks, cluster_labels = label_clusters_with_lda(tasks, num_topics=1)

    pretty_print_tasks(tasks, title="Tasks with Categories:")


if __name__ == "__main__":
    main()
