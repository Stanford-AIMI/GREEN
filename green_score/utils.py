from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import numpy as np


def compute_largest_cluster(sentences):  
    """
    Computes the largest cluster of sentences using K-means clustering, finds the sentences within the largest cluster, and orders them by their distance to the cluster center.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - sentences_of_largest_cluster (list): Sentences in the largest cluster, ordered by their proximity 
              to the cluster center.
    """
    if len(sentences)==0:
        return None, None
    embeddings, kmeans = compute_kmeans(sentences)
    cluster_sizes = np.bincount(kmeans.labels_)
    largest_cluster_idx = np.argmax(cluster_sizes)
    cluster_member_ids = np.where(kmeans.labels_ == largest_cluster_idx)[0]
    sentences_of_largest_cluster = [sentences[i] for i in cluster_member_ids]

    largest_cluster_mean = kmeans.cluster_centers_[largest_cluster_idx]
    embeddings_of_largest_cluster = [embeddings[i] for i in cluster_member_ids]
    distances = distance.cdist(
        embeddings_of_largest_cluster, [largest_cluster_mean], "cosine"
    ).flatten()
    closest_point_indices = np.argsort(distances)[0]

    sentences_of_largest_cluster = sentences_of_largest_cluster[closest_point_indices]

    return embeddings, sentences_of_largest_cluster

def compute_kmeans(sentences):
    """
    Computes K-means clustering for a list of sentences by generating their embeddings, normalizing the embeddings, and determining the optimal number of clusters using binary search.

    Args:
        sentences (list): List of sentences to be clustered.

    Returns:
        tuple: A tuple containing:
            - embeddings (ndarray): Normalized embeddings of the input sentences.
            - kmeans (KMeans): The KMeans object with the optimal number of clusters determined.
    """
    # sentence embeddings
    model = SentenceTransformer("sentence-transformers/paraphrase-mpnet-base-v2")
    embeddings = model.encode(sentences)
    # normalize the embeddings for equivalent computation of the cosine distance
    embeddings = preprocessing.normalize(embeddings)
    # compute the number of clusters with binary search
    kmeans = binary_search_optimal_kmeans(embeddings, min_k=0, max_k=len(sentences))
    return embeddings, kmeans

def binary_search_optimal_kmeans(data, min_k, max_k):
    """
    Finds the optimal k for KMeans clustering using binary search on the silhouette score.
    
    Args:
        data (list): cluster data.
        min_k: minimum k for binary search
        max_k: maximum k for binary search

    Returns:
        list: List of cleaned response strings.
    """
    best_k = min_k
    best_score = -1
    best_kmeans = KMeans(n_clusters=1, random_state=42).fit(data)  # start with 1 cluster for len(data) < 2

    while min_k <= max_k:
        mid_k = (min_k + max_k) // 2
        if mid_k < 2:
            break

        kmeans = KMeans(n_clusters=mid_k, random_state=42).fit(data)
        labels = kmeans.labels_
        score = silhouette_score(data, labels)
                
        if score > best_score:
            best_score = score
            best_k = mid_k
            best_kmeans = kmeans  # Update the best KMeans model
            min_k = mid_k + 1
        else:
            max_k = mid_k - 1

    print(f"Optimal k found: {best_k} with silhouette score: {best_score}")
    print(len(data))
    import ipdb; ipdb.set_trace()
    return best_kmeans

def flatten_values_lists_of_list_dicts_to_dict(item):
    """
    Flattens a list of dictionaries containing lists of values into a single dictionary.

    Args:
        item (list): List of dictionaries, where each dictionary's values are lists. If any element of the list is itself a list, the function will consider only the first dictionary in that sublist.

    Returns:
        dict: A dictionary where each key corresponds to the keys in the input dictionaries, and each value is a flattened list of all values associated with that key across all input dictionaries.
    """

    result = {}
    for i in item:
        if isinstance(i, list):
            i = i[0]
        for key, lists in i.items():
            if key not in result:
                result[key] = []
            result[key].extend(lists)

    return result

def process_responses(responses):
    """
    Processes a list of responses by removing unwanted tokens and returns the cleaned responses.

    Args:
        responses (list): List of response strings.

    Returns:
        list: List of cleaned response strings.
    """
    response_list = []
    for i in responses:
        if "<|assistant|>" in i:
            i = i.split("<|assistant|>")[-1]
            i = i.replace("</s>", "").replace("<unk>", "")
        response_list.append(i)
    return response_list


def make_prompt(text1, text2):
    """
    Creates a prompt for evaluating the accuracy of a candidate radiology report in comparison to a reference radiology report.

    Args:
        text1 (str): Reference radiology report.
        text2 (str): Candidate radiology report.

    Returns:
        str: Formatted prompt string.
    """
    prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
    return prompt


def tokenize_batch_as_chat(tokenizer, batch):
    """
    Tokenizes a batch of prompts as chat input for the model.

    Args:
        tokenizer: Tokenizer object for encoding the batch.
        batch (list): List of prompts to be tokenized.

    Returns:
        dict: Tokenized batch with input IDs and attention masks.
    """
    batch = [
        tokenizer.apply_chat_template(
            i, tokenize=False, add_generation_prompt=True
        )
        for i in batch
    ]

    batch = tokenizer.batch_encode_plus(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    return batch


def truncate_to_max_len(sentences, max_len):
    return [" ".join(sentence.split()[:max_len]) for sentence in sentences]
