import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_glove_embeddings(file_path="D:\SEM PROJECTS\SEM 2\EOC-2 and MFC-2\Code\glove.6B.100d.txt"):
    word_embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings

word_embeddings = load_glove_embeddings()

def preprocess_text(text):
    if pd.isnull(text): 
        return ""
    text = text.lower()
    text = ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])
    return text

def remove_stopwords(sentence):
    return " ".join([word for word in sentence.split() if word not in stop_words])

def sentence_vector(sentence):
    words = sentence.split()
    if len(words) == 0:
        return np.zeros((100,))
    return np.mean([word_embeddings.get(w, np.zeros((100,))) for w in words], axis=0)

def generate_summary(csv_file, num_sentences=5):
    csv_file.seek(0)
    df = pd.read_csv(csv_file)
    
    if 'ReviewContent' not in df.columns:
        return "CSV file should contain a 'reviewDescription' column."
    
    df = df.dropna(subset=['ReviewContent'])
    df['ReviewContent'] = df['ReviewContent'].astype(str).str.strip()
    df = df[df['ReviewContent'] != ""]  # Remove empty strings
    df = df[df['ReviewContent'].str.lower() != "nan"]
    sentences = []
    for review in df['ReviewContent']:
        sentences.extend(sent_tokenize(str(review)))  

    if len(sentences) == 0:
        return "No valid text to summarize."

    clean_sentences = [remove_stopwords(preprocess_text(sentence)) for sentence in sentences]

    sentence_vectors = np.array([sentence_vector(sentence) for sentence in clean_sentences])

    sim_mat = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summary = "\n".join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    
    return summary
