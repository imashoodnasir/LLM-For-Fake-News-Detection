from gensim.models import Word2Vec

# Create Word2Vec embeddings
def create_word2vec_embeddings(data):
    sentences = [text.split() for text in data['text']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    data['embeddings'] = data['text'].apply(lambda x: [model.wv[word] for word in x.split() if word in model.wv])
    return data

# Example usage
data = create_word2vec_embeddings(data)
print(data[['text', 'embeddings']].head())