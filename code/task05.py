import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown
from task04 import chunker

N, M = 12000, 1500
chunks = chunker(brown.words()[:N], M)
input_data = [{'index': i, 'text': c} for i, c in enumerate(chunks)]
texts = [d['text'] for d in input_data]

vec = CountVectorizer(min_df=5, max_df=0.9)
matrix = vec.fit_transform(texts)
vocab = vec.get_feature_names_out()

print("Размер матрицы (чанки x слова):", matrix.shape)
sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
print(f"Доля нулей: {sparsity:.2%}")
df = pd.DataFrame(matrix.toarray(), columns=vocab,
                  index=[f"chunk_{i}" for i in range(len(texts))])
print(df.iloc[:, :15].to_string())
print("\nТоп-10 слов по коллекции:")
print(df.sum().sort_values(ascending=False).head(10).to_string())
