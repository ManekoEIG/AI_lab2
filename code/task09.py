from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

def load_data(path):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def process(text):
    tok = RegexpTokenizer(r'[a-zA-Z]+')
    stop = set(stopwords.words('english'))
    st = SnowballStemmer('english')
    t = [w for w in tok.tokenize(text.lower()) if w not in stop and len(w) > 2]
    return [st.stem(w) for w in t]

lines = load_data('lab02_data.txt')
tokens = [process(l) for l in lines]
print("Токены предложений:")
for i, t in enumerate(tokens):
    print(i, t)

dict_tokens = corpora.Dictionary(tokens)
doc_term_mat = [dict_tokens.doc2bow(t) for t in tokens]
lda = models.ldamodel.LdaModel(doc_term_mat, num_topics=2, id2word=dict_tokens,
                               passes=25, random_state=42)

print("\nТемы (по 5 слов):")
for i, topic in lda.print_topics(num_words=5):
    print(f"Topic {i}: {topic}")

print("\nРаспределение тем по предложениям:")
for i, bow in enumerate(doc_term_mat):
    d = lda.get_document_topics(bow, minimum_probability=0)
    print(f"Предложение {i+1}: " + ", ".join(f"T{t}={p:.2f}" for t, p in d)
          + f"   | {lines[i][:45]}...")
