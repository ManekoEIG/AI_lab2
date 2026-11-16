from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

category_map = {'rec.sport.hockey':'Hockey','sci.space':'Space',
                'talk.politics.misc':'Politics','sci.electronics':'Electronics',
                'rec.autos':'Cars'}
print("Категории:", category_map)

train = fetch_20newsgroups(subset='train', categories=list(category_map),
                           shuffle=True, random_state=42)
cv = CountVectorizer()
train_counts = cv.fit_transform(train.data)
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_counts)

input_data = [
    "The goalie made an incredible save in the third period of the game",
    "NASA is planning a new mission to explore the surface of Mars",
    "The senator proposed a new law about taxes and healthcare",
    "I replaced the transistor and the resistor on the circuit board",
    "My car needs a new engine and the brakes are making noise",
]
clf = MultinomialNB().fit(train_tfidf, train.target)
# transform (не fit_transform!), чтобы словарь совпадал с обучающим
pred = clf.predict(tfidf.transform(cv.transform(input_data)))
for s, p in zip(input_data, pred):
    print(f"\nInput: {s}\nPredicted: {category_map[train.target_names[p]]}")
