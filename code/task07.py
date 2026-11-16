import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

def extract_features(word, N=2):
    return {'feature': word[-N:].lower()}

data = [(n,'male') for n in names.words('male.txt')] + \
       [(n,'female') for n in names.words('female.txt')]
random.seed(7)
random.shuffle(data)

test_names = ['Alexander','Maria','Mark','Amelia','Clark','Genelia','Ivan','Anastasia']
split = int(0.8*len(data))
train_raw, test_raw = data[:split], data[split:]

for N in range(1, 6):
    tr = [(extract_features(n, N), g) for n, g in train_raw]
    te = [(extract_features(n, N), g) for n, g in test_raw]
    clf = NaiveBayesClassifier.train(tr)
    print(f"\n=== N = {N}: accuracy = {nltk_accuracy(clf, te)*100:.2f}% ===")
    for name in test_names:
        print(f"  {name:>10} -> {clf.classify(extract_features(name, N))}")
