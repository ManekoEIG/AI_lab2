from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

def extract_features(words):
    return dict([(w, True) for w in words])

pos_ids, neg_ids = movie_reviews.fileids('pos'), movie_reviews.fileids('neg')
fp = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in pos_ids]
fn = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in neg_ids]

num_pos, num_neg = int(0.8*len(fp)), int(0.8*len(fn))
features_train = fp[:num_pos] + fn[:num_neg]
features_test = fp[num_pos:] + fn[num_neg:]
print("Train:", len(features_train), "Test:", len(features_test))

clf = NaiveBayesClassifier.train(features_train)
print("Accuracy:", nltk_accuracy(clf, features_test))
print("\nTop-20 informative words:")
clf.show_most_informative_features(20)

reviews = [
    "This movie was absolutely fantastic, the acting was brilliant",
    "The plot was boring and the characters were terrible",
    "An outstanding film with a touching story",
    "I hated this awful movie, a complete waste of time",
    "The film was okay, nothing special but not bad either",
]
print()
for r in reviews:
    pr = clf.prob_classify(extract_features(r.split()))
    print(f"{r}\n  -> {pr.max()} (pos={pr.prob('Positive'):.3f}, neg={pr.prob('Negative'):.3f})")
