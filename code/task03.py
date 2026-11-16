from nltk.stem import WordNetLemmatizer

words = ['writing','calves','be','branded','horse','randomize',
         'possibly','provision','hospital','kept','scratchy','code']
lem = WordNetLemmatizer()
print(f"{'WORD':>12} {'NOUN':>12} {'VERB':>12}")
for w in words:
    print(f"{w:>12} {lem.lemmatize(w, pos='n'):>12} {lem.lemmatize(w, pos='v'):>12}")
