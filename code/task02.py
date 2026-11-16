from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

words = ['writing','calves','be','branded','horse','randomize',
         'possibly','provision','hospital','kept','scratchy','code']
p, l, s = PorterStemmer(), LancasterStemmer(), SnowballStemmer('english')
print(f"{'WORD':>12} {'PORTER':>12} {'LANCASTER':>12} {'SNOWBALL':>12}")
for w in words:
    print(f"{w:>12} {p.stem(w):>12} {l.stem(w):>12} {s.stem(w):>12}")
