from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

text = ("Hello, Mr. Smith! Don't you think NLP isn't hard? "
        "It costs $3.50, e.g. in the U.S.A. Write to test@mail.com.")
print("ИСХОДНЫЙ ТЕКСТ:\n", text)
print("\nsent_tokenize:\n", sent_tokenize(text))
print("\nword_tokenize:\n", word_tokenize(text))
print("\nWordPunctTokenizer:\n", WordPunctTokenizer().tokenize(text))
