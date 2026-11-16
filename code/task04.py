from nltk.corpus import brown

def chunker(words, size):
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

if __name__ == "__main__":
    N, M = 12000, 700
    words = brown.words()[:N]
    chunks = chunker(words, M)
    print(f"Слов: {len(words)}, размер чанка: {M}, чанков: {len(chunks)}")
    for i, c in enumerate(chunks[:20]):
        print(f"Chunk {i}: {c[:50]}...")
