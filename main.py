import os

import nltk
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist

def find_ngrams(input_file):
    if not os.path.exists(input_file):
        print(f"Входной файл {input_file} не найден")
        return

    with open(input_file, "r") as f:
        text = f.read()

    tokens = word_tokenize(text.lower())

    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    freq_bigrams = FreqDist(bigrams)
    freq_trigrams = FreqDist(trigrams)

    print("Топ-5 биграмм:")
    for bigram, count in freq_bigrams.most_common(5):
        print(f"{' '.join(bigram)}: {count}")

    print("\nТоп-5 триграмм:")
    for trigram, count in freq_trigrams.most_common(5):
        print(f"{' '.join(trigram)}: {count}")


nltk.download('punkt_tab')

input_path = "input.txt"

find_ngrams(input_path)