import nltk
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist

with open("input.txt", "r") as f:
    text = f.read()

nltk.download('punkt_tab')

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