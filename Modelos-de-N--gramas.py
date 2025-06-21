import nltk
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

with open("CorpusEducacion.txt", encoding="latin-1") as archivo:
    texto = archivo.read()

texto = texto.lower()
stop_words = set(stopwords.words("spanish"))
oraciones = texto.split(".")

def limpiar_texto(texto):
    texto = texto.strip()
    texto = re.sub(f"[{string.punctuation}]", " ", texto)
    palabras = texto.split()
    palabras_limpias = [p for p in palabras if p not in stop_words and p.isalpha()]
    return " ".join(palabras_limpias)

oraciones_limpias = [limpiar_texto(ora) for ora in oraciones if len(ora.strip()) > 0]

vectorizer2 = CountVectorizer(ngram_range=(2,2), min_df=2)
X2 = vectorizer2.fit_transform(oraciones_limpias)
bigrama_frecuencias = X2.toarray().sum(axis=0)
bigrama_vocabulario = vectorizer2.get_feature_names_out()

vectorizer3 = CountVectorizer(ngram_range=(3,3), min_df=2)
X3 = vectorizer3.fit_transform(oraciones_limpias)
trigrama_frecuencias = X3.toarray().sum(axis=0)
trigrama_vocabulario = vectorizer3.get_feature_names_out()

print("Top 10 Bigramas:")
for i in bigrama_frecuencias.argsort()[::-1][:10]:
    print(bigrama_vocabulario[i], "->", bigrama_frecuencias[i])

print("\nTop 10 Trigramas:")
for i in trigrama_frecuencias.argsort()[::-1][:10]:
    print(trigrama_vocabulario[i], "->", trigrama_frecuencias[i])

plt.figure(figsize=(10,5))
plt.barh(bigrama_vocabulario[bigrama_frecuencias.argsort()[::-1][:10]], 
         bigrama_frecuencias[bigrama_frecuencias.argsort()[::-1][:10]],
         color='yellow')
plt.title("Top 10 Bigramas")
plt.xlabel("Frecuencia")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.barh(trigrama_vocabulario[trigrama_frecuencias.argsort()[::-1][:10]], 
         trigrama_frecuencias[trigrama_frecuencias.argsort()[::-1][:10]],
         color='green')
plt.title("Top 10 Trigramas")
plt.xlabel("Frecuencia")
plt.tight_layout()
plt.show()
