import spacy
from nltk.stem.snowball import SnowballStemmer

nlp = spacy.load("es_core_news_sm")
stemmer = SnowballStemmer("spanish")

palabras = ["universidades", "aulas", "solicitudes", "bibliotecas", "horarios"]

for p in palabras:
    print(f"{p} -> Stem: {stemmer.stem(p)} | Lema: {nlp(p)[0].lemma_}")