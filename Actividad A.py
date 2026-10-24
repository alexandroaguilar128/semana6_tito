import spacy
from nltk.stem.snowball import SnowballStemmer

nlp = spacy.load("es_core_news_sm")
stemmer = SnowballStemmer("spanish")

palabra = input("Escribe una palabra: ")
print("Stem:", stemmer.stem(palabra))
print("Lema:", nlp(palabra)[0].lemma_)

palabras = [
    "estudiantes",
    "bibliotecas",
    "profesores",
    "inscripciones",
    "aulas",
    "exámenes",
    "materias",
    "educación",
    "calificaciones",
    "dirección",
]

for p in palabras:
    print(f"{p} -> Stem: {stemmer.stem(p)} | Lema: {nlp(p)[0].lemma_}")