import os
import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer

nlp = spacy.load("es_core_news_sm")
stemmer = SnowballStemmer("spanish")


def lematizar(texto):
    return " ".join(
        [t.lemma_ for t in nlp(str(texto).lower()) if t.is_alpha]
    )


def stemmar(texto):
    return " ".join([stemmer.stem(p) for p in str(texto).lower().split()])


df = pd.read_csv("dataset_tito_v1.csv", encoding="utf-8")
os.makedirs("resultados", exist_ok=True)

df["texto_stem"] = df["texto"].apply(stemmar)
df["texto_lemma"] = df["texto"].apply(lematizar)


df[["texto", "texto_lemma"]].to_csv(
    "resultados/dataset_tito_lemma.csv", index=False, encoding="utf-8-sig"
)

pd.DataFrame(
    {
        "original": df["texto"],
        "stemming": df["texto_stem"],
        "lematizacion": df["texto_lemma"],
    }
).to_csv(
    "resultados/comparacion_stem_lemma.csv", index=False, encoding="utf-8-sig"
)

print(df[["texto", "texto_stem", "texto_lemma"]].head(10))