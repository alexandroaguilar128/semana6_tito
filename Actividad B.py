import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer

nlp = spacy.load("es_core_news_sm")
stemmer = SnowballStemmer("spanish")

df = pd.read_csv("dataset_tito_v1.csv", encoding="utf-8")

categorias = ["becas", "biblioteca", "control escolar"]
df_filtrado = df[df["categoria"].str.lower().isin(categorias)].copy()

df_filtrado["stemming"] = df_filtrado["texto"].apply(
    lambda x: " ".join([stemmer.stem(p) for p in str(x).lower().split()])
)
df_filtrado["lematizacion"] = df_filtrado["texto"].apply(
    lambda x: " ".join([t.lemma_ for t in nlp(str(x).lower()) if t.is_alpha])
)

print(
    df_filtrado[["categoria", "texto", "stemming", "lematizacion"]].head(10)
)