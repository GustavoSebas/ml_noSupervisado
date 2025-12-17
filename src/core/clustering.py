# clustering.py
# ------------------------------------
# Utilidades de clustering y duplicados con salidas legibles y robustas

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

# opcionales:
try:
    import hdbscan  # noqa
except Exception:
    hdbscan = None

try:
    import umap  # noqa
except Exception:
    umap = None

SQL_TASKS = """
SELECT t.id AS tarea_id,
       t.proyecto_id,
       COALESCE(t.titulo,'')       AS titulo,
       COALESCE(t.descripcion,'')  AS descripcion,
       COALESCE(t.prioridad,'')    AS prioridad,
       COALESCE(t.puntos,0)        AS puntos
FROM tarea t
WHERE t.proyecto_id = :pid
  AND (t.titulo IS NOT NULL OR t.descripcion IS NOT NULL)
"""

# ---------------------------
# Stopwords español (lista embebida)
# ---------------------------
ES_STOPWORDS = [
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con","no",
    "una","su","al","lo","como","más","pero","sus","le","ya","o","este","ha","sí","porque",
    "esta","son","entre","cuando","muy","sin","sobre","también","me","hasta","hay","donde",
    "quien","desde","todo","nos","durante","todos","uno","les","ni","contra","otros","ese",
    "eso","ante","ellos","e","esto","mí","antes","algunos","qué","unos","yo","otro","otras",
    "otra","él","tanto","esa","estos","mucho","quienes","nada","muchos","cual","poco","ella",
    "estar","estas","algunas","algo","nosotros","mi","mis","tú","te","ti","tu","tus","ellas",
    "nosotras","vosotros","vosotras","os","mío","mía","míos","mías","tuyo","tuya","tuyos",
    "tuyas","suyo","suya","suyos","suyas","nuestro","nuestra","nuestros","nuestras","vuestro",
    "vuestra","vuestros","vuestras","esos","esas","estoy","estás","está","estamos","estáis",
    "están","esté","estés","estemos","estéis","estén","estaré","estarás","estará","estaremos",
    "estaréis","estarán","estaba","estabas","estábamos","estabais","estaban","estuve",
    "estuviste","estuvo","estuvimos","estuvisteis","estuvieron","estuviera","estuvieras",
    "estuviéramos","estuvierais","estuvieran","estuviese","estuvieses","estuviésemos",
    "estuvieseis","estuviesen","estando","estado","estada","estados","estadas","estad","he",
    "has","ha","hemos","habéis","han","haya","hayas","hayamos","hayáis","hayan","habré",
    "habrás","habrá","habremos","habréis","habrán","había","habías","habíamos","habíais",
    "habían","hube","hubiste","hubo","hubimos","hubisteis","hubieron","hubiera","hubieras",
    "hubiéramos","hubierais","hubieran","hubiese","hubieses","hubiésemos","hubieseis",
    "hubiesen","habiendo","habido","habida","habidos","habidas","soy","eres","es","somos",
    "sois","son","sea","seas","seamos","seáis","sean","seré","serás","será","seremos","seréis",
    "serán","era","eras","éramos","erais","eran","fui","fuiste","fue","fuimos","fuisteis",
    "fueron","fuera","fueras","fuéramos","fuerais","fueran","fuese","fueses","fuésemos",
    "fueseis","fuesen","siendo","sido","tengo","tienes","tiene","tenemos","tenéis","tienen",
    "tenga","tengas","tengamos","tengáis","tengan","tendré","tendrás","tendrá","tendremos",
    "tendréis","tendrán","tenía","tenías","teníamos","teníais","tenían","tuve","tuviste",
    "tuvo","tuvimos","tuvisteis","tuvieron","tuviera","tuvieras","tuviéramos","tuvierais",
    "tuvieran","tuviese","tuvieses","tuviésemos","tuvieseis","tuviesen","teniendo","tenido",
    "tenida","tenidos","tenidas","que"
]

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model_cache = None

def get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(_MODEL_NAME)
    return _model_cache

def normalize_text(s: str) -> str:
    return " ".join((s or "").lower().split())

def build_text(title: str, desc: str) -> str:
    t = normalize_text(title)
    d = normalize_text(desc)
    return (t + ". " + d).strip(". ").strip()

def short_preview(title: str, desc: str, limit: int = 160) -> str:
    base = (title or "").strip()
    if desc and base:
        base = f"{base} — {desc.strip()}"
    elif desc:
        base = desc.strip()
    base = " ".join(base.split())
    return base[:limit] + ("…" if len(base) > limit else "")

def compute_embeddings(texts: list[str]) -> np.ndarray:
    model = get_model()
    # show_progress_bar=False para Logs más limpios en prod
    emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.array(emb)

def keyword_labels(texts: list[str], labels: np.ndarray, top_k=5):
    """
    Etiquetas por cluster vía TF-IDF. Robustas a documentos vacíos o sólo-stopwords.
    """
    df = pd.DataFrame({'text': texts, 'label': labels})
    kws = {}
    for lbl in np.unique(labels):
        if lbl == -1:
            kws[lbl] = ["ruido"]
            continue

        subset = df[df.label == lbl]['text'].tolist()
        if not subset:
            kws[lbl] = []
            continue

        vec = TfidfVectorizer(max_features=4000, stop_words=list(ES_STOPWORDS))
        try:
            X = vec.fit_transform(subset)
            # si no quedó vocabulario (todo eran stopwords)
            if X.shape[1] == 0:
                kws[lbl] = []
                continue
            idxs = np.argsort(X.sum(axis=0).A1)[::-1][:top_k]
            vocab = np.array(vec.get_feature_names_out())
            kws[lbl] = vocab[idxs].tolist()
        except ValueError:
            # “empty vocabulary” u otro caso raro → sin keywords
            kws[lbl] = []
    return kws

def detect_duplicates(emb: np.ndarray,
                      task_ids: list[int],
                      titles: list[str],
                      previews: list[str],
                      threshold=0.85,
                      top_n=5):
    S = cosine_similarity(emb)
    pairs = []
    n = len(task_ids)
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            score = float(S[i, j])
            if score >= threshold:
                sims.append((int(task_ids[j]), score))
        sims.sort(key=lambda x: x[1], reverse=True)

        similares = [{
            "tareaId": sid,
            "titulo": titles[task_ids.index(sid)] if sid in task_ids else "",
            "similitud": sc,
            "similitudPct": int(round(sc * 100))
        } for sid, sc in sims[:top_n]]

        posibles_compat = [{"tareaId": sid, "sim": sc} for sid, sc in sims[:top_n]]

        pairs.append({
            "tareaId": int(task_ids[i]),
            "titulo": titles[i],
            "texto": previews[i],
            "conteo": len(similares),
            "similares": similares,
            "posibles_duplicados": posibles_compat
        })
    return pairs

def run_clustering(df: pd.DataFrame,
                   algo="kmeans",
                   k=6,
                   min_cluster_size=5,
                   use_umap=True):
    # Textos normalizados para embeddings
    n = len(df)
    if n == 0:
        return [], np.zeros((0, 0)), [], [], []

    texts_norm = [
        build_text(str(df.iloc[i]['titulo']), str(df.iloc[i]['descripcion']))
        for i in range(n)
    ]
    ids = df['tarea_id'].astype(int).tolist()
    titles = df['titulo'].astype(str).tolist()
    previews = [short_preview(df.iloc[i]['titulo'], df.iloc[i]['descripcion']) for i in range(n)]

    emb = compute_embeddings(texts_norm)

    coords = None
    if use_umap and umap is not None and len(emb) >= 5:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        # convert numpy array to list for JSON serialization
        coords = reducer.fit_transform(emb).tolist()

    # Etiquetado por algoritmo (robusto a pocos puntos)
    algo_lower = (algo or "kmeans").lower()
    if algo_lower == "hdbscan" and hdbscan is not None:
        if len(emb) >= 2:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size or 5, metric='euclidean')
            labels = clusterer.fit_predict(emb)
        else:
            labels = np.zeros(len(emb), dtype=int)
    else:
        # kmeans por defecto
        if len(emb) < 2:
            labels = np.zeros(len(emb), dtype=int)
        else:
            k_eff = k if (k and k >= 2) else 2
            k_eff = min(k_eff, len(emb))
            clusterer = KMeans(n_clusters=k_eff, n_init="auto", random_state=42)
            labels = clusterer.fit_predict(emb)

    kw = keyword_labels(texts_norm, labels, top_k=5)

    clusters = []
    for lbl in sorted(np.unique(labels)):
        idxs = np.where(labels == lbl)[0].tolist()
        members = []
        for pos in idxs:
            members.append({
                "tarea_id": ids[pos],
                "titulo": titles[pos],
                "prioridad": df.iloc[pos]['prioridad'],
                "puntos": int(df.iloc[pos]['puntos']),
                "texto": previews[pos],
                # Ensure coord is serializable or None
                "coord": (coords[pos] if coords is not None and pos < len(coords) else None)
            })

        sample_titles = [titles[pos] for pos in idxs[:3]]
        label_keywords = kw.get(lbl, [])
        clusters.append({
            "cluster_id": int(lbl),
            "label_keywords": label_keywords,
            "label": ", ".join(label_keywords) if label_keywords else "",
            "size": len(idxs),
            "sample_titles": sample_titles,
            "members": members
        })

    return clusters, emb, ids, titles, previews

def load_tasks(engine, project_id: int):
    return pd.read_sql(text(SQL_TASKS), engine, params={"pid": project_id})
