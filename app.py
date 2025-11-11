# app.py
# ------------------------------------
from fastapi import FastAPI, Query
from db import get_engine
from clustering import (
    load_tasks, run_clustering, detect_duplicates, compute_embeddings,
    build_text, short_preview, _MODEL_NAME
)

app = FastAPI(title="ML Topics Service", version="1.1.0")

@app.get("/health")
def health():
    return {"ok": True, "model": _MODEL_NAME}

@app.get("/clusters")
def clusters(project_id: int = Query(..., description="ID del proyecto"),
             algo: str = Query("kmeans", description="kmeans | hdbscan"),
             k: int = Query(6, description="Solo para kmeans"),
             min_cluster_size: int = Query(5, description="Solo para hdbscan"),
             use_umap: bool = Query(True, description="Añade coords 2D con UMAP")):
    eng = get_engine()
    df = load_tasks(eng, project_id)
    if df.empty:
        return {
            "project_id": project_id,
            "model": _MODEL_NAME,
            "clusters": [],
            "message": "sin tareas"
        }

    clusters_list, emb, ids, titles, previews = run_clustering(
        df, algo=algo, k=k, min_cluster_size=min_cluster_size, use_umap=use_umap
    )

    return {
        "project_id": project_id,
        "model": _MODEL_NAME,
        "algo": algo,
        "k": (k if algo.lower() == "kmeans" else None),
        "min_cluster_size": (min_cluster_size if algo.lower() == "hdbscan" else None),
        "use_umap": use_umap,
        "clusters": clusters_list
    }

@app.get("/duplicates")
def duplicates(project_id: int = Query(..., description="ID del proyecto"),
               threshold: float = Query(0.90, description="Umbral de similitud coseno (0..1)"),
               top_n: int = Query(5, description="Máximo de similares por tarea")):
    eng = get_engine()
    df = load_tasks(eng, project_id)
    if df.empty:
        return {
            "project_id": project_id,
            "model": _MODEL_NAME,
            "threshold": threshold,
            "top_n": top_n,
            "duplicates": []
        }

    # Textos normalizados y previas legibles
    texts = [build_text(str(df.iloc[i]['titulo']), str(df.iloc[i]['descripcion'])) for i in range(len(df))]
    previews = [short_preview(df.iloc[i]['titulo'], df.iloc[i]['descripcion']) for i in range(len(df))]
    titles = df['titulo'].astype(str).tolist()
    ids = df['tarea_id'].astype(int).tolist()

    emb = compute_embeddings(texts)
    pairs = detect_duplicates(
        emb=emb,
        task_ids=ids,
        titles=titles,
        previews=previews,
        threshold=threshold,
        top_n=top_n
    )

    return {
        "project_id": project_id,
        "model": _MODEL_NAME,
        "threshold": threshold,
        "top_n": top_n,
        "duplicates": pairs
    }
