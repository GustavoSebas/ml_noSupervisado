import strawberry
from typing import Optional
from src.core.clustering import (
    load_tasks, run_clustering, detect_duplicates, compute_embeddings,
    build_text, short_preview, _MODEL_NAME
)
# Note: db.py is in src/db/db.py
import sys
# A bit of a hack to ensure imports work if running from root, but usually direct import works
from src.db.db import get_engine
from src.graphql.types import ClusterResponse, DuplicateResponse, Cluster, ClusterMember, DuplicatePair, DuplicateSimilar, DuplicateSimplified

@strawberry.type
class Query:
    @strawberry.field
    def get_clusters(self, 
                     project_id: int, 
                     algo: str = "kmeans", 
                     k: int = 6, 
                     min_cluster_size: int = 5, 
                     use_umap: bool = True) -> ClusterResponse:
        
        eng = get_engine()
        df = load_tasks(eng, project_id)
        
        if df.empty:
            return ClusterResponse(
                project_id=project_id,
                model=_MODEL_NAME,
                algo=algo,
                k=None,
                min_cluster_size=None,
                use_umap=use_umap,
                clusters=[],
                message="sin tareas"
            )

        clusters_list, _, _, _, _ = run_clustering(
            df, algo=algo, k=k, min_cluster_size=min_cluster_size, use_umap=use_umap
        )

        # Mapping dictionaries to Strawberry Types
        # run_clustering returns list of dicts suitable for JSON, but we must instantiate objects for Strawberry > 0.1
        # Or depend on Strawberry auto-mapping from dict if structure matches exact type keys. 
        # For safety and Clean Code, we instantiate.

        final_clusters = []
        for c in clusters_list:
            members_obj = [
                ClusterMember(
                    tarea_id=m["tarea_id"],
                    titulo=m["titulo"],
                    prioridad=m["prioridad"],
                    puntos=m["puntos"],
                    texto=m["texto"],
                    coord=m["coord"]
                ) for m in c["members"]
            ]
            
            final_clusters.append(Cluster(
                cluster_id=c["cluster_id"],
                label_keywords=c["label_keywords"],
                label=c["label"],
                size=c["size"],
                sample_titles=c["sample_titles"],
                members=members_obj
            ))

        return ClusterResponse(
            project_id=project_id,
            model=_MODEL_NAME,
            algo=algo,
            k=(k if algo.lower() == "kmeans" else None),
            min_cluster_size=(min_cluster_size if algo.lower() == "hdbscan" else None),
            use_umap=use_umap,
            clusters=final_clusters,
            message=None
        )

    @strawberry.field
    def get_duplicates(self, 
                       project_id: int, 
                       threshold: float = 0.90, 
                       top_n: int = 5) -> DuplicateResponse:
        
        eng = get_engine()
        df = load_tasks(eng, project_id)
        
        if df.empty:
            return DuplicateResponse(
                project_id=project_id,
                model=_MODEL_NAME,
                threshold=threshold,
                top_n=top_n,
                duplicates=[],
                message="sin tareas"
            )

        texts = [build_text(str(df.iloc[i]['titulo']), str(df.iloc[i]['descripcion'])) for i in range(len(df))]
        previews = [short_preview(df.iloc[i]['titulo'], df.iloc[i]['descripcion']) for i in range(len(df))]
        titles = df['titulo'].astype(str).tolist()
        ids = df['tarea_id'].astype(int).tolist()

        emb = compute_embeddings(texts)
        pairs_data = detect_duplicates(
            emb=emb,
            task_ids=ids,
            titles=titles,
            previews=previews,
            threshold=threshold,
            top_n=top_n
        )

        # Mapping to Types
        duplicates_obj = []
        for p in pairs_data:
            similares_obj = [
                DuplicateSimilar(
                    tareaId=s["tareaId"],
                    titulo=s["titulo"],
                    similitud=s["similitud"],
                    similitudPct=s["similitudPct"]
                ) for s in p["similares"]
            ]
            possible_obj = [
                 DuplicateSimplified(
                     tareaId=d["tareaId"],
                     sim=d["sim"]
                 ) for d in p["posibles_duplicados"]
            ]

            duplicates_obj.append(DuplicatePair(
                tareaId=p["tareaId"],
                titulo=p["titulo"],
                texto=p["texto"],
                conteo=p["conteo"],
                similares=similares_obj,
                posibles_duplicados=possible_obj
            ))

        return DuplicateResponse(
            project_id=project_id,
            model=_MODEL_NAME,
            threshold=threshold,
            top_n=top_n,
            duplicates=duplicates_obj,
            message=None
        )

schema = strawberry.Schema(query=Query)
