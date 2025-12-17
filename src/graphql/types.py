import strawberry
from typing import List, Optional

@strawberry.type
class ClusterMember:
    tarea_id: int
    titulo: str
    prioridad: str
    puntos: int
    texto: str
    coord: Optional[List[float]] = None

@strawberry.type
class Cluster:
    cluster_id: int
    label_keywords: List[str]
    label: str
    size: int
    sample_titles: List[str]
    members: List[ClusterMember]

@strawberry.type
class ClusterResponse:
    project_id: int
    model: str
    algo: Optional[str]
    k: Optional[int]
    min_cluster_size: Optional[int]
    use_umap: bool
    clusters: List[Cluster]
    message: Optional[str] = None

@strawberry.type
class DuplicateSimilar:
    tareaId: int
    titulo: str
    similitud: float
    similitudPct: int

@strawberry.type
class DuplicateSimplified:
    tareaId: int
    sim: float

@strawberry.type
class DuplicatePair:
    tareaId: int
    titulo: str
    texto: str
    conteo: int
    similares: List[DuplicateSimilar]
    posibles_duplicados: List[DuplicateSimplified]

@strawberry.type
class DuplicateResponse:
    project_id: int
    model: str
    threshold: float
    top_n: int
    duplicates: List[DuplicatePair]
    message: Optional[str] = None
