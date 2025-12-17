from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from src.graphql.schema import schema

app = FastAPI(title="ML Topics & Clustering GraphQL Service", version="2.0.0")

graphql_app = GraphQLRouter(schema)

# Mount GraphQL route
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
def health():
    """
    Health check logic indicating service is alive.
    For Kubernetes livenessProbe.
    """
    return {"status": "ok", "service": "ml_noSupervisado"}

if __name__ == "__main__":
    import uvicorn
    # En desarrollo local
    uvicorn.run(app, host="0.0.0.0", port=8000)
