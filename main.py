from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from starlette.middleware.cors import CORSMiddleware  # ✅
from src.graphql.schema import schema

app = FastAPI(title="ML Topics & Clustering GraphQL Service", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_methods=["*"],      
    allow_headers=["*"],    
    allow_credentials=False,  
)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/health")
def health():
    return {"status": "ok", "service": "ml_noSupervisado"}
