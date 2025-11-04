from fastapi import FastAPI
from .api.routers import papers, queries

app = FastAPI()
app.include_router(papers.router)
app.include_router(queries.router)