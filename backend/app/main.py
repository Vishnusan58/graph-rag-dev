"""
FastAPI Backend for GraphRAG Assistant

This module implements the FastAPI backend for the GraphRAG Assistant.
It provides API endpoints for:
1. Querying the GraphRAG system
2. Getting information about specific concepts, functions, modules
3. Comparing programming entities
"""

import os
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from .graph_builder import Neo4jGraphBuilder
from .langchain_chain import GraphRAGChain

# Load environment variables
load_dotenv()

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    language: str = "Rust"

class ConceptRequest(BaseModel):
    name: str
    language: str = "Rust"

class FunctionRequest(BaseModel):
    name: str
    module: Optional[str] = None
    language: str = "Rust"

class ModuleRequest(BaseModel):
    name: str
    language: str = "Rust"

class CompareRequest(BaseModel):
    entity1: str
    entity2: str
    language: str = "Rust"

class DocumentationRequest(BaseModel):
    docs_data: List[Dict[str, Any]]

class GraphResponse(BaseModel):
    answer: str
    context: Optional[str] = None
    graph_data: Optional[Dict[str, Any]] = None

# Create a lifespan context manager for the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the GraphRAG chain on startup
    app.state.graph_rag_chain = GraphRAGChain()
    
    yield
    
    # Close connections on shutdown
    app.state.graph_rag_chain.close()

# Initialize FastAPI app
app = FastAPI(
    title="GraphRAG Assistant API",
    description="API for the GraphRAG Assistant, providing programming language assistance with graph-based retrieval.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the GraphRAG chain
def get_graph_rag_chain():
    return app.state.graph_rag_chain

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the GraphRAG Assistant API"}

@app.post("/query", response_model=GraphResponse)
async def query(request: QueryRequest, graph_rag_chain: GraphRAGChain = Depends(get_graph_rag_chain)):
    """
    Process a general query through the GraphRAG pipeline.
    """
    try:
        result = graph_rag_chain.query(request.query, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/concept", response_model=GraphResponse)
async def get_concept(request: ConceptRequest, graph_rag_chain: GraphRAGChain = Depends(get_graph_rag_chain)):
    """
    Get information about a specific programming concept.
    """
    try:
        result = graph_rag_chain.query_concept(request.name, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving concept: {str(e)}")

@app.post("/function", response_model=GraphResponse)
async def get_function(request: FunctionRequest, graph_rag_chain: GraphRAGChain = Depends(get_graph_rag_chain)):
    """
    Get information about a specific function.
    """
    try:
        result = graph_rag_chain.query_function(request.name, request.module, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving function: {str(e)}")

@app.post("/module", response_model=GraphResponse)
async def get_module(request: ModuleRequest, graph_rag_chain: GraphRAGChain = Depends(get_graph_rag_chain)):
    """
    Get information about a specific module.
    """
    try:
        result = graph_rag_chain.query_module(request.name, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving module: {str(e)}")

@app.post("/compare", response_model=GraphResponse)
async def compare_entities(request: CompareRequest, graph_rag_chain: GraphRAGChain = Depends(get_graph_rag_chain)):
    """
    Compare two programming entities (functions, concepts, etc.).
    """
    try:
        result = graph_rag_chain.compare_entities(request.entity1, request.entity2, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing entities: {str(e)}")

# Admin endpoints for data ingestion
@app.post("/admin/ingest", status_code=201)
async def ingest_documentation(
    request: DocumentationRequest, 
    background_tasks: BackgroundTasks
):
    """
    Ingest documentation data into the knowledge graph.
    This is an admin endpoint that should be secured in production.
    """
    try:
        # Run ingestion in the background to avoid blocking the API
        background_tasks.add_task(ingest_documentation_task, request.docs_data)
        return {"message": "Documentation ingestion started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting ingestion: {str(e)}")

async def ingest_documentation_task(docs_data: List[Dict[str, Any]]):
    """Background task for ingesting documentation."""
    try:
        graph_builder = Neo4jGraphBuilder()
        graph_builder.connect()
        graph_builder.create_constraints()
        graph_builder.ingest_documentation(docs_data)
        graph_builder.close()
        print(f"Successfully ingested {len(docs_data)} documentation items")
    except Exception as e:
        print(f"Error during documentation ingestion: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
