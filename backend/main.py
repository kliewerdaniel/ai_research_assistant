from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List, Union
from neo4j import GraphDatabase
from datetime import datetime
import numpy as np
import config
from graph_processing import GraphProcessor

app = FastAPI(title="Knowledge Graph API")

class Article(BaseModel):
    title: str
    content: str
    metadata: Optional[Dict] = {}

class PathRequest(BaseModel):
    source_id: int
    target_id: int
    meta_path: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    personalization: Optional[Dict[int, float]] = None
    alpha: Optional[float] = 0.5

class CentralityRequest(BaseModel):
    node_ids: Optional[List[int]] = None
    measures: Optional[List[str]] = ['pagerank', 'betweenness']

# Initialize database and graph processor
db = DatabaseManager()
graph_processor = GraphProcessor(db)

@app.post("/add_article")
async def add_article(article: Article):
    try:
        result = db.add_article(
            title=article.title,
            content=article.content,
            metadata=article.metadata
        )
        # Rebuild graph after adding new article
        graph_processor.build_graph_from_neo4j()
        graph_processor.calculate_edge_weights()
        return {"message": "Article added successfully", "title": article.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/find_path")
async def find_path(request: PathRequest):
    """Find shortest path between two nodes with optional meta-path constraints"""
    try:
        path = graph_processor.find_shortest_path(
            request.source_id,
            request.target_id,
            request.meta_path
        )
        if not path:
            raise HTTPException(status_code=404, detail="No valid path found")
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/calculate_centrality")
async def calculate_centrality(request: CentralityRequest):
    """Calculate centrality measures for specified nodes"""
    try:
        measures = graph_processor.calculate_centrality_measures()
        
        # Filter by requested measures and nodes
        result = {}
        for measure_name, scores in measures.items():
            if measure_name in request.measures:
                if request.node_ids:
                    result[measure_name] = {
                        node_id: score 
                        for node_id, score in scores.items()
                        if node_id in request.node_ids
                    }
                else:
                    result[measure_name] = scores
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detect_communities")
async def detect_communities():
    """Detect communities in the graph"""
    try:
        communities = graph_processor.detect_communities()
        return {"communities": communities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hybrid_search")
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search combining vector similarity and graph structure"""
    try:
        # Generate query embedding using the language model
        with torch.no_grad():
            inputs = graph_processor.tokenizer(
                request.query,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            query_embedding = graph_processor.language_model(**inputs).last_hidden_state.mean(dim=1)
            query_embedding = query_embedding.numpy().flatten()
        
        results = graph_processor.hybrid_search(
            query_embedding,
            request.personalization,
            request.alpha
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize graph on startup"""
    graph_processor.build_graph_from_neo4j()
    graph_processor.calculate_edge_weights()
    graph_processor.generate_node_embeddings()

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
