# AI Research Assistant

A powerful AI research assistant that combines a Chrome extension frontend with a FastAPI backend to help users analyze and explore research content through an intelligent knowledge graph system.

## Features

- **Knowledge Graph Integration**: Store and analyze relationships between research articles
- **Intelligent Path Finding**: Discover connections between different research topics
- **Community Detection**: Identify clusters of related research
- **Hybrid Search**: Combine vector similarity and graph structure for enhanced search results
- **Centrality Analysis**: Calculate importance measures for research topics
- **Chrome Extension Interface**: Seamless browser integration for easy research assistance

## Architecture

### Frontend (Chrome Extension)
- Browser-based user interface for direct research assistance
- Content scripts for webpage analysis and interaction
- Popup interface for quick access to features
- Permissions for active tab interaction and data storage

### Backend (FastAPI)
- RESTful API service built with FastAPI
- Neo4j graph database integration
- Advanced graph processing capabilities
- Machine learning model integration
- Hybrid search combining vector similarity and graph structure

### Models
- Graph embedding generation
- Natural language processing
- Community detection algorithms
- Centrality measure calculations

## Installation

### Backend Setup

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Configure environment variables:
Create a `.env` file in the backend directory with:
```
NEO4J_URI=your_neo4j_uri
NEO4J_USER=your_username
NEO4J_PASSWORD=your_password
```

3. Start the backend server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Load the extension in Chrome:
- Open Chrome and navigate to `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked" and select the `frontend` directory

## API Endpoints

### POST /add_article
Add a new article to the knowledge graph
```json
{
    "title": "string",
    "content": "string",
    "metadata": {}
}
```

### POST /find_path
Find shortest path between two nodes
```json
{
    "source_id": "integer",
    "target_id": "integer",
    "meta_path": ["string"]
}
```

### POST /calculate_centrality
Calculate centrality measures for nodes
```json
{
    "node_ids": ["integer"],
    "measures": ["pagerank", "betweenness"]
}
```

### GET /detect_communities
Detect communities in the knowledge graph

### POST /hybrid_search
Perform hybrid search combining vector similarity and graph structure
```json
{
    "query": "string",
    "personalization": {},
    "alpha": 0.5
}
```

## Dependencies

### Backend
- fastapi==0.109.1
- uvicorn==0.27.1
- neo4j==5.17.0
- python-dotenv==1.0.1
- pydantic==2.6.1
- networkx==3.2.1
- torch==2.2.0
- torch-geometric==2.5.0
- transformers==4.37.2
- scikit-learn==1.4.0
- numpy==1.26.3
- pandas==2.2.0
- community==1.0.0b1
- node2vec==0.4.6

### Frontend
- Chrome Extension Manifest V3
- Standard Web APIs (activeTab, storage, contextMenus, webRequest)

## Development

### Backend Development
1. Start Neo4j database
2. Configure environment variables
3. Run the development server:
```bash
uvicorn main:app --reload
```

### Frontend Development
1. Make changes to extension files
2. Reload the extension in Chrome to test changes
3. Use Chrome DevTools for debugging

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
