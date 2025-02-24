import networkx as nx
import torch
import numpy as np
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import random
import torch.nn as nn
import torch.nn.functional as F
import community as community_louvain

class GraphProcessor:
    def __init__(self, db_manager):
        self.db = db_manager
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.language_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
    def build_graph_from_neo4j(self):
        """Convert Neo4j graph to NetworkX for advanced processing"""
        with self.db._driver.session() as session:
            # Get all nodes
            nodes = session.run("""
                MATCH (n)
                RETURN n, labels(n) as labels, id(n) as id
            """)
            for record in nodes:
                node = record["n"]
                node_id = record["id"]
                self.graph.add_node(node_id, **dict(node), labels=record["labels"])
            
            # Get all relationships
            relationships = session.run("""
                MATCH ()-[r]->()
                RETURN id(startNode(r)) as source, id(endNode(r)) as target,
                       type(r) as type, properties(r) as properties
            """)
            for record in relationships:
                self.graph.add_edge(
                    record["source"],
                    record["target"],
                    type=record["type"],
                    **record["properties"]
                )
        return self.graph

    def calculate_edge_weights(self) -> None:
        """Calculate edge weights based on multiple factors"""
        for u, v, data in self.graph.edges(data=True):
            # Base weight
            weight = 1.0
            
            # Temporal proximity weight
            if 'timestamp' in self.graph.nodes[u] and 'timestamp' in self.graph.nodes[v]:
                time_diff = abs(self.graph.nodes[u]['timestamp'] - self.graph.nodes[v]['timestamp'])
                temporal_weight = 1.0 / (1.0 + np.log1p(time_diff))
                weight *= temporal_weight
            
            # Semantic similarity weight
            if 'content' in self.graph.nodes[u] and 'content' in self.graph.nodes[v]:
                similarity = self._calculate_semantic_similarity(
                    self.graph.nodes[u]['content'],
                    self.graph.nodes[v]['content']
                )
                weight *= (1.0 + similarity)
            
            # Co-occurrence frequency weight
            if 'frequency' in data:
                weight *= (1.0 + np.log1p(data['frequency']))
            
            self.graph[u][v]['weight'] = weight

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using language model"""
        with torch.no_grad():
            inputs1 = self.tokenizer(text1, return_tensors='pt', truncation=True, max_length=512)
            inputs2 = self.tokenizer(text2, return_tensors='pt', truncation=True, max_length=512)
            
            embeddings1 = self.language_model(**inputs1).last_hidden_state.mean(dim=1)
            embeddings2 = self.language_model(**inputs2).last_hidden_state.mean(dim=1)
            
            similarity = torch.cosine_similarity(embeddings1, embeddings2).item()
            return similarity

    def find_shortest_path(self, source: int, target: int, 
                          meta_path: Optional[List[str]] = None) -> List[int]:
        """Find shortest path considering edge weights and optional meta-path constraints"""
        if meta_path:
            return self._meta_path_shortest_path(source, target, meta_path)
        return nx.shortest_path(self.graph, source, target, weight='weight')

    def _meta_path_shortest_path(self, source: int, target: int, 
                                meta_path: List[str]) -> List[int]:
        """Find shortest path that follows a specific meta-path pattern"""
        def is_valid_path(path: List[int]) -> bool:
            if len(path) - 1 != len(meta_path):
                return False
            for i in range(len(path) - 1):
                edge_type = self.graph[path[i]][path[i+1]]['type']
                if edge_type != meta_path[i]:
                    return False
            return True
        
        paths = nx.all_simple_paths(self.graph, source, target)
        valid_paths = [p for p in paths if is_valid_path(p)]
        
        if not valid_paths:
            return []
        
        # Return the shortest valid path
        return min(valid_paths, key=lambda p: sum(
            self.graph[p[i]][p[i+1]]['weight'] 
            for i in range(len(p)-1)
        ))

    def calculate_centrality_measures(self) -> Dict[str, Dict[int, float]]:
        """Calculate various centrality measures for nodes"""
        centrality_measures = {
            'pagerank': nx.pagerank(self.graph, weight='weight'),
            'betweenness': nx.betweenness_centrality(self.graph, weight='weight'),
            'eigenvector': nx.eigenvector_centrality(self.graph, weight='weight'),
            'closeness': nx.closeness_centrality(self.graph, distance='weight')
        }
        return centrality_measures

    def detect_communities(self) -> Dict[int, int]:
        """Detect communities using Louvain algorithm"""
        # Convert directed graph to undirected for community detection
        undirected_graph = self.graph.to_undirected()
        return community_louvain.best_partition(undirected_graph)

    def _random_walk(self, start_node: int, walk_length: int) -> List[int]:
        """Generate a random walk starting from start_node"""
        walk = [start_node]
        for _ in range(walk_length - 1):
            cur = walk[-1]
            neighbors = list(self.graph.neighbors(cur))
            if not neighbors:
                break
            # Use edge weights for transition probabilities
            weights = [self.graph[cur][nbr].get('weight', 1.0) for nbr in neighbors]
            total = sum(weights)
            weights = [w/total for w in weights]
            next_node = random.choices(neighbors, weights=weights)[0]
            walk.append(next_node)
        return walk

    def generate_node_embeddings(self, dimensions: int = 128) -> Dict[int, np.ndarray]:
        """Generate node embeddings using random walks and skip-gram"""
        num_walks = 10
        walk_length = 80
        
        # Generate random walks
        walks = []
        for node in self.graph.nodes():
            for _ in range(num_walks):
                walks.append(self._random_walk(node, walk_length))
        
        # Create node to index mapping
        node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        vocab_size = len(node_to_idx)
        
        # Convert walks to tensor format
        walk_tensors = []
        for walk in walks:
            indices = [node_to_idx[node] for node in walk]
            walk_tensors.extend(
                (indices[i], indices[i+1])
                for i in range(len(indices)-1)
            )
        
        # Create skip-gram model
        class SkipGramModel(nn.Module):
            def __init__(self, vocab_size: int, embedding_dim: int):
                super().__init__()
                self.embeddings = nn.Embedding(vocab_size, embedding_dim)
                self.linear = nn.Linear(embedding_dim, vocab_size)
                
            def forward(self, inputs):
                embeds = self.embeddings(inputs)
                out = self.linear(embeds)
                return F.log_softmax(out, dim=1)
        
        model = SkipGramModel(vocab_size, dimensions)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Train the model
        model.train()
        batch_size = 32
        for epoch in range(5):
            for i in range(0, len(walk_tensors), batch_size):
                batch = walk_tensors[i:i+batch_size]
                if not batch:
                    continue
                inputs = torch.tensor([p[0] for p in batch])
                targets = torch.tensor([p[1] for p in batch])
                
                optimizer.zero_grad()
                log_probs = model(inputs)
                loss = F.nll_loss(log_probs, targets)
                loss.backward()
                optimizer.step()
        
        # Extract embeddings
        with torch.no_grad():
            embeddings = model.embeddings.weight.numpy()
            self.node_embeddings = {
                idx_to_node[i]: embeddings[i]
                for i in range(vocab_size)
            }
        
        return self.node_embeddings

    def hybrid_search(self, query_embedding: np.ndarray, 
                     personalization: Optional[Dict[int, float]] = None,
                     alpha: float = 0.5) -> List[Tuple[int, float]]:
        """Combine vector similarity with personalized PageRank for search"""
        if not self.node_embeddings:
            self.generate_node_embeddings()
        
        # Calculate vector similarities
        similarities = {
            node: np.dot(emb, query_embedding) / (
                np.linalg.norm(emb) * np.linalg.norm(query_embedding)
            )
            for node, emb in self.node_embeddings.items()
        }
        
        # Calculate personalized PageRank
        pagerank = nx.pagerank(
            self.graph,
            personalization=personalization,
            weight='weight'
        )
        
        # Combine scores
        hybrid_scores = {
            node: alpha * similarities[node] + (1 - alpha) * pagerank[node]
            for node in self.graph.nodes()
        }
        
        # Return sorted results
        return sorted(
            hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

    def setup_gnn(self, num_node_features: int, num_classes: int):
        """Initialize Graph Neural Network with attention mechanism"""
        self.gnn = GNNModel(num_node_features, num_classes)
        
    def prepare_gnn_data(self) -> Data:
        """Prepare graph data for GNN processing"""
        # Convert NetworkX graph to PyTorch Geometric Data
        edge_index = torch.tensor(list(self.graph.edges())).t().contiguous()
        
        # Prepare node features (using node embeddings if available)
        if not self.node_embeddings:
            self.generate_node_embeddings()
        
        x = torch.tensor([
            self.node_embeddings[node] 
            for node in sorted(self.graph.nodes())
        ], dtype=torch.float)
        
        # Prepare edge weights
        edge_weights = torch.tensor([
            self.graph[u][v]['weight']
            for u, v in self.graph.edges()
        ], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weights)

class GNNModel(torch.nn.Module):
    """Graph Neural Network with attention mechanism"""
    def __init__(self, num_node_features: int, num_classes: int):
        super().__init__()
        self.conv1 = GATv2Conv(num_node_features, 64, heads=4)
        self.conv2 = GATv2Conv(64 * 4, num_classes)
        
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        
        # First GAT layer with multi-head attention
        x = torch.relu(self.conv1(x, edge_index, edge_weight))
        
        # Second GAT layer combining attention heads
        x = self.conv2(x, edge_index, edge_weight)
        
        return torch.log_softmax(x, dim=1)
