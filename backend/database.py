from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable
from typing import Optional, List, Dict, Any
from datetime import datetime
import time
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

MAX_RETRIES = 30
RETRY_INTERVAL = 2  # seconds

class DatabaseManager:
    _instance = None
    _driver: Optional[Driver] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize only if not already initialized
        if not self._driver:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    self._driver = GraphDatabase.driver(
                        NEO4J_URI,
                        auth=(NEO4J_USER, NEO4J_PASSWORD)
                    )
                    # Test the connection
                    self._driver.verify_connectivity()
                    print("Successfully connected to Neo4j database")
                    break
                except ServiceUnavailable as e:
                    retries += 1
                    if retries == MAX_RETRIES:
                        raise Exception(f"Failed to connect to Neo4j after {MAX_RETRIES} attempts: {str(e)}")
                    print(f"Neo4j not ready (attempt {retries}/{MAX_RETRIES}), retrying in {RETRY_INTERVAL} seconds...")
                    time.sleep(RETRY_INTERVAL)

    def close(self):
        """Close the database connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def verify_connectivity(self) -> bool:
        """Verify that the connection to Neo4j is working."""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def add_article(self, title: str, content: str, metadata: Dict = None) -> Dict[str, Any]:
        """
        Add an article node to the database.
        
        Args:
            title: The article title
            content: The article content
            metadata: Optional metadata dictionary
        
        Returns:
            Dictionary containing the created article's properties
        """
        if metadata is None:
            metadata = {}
        
        query = """
        MERGE (a:Article {title: $title})
        SET a.content = $content,
            a.metadata = $metadata,
            a.timestamp = datetime()
        RETURN a
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, title=title, content=content, metadata=metadata)
                record = result.single()
                return dict(record["a"])
        except Exception as e:
            print(f"Error adding article: {e}")
            raise

    def add_author(self, name: str) -> Dict[str, Any]:
        """
        Add an author node to the database.
        
        Args:
            name: The author's name
        
        Returns:
            Dictionary containing the created author's properties
        """
        query = """
        MERGE (a:Author {name: $name})
        RETURN a
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, name=name)
                record = result.single()
                return dict(record["a"])
        except Exception as e:
            print(f"Error adding author: {e}")
            raise

    def add_concept(self, name: str, description: str = "") -> Dict[str, Any]:
        """
        Add a concept node to the database.
        
        Args:
            name: The concept name
            description: Optional description of the concept
        
        Returns:
            Dictionary containing the created concept's properties
        """
        query = """
        MERGE (c:Concept {name: $name})
        SET c.description = $description
        RETURN c
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, name=name, description=description)
                record = result.single()
                return dict(record["c"])
        except Exception as e:
            print(f"Error adding concept: {e}")
            raise

    def link_article_to_author(self, article_title: str, author_name: str) -> bool:
        """
        Create a relationship between an article and its author.
        
        Args:
            article_title: The title of the article
            author_name: The name of the author
        
        Returns:
            True if the relationship was created successfully
        """
        query = """
        MATCH (a:Article {title: $article_title})
        MATCH (auth:Author {name: $author_name})
        MERGE (auth)-[:WROTE]->(a)
        RETURN auth, a
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, article_title=article_title, author_name=author_name)
                return result.single() is not None
        except Exception as e:
            print(f"Error linking article to author: {e}")
            raise

    def link_article_to_concepts(self, article_title: str, concept_names: List[str]) -> bool:
        """
        Create relationships between an article and multiple concepts.
        
        Args:
            article_title: The title of the article
            concept_names: List of concept names to link to the article
        
        Returns:
            True if all relationships were created successfully
        """
        query = """
        MATCH (a:Article {title: $article_title})
        UNWIND $concept_names as concept_name
        MERGE (c:Concept {name: concept_name})
        MERGE (a)-[:DISCUSSES]->(c)
        RETURN count(c) as concept_count
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, article_title=article_title, concept_names=concept_names)
                record = result.single()
                return record and record["concept_count"] == len(concept_names)
        except Exception as e:
            print(f"Error linking article to concepts: {e}")
            raise

    def link_related_concepts(self, concept1_name: str, concept2_name: str, 
                            relationship_type: str = "RELATED_TO",
                            properties: Dict[str, Any] = None) -> bool:
        """
        Create a relationship between two concepts with optional properties.
        
        Args:
            concept1_name: The name of the first concept
            concept2_name: The name of the second concept
            relationship_type: The type of relationship between concepts
            properties: Optional dictionary of relationship properties
        
        Returns:
            True if the relationship was created successfully
        """
        # Default properties if none provided
        if properties is None:
            properties = {
                "weight": 1.0,
                "timestamp": datetime.now().isoformat()
            }
        
        query = f"""
        MATCH (c1:Concept {{name: $concept1_name}})
        MATCH (c2:Concept {{name: $concept2_name}})
        MERGE (c1)-[r:{relationship_type}]->(c2)
        SET r += $properties
        RETURN c1, r, c2
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query, 
                    concept1_name=concept1_name,
                    concept2_name=concept2_name,
                    properties=properties
                )
                return result.single() is not None
        except Exception as e:
            print(f"Error linking related concepts: {e}")
            raise

    def update_edge_weight(self, source_id: int, target_id: int, weight: float) -> bool:
        """
        Update the weight of an edge between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            weight: New weight value
            
        Returns:
            True if the weight was updated successfully
        """
        query = """
        MATCH (source)-[r]->(target)
        WHERE id(source) = $source_id AND id(target) = $target_id
        SET r.weight = $weight
        RETURN r
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id,
                    weight=weight
                )
                return result.single() is not None
        except Exception as e:
            print(f"Error updating edge weight: {e}")
            raise

    def get_node_properties(self, node_id: int) -> Dict[str, Any]:
        """
        Get all properties of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary of node properties
        """
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        RETURN properties(n) as props
        """
        try:
            with self._driver.session() as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                return dict(record["props"]) if record else {}
        except Exception as e:
            print(f"Error getting node properties: {e}")
            raise

    def get_edge_properties(self, source_id: int, target_id: int) -> Dict[str, Any]:
        """
        Get all properties of an edge between two nodes.
        
        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            
        Returns:
            Dictionary of edge properties
        """
        query = """
        MATCH (source)-[r]->(target)
        WHERE id(source) = $source_id AND id(target) = $target_id
        RETURN properties(r) as props
        """
        try:
            with self._driver.session() as session:
                result = session.run(
                    query,
                    source_id=source_id,
                    target_id=target_id
                )
                record = result.single()
                return dict(record["props"]) if record else {}
        except Exception as e:
            print(f"Error getting edge properties: {e}")
            raise
