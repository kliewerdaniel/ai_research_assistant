from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Create Neo4j constraints and indexes on startup
STARTUP_CYPHER = """
CREATE CONSTRAINT article_title IF NOT EXISTS
FOR (a:Article) REQUIRE a.title IS UNIQUE
"""
