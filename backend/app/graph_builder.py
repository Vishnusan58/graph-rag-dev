"""
Graph Builder for GraphRAG Assistant

This module handles the construction of the knowledge graph in Neo4j.
It includes functions for:
1. Connecting to Neo4j
2. Creating graph schema
3. Ingesting documentation data
4. Building relationships between concepts, modules, functions, etc.
"""

import os
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Neo4jGraphBuilder:
    """
    Class for building and managing the knowledge graph in Neo4j.
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (defaults to environment variable)
            username: Neo4j username (defaults to environment variable)
            password: Neo4j password (defaults to environment variable)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            print(f"Connected to Neo4j database at {self.uri}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")
    
    def create_constraints(self) -> None:
        """Create necessary constraints and indexes in Neo4j."""
        constraints = [
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT function_name IF NOT EXISTS FOR (f:Function) REQUIRE (f.name, f.module) IS UNIQUE",
            "CREATE CONSTRAINT class_name IF NOT EXISTS FOR (c:Class) REQUIRE (c.name, c.module) IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Error creating constraint: {e}")
    
    def add_concept(self, name: str, description: str, language: str, 
                    related_concepts: Optional[List[str]] = None) -> None:
        """
        Add a programming concept to the graph.
        
        Args:
            name: Name of the concept
            description: Description of the concept
            language: Programming language this concept belongs to
            related_concepts: List of related concept names
        """
        query = """
        MERGE (c:Concept {name: $name})
        SET c.description = $description,
            c.language = $language
        RETURN c
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, description=description, language=language)
            
            # Create relationships to related concepts
            if related_concepts:
                for related in related_concepts:
                    rel_query = """
                    MATCH (c1:Concept {name: $name})
                    MATCH (c2:Concept {name: $related})
                    MERGE (c1)-[:RELATED_TO]->(c2)
                    """
                    session.run(rel_query, name=name, related=related)
    
    def add_module(self, name: str, description: str, language: str) -> None:
        """
        Add a module to the graph.
        
        Args:
            name: Name of the module
            description: Description of the module
            language: Programming language this module belongs to
        """
        query = """
        MERGE (m:Module {name: $name})
        SET m.description = $description,
            m.language = $language
        RETURN m
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, description=description, language=language)
    
    def add_function(self, name: str, module: str, description: str, 
                     parameters: Optional[List[Dict[str, str]]] = None,
                     return_type: Optional[str] = None,
                     examples: Optional[List[str]] = None,
                     related_concepts: Optional[List[str]] = None) -> None:
        """
        Add a function to the graph.
        
        Args:
            name: Name of the function
            module: Module the function belongs to
            description: Description of the function
            parameters: List of parameter dictionaries with name, type, and description
            return_type: Return type of the function
            examples: List of code examples
            related_concepts: List of related concept names
        """
        query = """
        MATCH (m:Module {name: $module})
        MERGE (f:Function {name: $name, module: $module})
        SET f.description = $description,
            f.parameters = $parameters,
            f.return_type = $return_type,
            f.examples = $examples
        MERGE (f)-[:BELONGS_TO]->(m)
        RETURN f
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, module=module, description=description,
                       parameters=parameters, return_type=return_type, examples=examples)
            
            # Create relationships to related concepts
            if related_concepts:
                for concept in related_concepts:
                    rel_query = """
                    MATCH (f:Function {name: $name, module: $module})
                    MATCH (c:Concept {name: $concept})
                    MERGE (f)-[:IMPLEMENTS]->(c)
                    """
                    session.run(rel_query, name=name, module=module, concept=concept)
    
    def add_class(self, name: str, module: str, description: str,
                 methods: Optional[List[Dict[str, Any]]] = None,
                 attributes: Optional[List[Dict[str, str]]] = None,
                 examples: Optional[List[str]] = None,
                 related_concepts: Optional[List[str]] = None) -> None:
        """
        Add a class to the graph.
        
        Args:
            name: Name of the class
            module: Module the class belongs to
            description: Description of the class
            methods: List of method dictionaries
            attributes: List of attribute dictionaries
            examples: List of code examples
            related_concepts: List of related concept names
        """
        query = """
        MATCH (m:Module {name: $module})
        MERGE (c:Class {name: $name, module: $module})
        SET c.description = $description,
            c.methods = $methods,
            c.attributes = $attributes,
            c.examples = $examples
        MERGE (c)-[:BELONGS_TO]->(m)
        RETURN c
        """
        
        with self.driver.session() as session:
            session.run(query, name=name, module=module, description=description,
                       methods=methods, attributes=attributes, examples=examples)
            
            # Create relationships to related concepts
            if related_concepts:
                for concept in related_concepts:
                    rel_query = """
                    MATCH (c:Class {name: $name, module: $module})
                    MATCH (con:Concept {name: $concept})
                    MERGE (c)-[:IMPLEMENTS]->(con)
                    """
                    session.run(rel_query, name=name, module=module, concept=concept)
    
    def add_relationship(self, from_type: str, from_name: str, from_module: Optional[str],
                        to_type: str, to_name: str, to_module: Optional[str],
                        relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a relationship between two nodes.
        
        Args:
            from_type: Type of the source node (Concept, Module, Function, Class)
            from_name: Name of the source node
            from_module: Module of the source node (if applicable)
            to_type: Type of the target node
            to_name: Name of the target node
            to_module: Module of the target node (if applicable)
            relationship_type: Type of relationship (e.g., CALLS, INHERITS, IMPLEMENTS)
            properties: Optional properties for the relationship
        """
        # Build match clauses based on node types
        from_match = self._build_match_clause(from_type, from_name, from_module)
        to_match = self._build_match_clause(to_type, to_name, to_module)
        
        # Build the relationship query
        rel_query = f"""
        {from_match}
        {to_match}
        MERGE (from)-[r:{relationship_type}]->(to)
        """
        
        # Add properties if provided
        if properties:
            props_str = ", ".join([f"r.{k} = ${k}" for k in properties.keys()])
            rel_query += f"SET {props_str}"
        
        rel_query += "RETURN r"
        
        # Execute the query
        with self.driver.session() as session:
            params = {
                "from_name": from_name,
                "to_name": to_name,
                **({"from_module": from_module} if from_module else {}),
                **({"to_module": to_module} if to_module else {}),
                **(properties or {})
            }
            session.run(rel_query, **params)
    
    def _build_match_clause(self, node_type: str, name: str, module: Optional[str]) -> str:
        """Helper method to build match clauses for different node types."""
        if node_type in ["Function", "Class"] and module:
            return f"MATCH (from:{node_type} {{name: $from_name, module: $from_module}})"
        else:
            return f"MATCH (from:{node_type} {{name: $from_name}})"
    
    def ingest_documentation(self, docs_data: List[Dict[str, Any]]) -> None:
        """
        Ingest documentation data into the graph.
        
        Args:
            docs_data: List of documentation data dictionaries
        """
        for doc in docs_data:
            doc_type = doc.get("type")
            
            if doc_type == "concept":
                self.add_concept(
                    name=doc["name"],
                    description=doc["description"],
                    language=doc["language"],
                    related_concepts=doc.get("related_concepts")
                )
            elif doc_type == "module":
                self.add_module(
                    name=doc["name"],
                    description=doc["description"],
                    language=doc["language"]
                )
            elif doc_type == "function":
                self.add_function(
                    name=doc["name"],
                    module=doc["module"],
                    description=doc["description"],
                    parameters=doc.get("parameters"),
                    return_type=doc.get("return_type"),
                    examples=doc.get("examples"),
                    related_concepts=doc.get("related_concepts")
                )
            elif doc_type == "class":
                self.add_class(
                    name=doc["name"],
                    module=doc["module"],
                    description=doc["description"],
                    methods=doc.get("methods"),
                    attributes=doc.get("attributes"),
                    examples=doc.get("examples"),
                    related_concepts=doc.get("related_concepts")
                )
            
            # Process relationships if present
            if "relationships" in doc:
                for rel in doc["relationships"]:
                    self.add_relationship(
                        from_type=doc_type.capitalize(),
                        from_name=doc["name"],
                        from_module=doc.get("module"),
                        to_type=rel["to_type"].capitalize(),
                        to_name=rel["to_name"],
                        to_module=rel.get("to_module"),
                        relationship_type=rel["type"].upper(),
                        properties=rel.get("properties")
                    )

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    sample_docs = [
        {
            "type": "concept",
            "name": "Ownership",
            "description": "Rust's central feature is ownership, which manages memory safety without garbage collection.",
            "language": "Rust",
            "related_concepts": ["Borrowing", "Lifetimes"]
        },
        {
            "type": "module",
            "name": "std::collections",
            "description": "Collection types in the Rust standard library.",
            "language": "Rust"
        },
        {
            "type": "function",
            "name": "vec",
            "module": "std::vec",
            "description": "Creates a new vector with the given elements.",
            "parameters": [
                {"name": "elements", "type": "T", "description": "Elements to include in the vector"}
            ],
            "return_type": "Vec<T>",
            "examples": ["let v = vec![1, 2, 3];"],
            "related_concepts": ["Ownership"],
            "relationships": [
                {
                    "to_type": "module",
                    "to_name": "std::collections",
                    "type": "related_to"
                }
            ]
        }
    ]
    
    # Initialize and test the graph builder
    graph_builder = Neo4jGraphBuilder()
    graph_builder.connect()
    graph_builder.create_constraints()
    graph_builder.ingest_documentation(sample_docs)
    graph_builder.close()
