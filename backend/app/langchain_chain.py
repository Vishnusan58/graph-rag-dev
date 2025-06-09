"""
LangChain Integration for GraphRAG Assistant

This module implements the GraphRAG retrieval pipeline using LangChain.
It includes:
1. Graph-based retrieval from Neo4j
2. LLM integration for generating responses
3. Chain components for the complete GraphRAG pipeline
"""

import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain_core import load
from langchain_community.llms import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.memory import ConversationBufferMemory
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

class GraphRetriever:
    """
    Class for retrieving information from the Neo4j knowledge graph.
    """
    
    def __init__(self, uri: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize the Neo4j connection for retrieval.
        
        Args:
            uri: Neo4j connection URI (defaults to environment variable)
            username: Neo4j username (defaults to environment variable)
            password: Neo4j password (defaults to environment variable)
            database: Neo4j database name (defaults to environment variable)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.driver = None
        self.embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

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
    
    def search_by_keyword(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for nodes containing the keyword.
        
        Args:
            keyword: Keyword to search for
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing node information
        """
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword OR n.description CONTAINS $keyword
        RETURN n, labels(n) as types
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, keyword=keyword, limit=limit)
            return [self._node_to_dict(record["n"], record["types"]) for record in result]
    
    def get_concept_info(self, concept_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a concept.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Dictionary containing concept information and related nodes
        """
        query = """
        MATCH (c:Concept {name: $name})
        OPTIONAL MATCH (c)-[:RELATED_TO]->(rc:Concept)
        OPTIONAL MATCH (f:Function)-[:IMPLEMENTS]->(c)
        OPTIONAL MATCH (cl:Class)-[:IMPLEMENTS]->(c)
        RETURN c, 
               collect(distinct rc) as related_concepts,
               collect(distinct f) as functions,
               collect(distinct cl) as classes
        """
        
        with self.driver.session() as session:
            result = session.run(query, name=concept_name).single()
            if not result:
                return {}
            
            concept = self._node_to_dict(result["c"], ["Concept"])
            concept["related_concepts"] = [self._node_to_dict(rc, ["Concept"]) for rc in result["related_concepts"] if rc]
            concept["functions"] = [self._node_to_dict(f, ["Function"]) for f in result["functions"] if f]
            concept["classes"] = [self._node_to_dict(cl, ["Class"]) for cl in result["classes"] if cl]
            
            return concept
    
    def get_function_info(self, function_name: str, module: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a function.
        
        Args:
            function_name: Name of the function
            module: Module the function belongs to (optional)
            
        Returns:
            Dictionary containing function information and related nodes
        """
        if module:
            query = """
            MATCH (f:Function {name: $name, module: $module})
            OPTIONAL MATCH (f)-[:BELONGS_TO]->(m:Module)
            OPTIONAL MATCH (f)-[:IMPLEMENTS]->(c:Concept)
            OPTIONAL MATCH (f)-[r]->(other)
            RETURN f, m, collect(distinct c) as concepts, 
                   collect(distinct {type: type(r), node: other}) as relationships
            """
            params = {"name": function_name, "module": module}
        else:
            query = """
            MATCH (f:Function {name: $name})
            OPTIONAL MATCH (f)-[:BELONGS_TO]->(m:Module)
            OPTIONAL MATCH (f)-[:IMPLEMENTS]->(c:Concept)
            OPTIONAL MATCH (f)-[r]->(other)
            RETURN f, m, collect(distinct c) as concepts, 
                   collect(distinct {type: type(r), node: other}) as relationships
            """
            params = {"name": function_name}
        
        with self.driver.session() as session:
            result = session.run(query, **params).single()
            if not result:
                return {}
            
            function = self._node_to_dict(result["f"], ["Function"])
            if result["m"]:
                function["module"] = self._node_to_dict(result["m"], ["Module"])
            function["concepts"] = [self._node_to_dict(c, ["Concept"]) for c in result["concepts"] if c]
            function["relationships"] = []
            
            for rel in result["relationships"]:
                if rel["node"]:
                    node_type = list(rel["node"].labels)[0]
                    function["relationships"].append({
                        "type": rel["type"],
                        "node": self._node_to_dict(rel["node"], [node_type])
                    })
            
            return function
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dictionary containing module information and related nodes
        """
        query = """
        MATCH (m:Module {name: $name})
        OPTIONAL MATCH (f:Function)-[:BELONGS_TO]->(m)
        OPTIONAL MATCH (c:Class)-[:BELONGS_TO]->(m)
        RETURN m, collect(distinct f) as functions, collect(distinct c) as classes
        """
        
        with self.driver.session() as session:
            result = session.run(query, name=module_name).single()
            if not result:
                return {}
            
            module = self._node_to_dict(result["m"], ["Module"])
            module["functions"] = [self._node_to_dict(f, ["Function"]) for f in result["functions"] if f]
            module["classes"] = [self._node_to_dict(c, ["Class"]) for c in result["classes"] if c]
            
            return module
    
    def semantic_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Perform semantic search on the graph using keywords extracted from the query.

        Args:
            query: The user query
            k: Number of top results to return

        Returns:
            Dictionary containing nodes and relationships for visualization
        """
        # Extract keywords from the query
        keywords = [word.lower() for word in query.split() if len(word) > 3]

        # Cypher query for semantic search
        cypher_query = """
        MATCH (n)
        WHERE n:Concept OR n:Module OR n:Function OR n:Class
        WHERE any(keyword IN $keywords WHERE 
                 toLower(n.name) CONTAINS keyword OR 
                 toLower(coalesce(n.description, '')) CONTAINS keyword)
        WITH n
        LIMIT 5
        MATCH path = (n)-[r*0..2]-(related)
        RETURN path
        LIMIT 50
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, keywords=keywords)

            # Process the paths to extract nodes and relationships
            nodes = {}
            relationships = []

            for record in result:
                path = record["path"]
                for node in path.nodes:
                    node_id = node.id
                    if node_id not in nodes:
                        node_type = list(node.labels)[0]
                        nodes[node_id] = self._node_to_dict(node, [node_type])

                for rel in path.relationships:
                    relationships.append({
                        "type": rel.type,
                        "source": rel.start_node.id,
                        "target": rel.end_node.id,
                        "properties": dict(rel)
                    })

            return {
                "nodes": list(nodes.values()),
                "relationships": relationships
            }

    def get_subgraph_for_query(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Extract a relevant subgraph based on the query.
        
        Args:
            query_text: Query text to search for
            
        Returns:
            List of dictionaries containing node information and relationships
        """
        # First, identify key entities in the query
        # This is a simplified approach - in a real implementation, 
        # you might use NER or other techniques to extract entities
        keywords = query_text.lower().split()
        keywords = [k for k in keywords if len(k) > 3]  # Filter out short words
        
        # Build a Cypher query to find relevant nodes and their relationships
        cypher_query = """
        MATCH (n)
        WHERE any(keyword IN $keywords WHERE 
                 toLower(n.name) CONTAINS keyword OR 
                 toLower(coalesce(n.description, '')) CONTAINS keyword)
        WITH n
        LIMIT 5
        MATCH path = (n)-[r*0..2]-(related)
        RETURN path
        LIMIT 50
        """
        
        with self.driver.session() as session:
            result = session.run(cypher_query, keywords=keywords)
            
            # Process the paths to extract nodes and relationships
            nodes = {}
            relationships = []
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    node_id = node.id
                    if node_id not in nodes:
                        node_type = list(node.labels)[0]
                        nodes[node_id] = self._node_to_dict(node, [node_type])
                
                for rel in path.relationships:
                    relationships.append({
                        "type": rel.type,
                        "source": rel.start_node.id,
                        "target": rel.end_node.id,
                        "properties": dict(rel)
                    })
            
            return {
                "nodes": list(nodes.values()),
                "relationships": relationships
            }
    
    def _node_to_dict(self, node, node_types: List[str]) -> Dict[str, Any]:
        """Convert a Neo4j node to a dictionary."""
        if not node:
            return {}
        
        result = dict(node)
        result["id"] = node.id
        result["type"] = node_types[0] if node_types else "Unknown"
        return result


class GraphRAGChain:
    """
    Class implementing the GraphRAG retrieval and generation chain.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the GraphRAG chain.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.retriever = GraphRetriever()
        self.retriever.connect()
        
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Define the prompt template for the LLM
        self.qa_template = """
        You are an AI assistant specialized in programming knowledge, particularly about {language}.
        
        Use the following information from the knowledge graph to answer the question:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Provide a detailed and accurate answer. If the information is not in the context, say so.
        If appropriate, include code examples to illustrate your answer.
        """
        
        self.qa_prompt = PromptTemplate(
            input_variables=["language", "context", "chat_history", "question"],
            template=self.qa_template
        )
        
        self.qa_chain = ({"chat_history": lambda x: self.memory.load_memory_variables({})["chat_history"],
                         "question": lambda x: x["question"],
                         "language": lambda x: x["language"],
                         "context": lambda x: x["context"]} | self.qa_prompt | self.llm)
    
    def close(self):
        """Close the Neo4j connection."""
        self.retriever.close()
    
    def _format_context(self, graph_data: Dict[str, Any]) -> str:
        """
        Format the graph data into a context string for the LLM.
        
        Args:
            graph_data: Graph data retrieved from Neo4j
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Format nodes
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                node_type = node.get("type", "Unknown")
                node_name = node.get("name", "Unnamed")
                node_desc = node.get("description", "")
                
                context_parts.append(f"{node_type}: {node_name}")
                if node_desc:
                    context_parts.append(f"Description: {node_desc}")
                
                # Add specific properties based on node type
                if node_type == "Function":
                    if "parameters" in node and node["parameters"]:
                        params_str = ", ".join([f"{p.get('name', '')}: {p.get('type', '')}" for p in node["parameters"]])
                        context_parts.append(f"Parameters: {params_str}")
                    if "return_type" in node:
                        context_parts.append(f"Returns: {node.get('return_type', '')}")
                    if "examples" in node and node["examples"]:
                        context_parts.append("Examples:")
                        for example in node["examples"]:
                            context_parts.append(f"```\n{example}\n```")
                
                context_parts.append("")  # Empty line for separation
        
        # Format relationships if available
        if "relationships" in graph_data:
            rel_dict = {}
            for rel in graph_data["relationships"]:
                source_id = rel["source"]
                target_id = rel["target"]
                rel_type = rel["type"]
                
                # Find source and target nodes
                source_node = next((n for n in graph_data["nodes"] if n.get("id") == source_id), {})
                target_node = next((n for n in graph_data["nodes"] if n.get("id") == target_id), {})
                
                source_name = source_node.get("name", f"Node {source_id}")
                target_name = target_node.get("name", f"Node {target_id}")
                
                key = f"{source_name} {rel_type} {target_name}"
                rel_dict[key] = True
            
            if rel_dict:
                context_parts.append("Relationships:")
                for rel in rel_dict.keys():
                    context_parts.append(f"- {rel}")
                context_parts.append("")
        
        # Handle single node results (e.g., from get_concept_info)
        if "type" in graph_data:
            node_type = graph_data.get("type", "Unknown")
            node_name = graph_data.get("name", "Unnamed")
            node_desc = graph_data.get("description", "")
            
            context_parts.append(f"{node_type}: {node_name}")
            if node_desc:
                context_parts.append(f"Description: {node_desc}")
            
            # Add related concepts
            if "related_concepts" in graph_data and graph_data["related_concepts"]:
                context_parts.append("Related Concepts:")
                for concept in graph_data["related_concepts"]:
                    context_parts.append(f"- {concept.get('name', '')}: {concept.get('description', '')}")
            
            # Add functions that implement this concept
            if "functions" in graph_data and graph_data["functions"]:
                context_parts.append("Related Functions:")
                for func in graph_data["functions"]:
                    context_parts.append(f"- {func.get('name', '')}: {func.get('description', '')}")
            
            # Add classes that implement this concept
            if "classes" in graph_data and graph_data["classes"]:
                context_parts.append("Related Classes:")
                for cls in graph_data["classes"]:
                    context_parts.append(f"- {cls.get('name', '')}: {cls.get('description', '')}")
        
        return "\n".join(context_parts)
    
    def query(self, question: str, language: str = "Rust") -> Dict[str, Any]:
        """
        Process a user query through the GraphRAG pipeline.
        
        Args:
            question: User's question
            language: Programming language context
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        # Step 1: Retrieve relevant subgraph from Neo4j
        graph_data = self.retriever.get_subgraph_for_query(question)
        
        # Step 2: Format the graph data into context for the LLM
        context = self._format_context(graph_data)
        
        # Step 3: Generate answer using the LLM
        response = self.qa_chain({
            "language": language,
            "context": context,
            "question": question
        })
        
        return {
            "answer": response["text"],
            "context": context,
            "graph_data": graph_data
        }
    
    def query_concept(self, concept_name: str, language: str = "Rust") -> Dict[str, Any]:
        """
        Query information about a specific concept.
        
        Args:
            concept_name: Name of the concept
            language: Programming language context
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        # Retrieve concept information
        concept_data = self.retriever.get_concept_info(concept_name)
        
        if not concept_data:
            return {
                "answer": f"I couldn't find information about the concept '{concept_name}'.",
                "context": "",
                "graph_data": {}
            }
        
        # Format the concept data into context for the LLM
        context = self._format_context(concept_data)
        
        # Generate answer using the LLM
        response = self.qa_chain({
            "language": language,
            "context": context,
            "question": f"Explain the concept of {concept_name} in {language}."
        })
        
        return {
            "answer": response["text"],
            "context": context,
            "graph_data": concept_data
        }
    
    def query_function(self, function_name: str, module: Optional[str] = None, language: str = "Rust") -> Dict[str, Any]:
        """
        Query information about a specific function.
        
        Args:
            function_name: Name of the function
            module: Module the function belongs to (optional)
            language: Programming language context
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        # Retrieve function information
        function_data = self.retriever.get_function_info(function_name, module)
        
        if not function_data:
            return {
                "answer": f"I couldn't find information about the function '{function_name}'.",
                "context": "",
                "graph_data": {}
            }
        
        # Format the function data into context for the LLM
        context = self._format_context(function_data)
        
        # Generate answer using the LLM
        module_str = f" in the {module} module" if module else ""
        response = self.qa_chain({
            "language": language,
            "context": context,
            "question": f"Explain how to use the {function_name} function{module_str} in {language}."
        })
        
        return {
            "answer": response["text"],
            "context": context,
            "graph_data": function_data
        }
    
    def query_module(self, module_name: str, language: str = "Rust") -> Dict[str, Any]:
        """
        Query information about a specific module.
        
        Args:
            module_name: Name of the module
            language: Programming language context
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        # Retrieve module information
        module_data = self.retriever.get_module_info(module_name)
        
        if not module_data:
            return {
                "answer": f"I couldn't find information about the module '{module_name}'.",
                "context": "",
                "graph_data": {}
            }
        
        # Format the module data into context for the LLM
        context = self._format_context(module_data)
        
        # Generate answer using the LLM
        response = self.qa_chain({
            "language": language,
            "context": context,
            "question": f"Explain the purpose and contents of the {module_name} module in {language}."
        })
        
        return {
            "answer": response["text"],
            "context": context,
            "graph_data": module_data
        }
    
    def compare_entities(self, entity1: str, entity2: str, language: str = "Rust") -> Dict[str, Any]:
        """
        Compare two programming entities (functions, concepts, etc.).
        
        Args:
            entity1: Name of the first entity
            entity2: Name of the second entity
            language: Programming language context
            
        Returns:
            Dictionary containing the answer and relevant context
        """
        # Search for both entities
        entity1_results = self.retriever.search_by_keyword(entity1, limit=1)
        entity2_results = self.retriever.search_by_keyword(entity2, limit=1)
        
        if not entity1_results or not entity2_results:
            missing = []
            if not entity1_results:
                missing.append(entity1)
            if not entity2_results:
                missing.append(entity2)
            
            return {
                "answer": f"I couldn't find information about: {', '.join(missing)}",
                "context": "",
                "graph_data": {}
            }
        
        # Get detailed information based on entity types
        entity1_data = entity1_results[0]
        entity2_data = entity2_results[0]
        
        entity1_type = entity1_data.get("type")
        entity2_type = entity2_data.get("type")
        
        if entity1_type == "Concept":
            entity1_full = self.retriever.get_concept_info(entity1_data.get("name"))
        elif entity1_type == "Function":
            entity1_full = self.retriever.get_function_info(entity1_data.get("name"), entity1_data.get("module"))
        elif entity1_type == "Module":
            entity1_full = self.retriever.get_module_info(entity1_data.get("name"))
        else:
            entity1_full = entity1_data
        
        if entity2_type == "Concept":
            entity2_full = self.retriever.get_concept_info(entity2_data.get("name"))
        elif entity2_type == "Function":
            entity2_full = self.retriever.get_function_info(entity2_data.get("name"), entity2_data.get("module"))
        elif entity2_type == "Module":
            entity2_full = self.retriever.get_module_info(entity2_data.get("name"))
        else:
            entity2_full = entity2_data
        
        # Combine the data
        combined_data = {
            "entity1": entity1_full,
            "entity2": entity2_full
        }
        
        # Format the combined data into context for the LLM
        context = f"Entity 1 ({entity1_type}): {entity1_data.get('name')}\n"
        context += self._format_context(entity1_full)
        context += f"\n\nEntity 2 ({entity2_type}): {entity2_data.get('name')}\n"
        context += self._format_context(entity2_full)
        
        # Generate answer using the LLM
        response = self.qa_chain({
            "language": language,
            "context": context,
            "question": f"Compare and contrast {entity1} and {entity2} in {language}. Explain their similarities, differences, and when to use each."
        })
        
        return {
            "answer": response["text"],
            "context": context,
            "graph_data": combined_data
        }


# Example usage
if __name__ == "__main__":
    # Initialize the GraphRAG chain
    chain = GraphRAGChain()
    
    # Example query
    result = chain.query("How does ownership work in Rust?")
    print(result["answer"])
    
    # Close connections
    chain.close()
