# GraphRAG Assistant

A chatbot application that uses Graph-based Retrieval Augmented Generation (GraphRAG) to provide accurate and contextually relevant responses for programming language assistance.

## Project Overview

GraphRAG Assistant is an AI-powered developer assistant that helps users learn and work with programming languages. It combines the power of large language models with a graph-based knowledge representation to provide contextually relevant answers, code examples, and best practices.

### Key Features

- **Semantic understanding** of documentation, modules, and syntax
- **Context-aware Q&A**: What does a function do? How is one concept different from another?
- **Best practice suggestions** based on official guidelines
- **Code snippet generation** based on real-world examples
- **Module dependency visualization** through an interactive graph view

## Architecture

The application uses a GraphRAG (Graph-based Retrieval Augmented Generation) architecture:

1. **Knowledge Graph**: Programming concepts, modules, functions, and their relationships are stored in a Neo4j graph database
2. **Retrieval Layer**: User queries trigger the retrieval of relevant subgraphs from the knowledge base
3. **LLM Integration**: The retrieved context is sent to a language model (like GPT-3.5) to generate accurate responses
4. **Frontend**: A responsive web interface with chat, code preview, and graph visualization

## Project Structure

```
graphrag-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI backend API
│   │   ├── langchain_chain.py   # GraphRAG retrieval pipeline
│   │   ├── graph_builder.py     # Neo4j knowledge graph construction
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Container configuration
├── frontend/                    # Next.js frontend
│   ├── pages/                   # Application pages
│   ├── components/              # React components
│   ├── styles/                  # CSS modules
│   └── next.config.js           # Next.js configuration
├── data/                        # Documentation data
│   └── rust_docs/               # Rust documentation
└── README.md                    # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 14+
- Neo4j Database (local or Neo4j Aura)
- OpenAI API key (or other LLM provider)

### Environment Setup

1. **Create environment files**

   Create a `.env` file in the `backend` directory:

   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   OPENAI_API_KEY=your_openai_api_key
   ```

### Installation

1. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

3. **Neo4j Setup**
   - Install Neo4j Desktop or use Neo4j Aura (cloud service)
   - Create a new database
   - Set your username and password in the `.env` file

### Running the Application

1. **Start the backend**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**
   Open your browser and navigate to `http://localhost:3000`

## Usage

1. **Ask questions** about programming concepts, functions, or modules
2. **View code examples** in the code preview panel
3. **Toggle the graph view** to see the knowledge graph visualization
4. **Select a programming language** from the dropdown (default is Rust)

## Data Ingestion

To add new documentation to the knowledge graph:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "Content-Type: application/json" \
  -d @path/to/your/docs_data.json
```

The documentation data should follow this format:

```json
[
  {
    "type": "concept",
    "name": "Ownership",
    "description": "Rust's central feature that manages memory safety without garbage collection",
    "language": "Rust",
    "related_concepts": ["Borrowing", "Lifetimes"]
  },
  {
    "type": "function",
    "name": "vec",
    "module": "std::vec",
    "description": "Creates a new vector",
    "parameters": [
      {"name": "elements", "type": "T", "description": "Elements to include"}
    ],
    "return_type": "Vec<T>",
    "examples": ["let v = vec![1, 2, 3];"]
  }
]
```

## Development

### Backend Components

- **graph_builder.py**: Handles Neo4j connection and knowledge graph construction
- **langchain_chain.py**: Implements the GraphRAG retrieval pipeline and LLM integration
- **main.py**: Provides the FastAPI endpoints for the frontend

### Frontend Components

- **ChatInterface**: Handles the chat UI and message display
- **CodePreview**: Displays code snippets with syntax highlighting
- **GraphView**: Visualizes the knowledge graph data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/hwchase17/langchain)
- Graph database powered by [Neo4j](https://neo4j.com/)
- Frontend built with [Next.js](https://nextjs.org/)
