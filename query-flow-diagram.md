# RAG Chatbot Query Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Frontend<br/>(script.js)
    participant FastAPI as FastAPI Backend<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant Claude as Claude API<br/>(Anthropic)
    participant Tools as Search Tools<br/>(search_tools.py)
    participant VectorDB as Vector Store<br/>(ChromaDB)
    participant Session as Session Manager<br/>(session_manager.py)

    User->>Frontend: Types query & clicks Send
    
    Note over Frontend: Disables input<br/>Shows loading indicator
    
    Frontend->>FastAPI: POST /api/query<br/>{query, session_id}
    
    FastAPI->>Session: Create/Get session_id
    Session-->>FastAPI: session_id
    
    FastAPI->>RAG: query(query, session_id)
    
    RAG->>Session: Get conversation history
    Session-->>RAG: Previous messages
    
    RAG->>AI: generate_response()<br/>+ query<br/>+ history<br/>+ tool definitions
    
    AI->>Claude: API call with:<br/>- System prompt<br/>- User query<br/>- Tool definitions
    
    Claude-->>AI: Response with<br/>tool_use request
    
    AI->>Tools: Execute search tool
    Tools->>VectorDB: Semantic search<br/>for relevant chunks
    
    VectorDB-->>Tools: Matching course<br/>chunks & metadata
    Tools-->>AI: Search results
    
    AI->>Claude: Follow-up call with<br/>tool results
    Claude-->>AI: Final response
    
    AI-->>RAG: Generated answer
    
    RAG->>Tools: Get sources used
    Tools-->>RAG: Source list
    
    RAG->>Session: Store Q&A exchange
    
    RAG-->>FastAPI: (answer, sources)
    
    FastAPI-->>Frontend: JSON Response<br/>{answer, sources, session_id}
    
    Note over Frontend: Removes loading<br/>Displays answer<br/>Shows sources
    
    Frontend-->>User: Formatted response<br/>with citations
```

## Key Components

### 1. **Frontend Layer**
- **script.js**: Handles UI interactions, sends AJAX requests, displays responses

### 2. **API Layer**
- **FastAPI (app.py)**: REST API endpoints, request/response models, CORS handling

### 3. **RAG Orchestration**
- **RAGSystem**: Coordinates between all components
- **SessionManager**: Maintains conversation context
- **ToolManager**: Manages available search tools

### 4. **AI Processing**
- **AIGenerator**: Interfaces with Claude API
- **Claude API**: Generates responses, decides when to use tools

### 5. **Knowledge Base**
- **SearchTools**: Executes semantic searches
- **VectorStore (ChromaDB)**: Stores embeddings of course content
- **DocumentProcessor**: Chunks and processes course documents

## Data Flow

1. **Request Flow** (↓)
   - User input → Frontend → Backend → RAG System → AI Generator → Claude API

2. **Tool Execution** (↔)
   - Claude requests tool use → Search Tools → Vector Database → Return results

3. **Response Flow** (↑)
   - Claude response → AI Generator → RAG System → Backend → Frontend → User

## Session Management
- Session IDs maintain conversation continuity
- History is passed to Claude for context-aware responses
- Each Q&A exchange is stored for future reference

## Search Process
1. Claude analyzes query and decides if search is needed
2. Executes CourseSearchTool with semantic search
3. ChromaDB returns relevant chunks based on embeddings
4. Claude synthesizes search results into coherent answer