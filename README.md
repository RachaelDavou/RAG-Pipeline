# RAG Pipeline for Health & Medicine Q&A

A Jupyter notebook demonstrating Retrieval Augmented Generation (RAG) for answering domain-specific health questions. The system retrieves relevant information from documents before generating answers, reducing hallucinations and grounding responses in source material.Without RAG, LLMs answer from their training data which may be outdated or incomplete. 


## Requirements

Install the dependencies:

```
pip install langchain-community langchain-text-splitters langchain-core wikipedia sentence-transformers faiss-cpu openai numpy beautifulsoup4
```


## API Key Setup

This notebook requires an OpenAI API key for the generation step.

1. Open `rag_pipeline.ipynb`
2. Replace `"your-openai-api-key-here"` with your actual API key

To get an API key, go to https://platform.openai.com/api-keys


## How to Run

```
jupyter notebook rag_pipeline.ipynb
```

Run all cells sequentially to see the full RAG pipeline in action.


## How It Works

RAG combines retrieval with generation. Instead of asking the LLM to answer from memory, we first find relevant documents and include them in the prompt.

1. **Load Documents** - Wikipedia articles and web pages on health topics are loaded using the Wikipedia API and LangChain's WebBaseLoader.

2. **Split into Chunks** - Documents are split into smaller chunks using `RecursiveCharacterTextSplitter`. This is necessary because LLMs have token limits, and smaller chunks allow more precise retrieval. We use two key parameters:

   - `chunk_size=500`: Each chunk is ~500 characters, small enough to be precise but large enough to contain a complete idea or paragraph.
   
   - `chunk_overlap=100`: Chunks are overlapped and each consecutive chunks share 100 characters. This prevents losing information when sentences are split at chunk boundaries. For example, if a sentence gets cut in half, the overlap ensures the full sentence exists in at least one chunk.
   
   The splitter tries to break on natural boundaries in order: paragraph breaks → line breaks → sentence endings → spaces, keeping text as coherent as possible.

3. **Create Embeddings** - Each chunk is converted to a 384-dimensional vector using SentenceTransformers (`all-MiniLM-L6-v2`).

4. **Build FAISS Index** - Vectors are stored in a FAISS index for fast similarity search.

5. **RAG Query Pipeline**:
   - User asks a question.
   - Question is embedded and compared against chunk vectors.
   - Top-k most similar chunks are retrieved.
   - Chunks are combined with the question into an augmented prompt.
   - OpenAI generates an answer using only the provided context.


## Sample Questions

The notebook tests the system with 10 health-related questions:

- Why is vaccination important for public health?
- What are the common side effects of antibiotics?
- How does vitamin D deficiency affect the body?
- What are the major risk factors associated with heart disease?
- How does Covid‑19 primarily spread between people?
- What are the main symptoms of malaria?
- What is the recommended diet for preventing type 2 diabetes?
- How can high blood pressure be controlled through lifestyle?
- What are the typical signs of hepatitis infection?
- Why is regular physical activity beneficial for overall health?



