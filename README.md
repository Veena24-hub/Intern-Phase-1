# Retrieval-Augmented Generation (RAG) Chatbot with MongoDB & Azure OpenAI

This notebook demonstrates the complete workflow for building a **Retrieval-Augmented Generation (RAG) chatbot**.  
The system integrates the following components:  

- **Knowledge Base Construction** – curating documents and splitting them into chunks.  
- **Embeddings** – generating dense vector representations using `SentenceTransformers`.  
- **Vector Database** – storing and retrieving embeddings with **MongoDB Atlas Vector Search** (with cosine similarity fallback).  
- **LLM Integration** – connecting to **Azure OpenAI GPT-3.5 Turbo** for context-aware responses.  
- **RAG Pipeline** – combining retrieval and generation for grounded answers.  
- **Evaluation & Testing** – validating chatbot responses on sample and edge queries.  

This notebook is structured for clarity, reproducibility, and deployment-readiness, making it a solid reference for **end-to-end RAG implementation**.


### Package Installation  

In this step, we install all the required Python libraries:  

- **sentence-transformers** – for generating embeddings of text chunks.  
- **numpy & scikit-learn** – for numerical operations and similarity calculations.  
- **langchain & langchain-text-splitters** – to handle document chunking and retrieval workflows.   
- **pymongo & dnspython** – to connect and interact with **MongoDB Atlas** (vector database).  
- **openai** – to integrate with **Azure OpenAI GPT models** for LLM responses.  



```python
# Install packages

!pip install sentence-transformers numpy scikit-learn
!pip install langchain langchain-text-splitters
!pip install pymongo dnspython --quiet
!pip install openai --quiet
```

    Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.12/dist-packages (5.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
    Requirement already satisfied: transformers<5.0.0,>=4.41.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.56.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.67.1)
    Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (2.8.0+cu126)
    Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (1.16.2)
    Requirement already satisfied: huggingface-hub>=0.20.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (0.35.0)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (11.3.0)
    Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.12/dist-packages (from sentence-transformers) (4.15.0)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (1.5.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.12/dist-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.19.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)
    Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (25.0)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2.32.4)
    Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub>=0.20.0->sentence-transformers) (1.1.10)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (75.2.0)
    Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.13.3)
    Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.6)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.80)
    Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (9.10.2.21)
    Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.4.1)
    Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (11.3.0.4)
    Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (10.3.7.77)
    Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (11.7.1.2)
    Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.5.4.2)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (0.7.1)
    Requirement already satisfied: nvidia-nccl-cu12==2.27.3 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (2.27.3)
    Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.77)
    Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (12.6.85)
    Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (1.11.1.6)
    Requirement already satisfied: triton==3.4.0 in /usr/local/lib/python3.12/dist-packages (from torch>=1.11.0->sentence-transformers) (3.4.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2024.11.6)
    Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.22.0)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.6.2)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=1.11.0->sentence-transformers) (1.3.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (3.0.2)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.4.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface-hub>=0.20.0->sentence-transformers) (2025.8.3)
    Requirement already satisfied: langchain in /usr/local/lib/python3.12/dist-packages (0.3.27)
    Requirement already satisfied: langchain-text-splitters in /usr/local/lib/python3.12/dist-packages (0.3.11)
    Requirement already satisfied: langchain-core<1.0.0,>=0.3.72 in /usr/local/lib/python3.12/dist-packages (from langchain) (0.3.76)
    Requirement already satisfied: langsmith>=0.1.17 in /usr/local/lib/python3.12/dist-packages (from langchain) (0.4.28)
    Requirement already satisfied: pydantic<3.0.0,>=2.7.4 in /usr/local/lib/python3.12/dist-packages (from langchain) (2.11.9)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.12/dist-packages (from langchain) (2.0.43)
    Requirement already satisfied: requests<3,>=2 in /usr/local/lib/python3.12/dist-packages (from langchain) (2.32.4)
    Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.12/dist-packages (from langchain) (6.0.2)
    Requirement already satisfied: tenacity!=8.4.0,<10.0.0,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from langchain-core<1.0.0,>=0.3.72->langchain) (8.5.0)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /usr/local/lib/python3.12/dist-packages (from langchain-core<1.0.0,>=0.3.72->langchain) (1.33)
    Requirement already satisfied: typing-extensions>=4.7 in /usr/local/lib/python3.12/dist-packages (from langchain-core<1.0.0,>=0.3.72->langchain) (4.15.0)
    Requirement already satisfied: packaging>=23.2 in /usr/local/lib/python3.12/dist-packages (from langchain-core<1.0.0,>=0.3.72->langchain) (25.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from langsmith>=0.1.17->langchain) (0.28.1)
    Requirement already satisfied: orjson>=3.9.14 in /usr/local/lib/python3.12/dist-packages (from langsmith>=0.1.17->langchain) (3.11.3)
    Requirement already satisfied: requests-toolbelt>=1.0.0 in /usr/local/lib/python3.12/dist-packages (from langsmith>=0.1.17->langchain) (1.0.0)
    Requirement already satisfied: zstandard>=0.23.0 in /usr/local/lib/python3.12/dist-packages (from langsmith>=0.1.17->langchain) (0.25.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.7.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /usr/local/lib/python3.12/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /usr/local/lib/python3.12/dist-packages (from pydantic<3.0.0,>=2.7.4->langchain) (0.4.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2->langchain) (3.4.3)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2->langchain) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2->langchain) (2.5.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2->langchain) (2025.8.3)
    Requirement already satisfied: greenlet>=1 in /usr/local/lib/python3.12/dist-packages (from SQLAlchemy<3,>=1.4->langchain) (3.2.4)
    Requirement already satisfied: anyio in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (4.10.0)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.12/dist-packages (from httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (1.0.9)
    Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.12/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (0.16.0)
    Requirement already satisfied: jsonpointer>=1.9 in /usr/local/lib/python3.12/dist-packages (from jsonpatch<2.0,>=1.33->langchain-core<1.0.0,>=0.3.72->langchain) (3.0.0)
    Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.12/dist-packages (from anyio->httpx<1,>=0.23.0->langsmith>=0.1.17->langchain) (1.3.1)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m36.0 MB/s[0m eta [36m0:00:00[0m
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m331.1/331.1 kB[0m [31m29.7 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

```

# Section 1: Create a knowledge base

In this step, we define a **small custom knowledge base** containing documents on topics like Python, Machine Learning, Web Development, and Discord Bots.  
Each document is wrapped into a **LangChain `Document` object** with metadata, making it easier to split, embed, and later retrieve relevant text chunks.  

This forms the foundation for our RAG pipeline.  


```python
# Create a simple knowledge base
print("Setting up knowledge base...")

long_docs = [
    """Python Programming Language Overview:
    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
    It emphasizes code readability with its notable use of significant whitespace.
    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
    The language is widely used for web development, data analysis, artificial intelligence, and automation.
    Popular Python frameworks include Django for web development, NumPy for scientific computing,
    and TensorFlow for machine learning applications.""",

    """Machine Learning and Artificial Intelligence:
    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    Supervised learning uses labeled data to train models for prediction tasks.
    Unsupervised learning finds patterns in data without labels, such as clustering similar items.
    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.
    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.""",

    """Web Development Technologies:
    Web development involves creating websites and web applications using various technologies.
    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.
    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.
    REST APIs provide a way for different systems to communicate over HTTP using standard methods.
    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.
    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.""",

    """Discord Bot Development:
    Discord bots are applications that can interact with Discord servers automatically.
    They can respond to messages, moderate chat, play music, and perform various automated tasks.
    Discord bots are built using Discord's API and can be developed in multiple programming languages.
    Python developers often use the discord.py library to create bots with features like slash commands.
    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.
    Common bot features include welcome messages, role management, music playback, and custom commands."""
]

# Create LangChain Document objects but keep a different variable name to avoid future shadowing
from langchain.schema import Document

langchain_documents = [
    Document(page_content=text, metadata={"source": f"doc_{i+1}", "doc_id": i})
    for i, text in enumerate(long_docs)
]

print(f"Created knowledge base with {len(langchain_documents)} documents")

# Quick validation prints (helps spot problems immediately)
print("\nSample metadata for doc 1:", langchain_documents[0].metadata)
print("\nSample text (first 300 chars):\n", langchain_documents[0].page_content[:300])

```

    Setting up knowledge base...
    Created knowledge base with 4 documents
    
    Sample metadata for doc 1: {'source': 'doc_1', 'doc_id': 0}
    
    Sample text (first 300 chars):
     Python Programming Language Overview:
        Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.
        It emphasizes code readability with its notable use of significant whitespace.
        Python supports multiple programming paradigms including procedural, object-o


# Section 2: Chunking and splitting

n this step, we prepare the knowledge base for embedding by **splitting long documents into smaller, overlapping text chunks**.  

- **Why chunking?**  
  Large documents are difficult to process directly with embeddings or LLMs. Breaking them into **smaller, context-rich chunks** allows for more precise retrieval and avoids exceeding token limits.  

- **How it works:**  
  - We use **LangChain’s `RecursiveCharacterTextSplitter`**.  
  - Each chunk is set to **1000 characters** with an **overlap of 50 characters**.  
  - Overlap ensures that information at the boundary of one chunk is also present in the next, reducing context loss.  
  - Chunks are stored as **LangChain `Document` objects** with metadata, then extracted into plain text (`chunks_texts`) for embedding.  

This chunking step bridges raw documents and vector embeddings, making retrieval accurate and contextually relevant.  





```python
# Creating a langchain text splitter

from langchain.text_splitter import RecursiveCharacterTextSplitter

chunk_size = 1000
chunk_overlap = 50
# separators order matters; keep paragraph and newline splits before sentences/space
separators = ["\n\n", "\n", ". ", ".", " "]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=separators
)
print(f"Text splitter created: {chunk_size} chars, {chunk_overlap} overlap")

```

    Text splitter created: 1000 chars, 50 overlap



```python
# Convert long_docs into LangChain Document objects

from langchain.schema import Document

if 'langchain_documents' in globals():
    documents = langchain_documents
else:
    documents = [
        Document(page_content=doc, metadata={"source": f"documents_{i+1}", "doc_id": i})
        for i, doc in enumerate(long_docs)
    ]

print(f"Created {len(documents)} documents to langchain format")

```

    Created 4 documents to langchain format



```python
# Split all documents into chunks

chunks = text_splitter.split_documents(documents)  # pass the list directly
print(f"Created {len(chunks)} total chunks from {len(documents)} original documents")

```

    Created 4 total chunks from 4 original documents



```python
# Show Chunks

print("Chunking breakdown:")
for i, chunk in enumerate(chunks):
    print(f"--- CHUNK {i+1} ---")
    print(f"Length: {len(chunk.page_content)} characters")
    print(f"Source: {chunk.metadata.get('source')}")
    # Print a shortened preview to keep output compact
    preview = chunk.page_content[:300].replace("\n", " ")
    print(f"Text (preview): '{preview}...'")

    # Show only first 3 chunks in detail to avoid huge output; indicate remaining count
    if i >= 2:
        remaining = max(0, len(chunks) - (i+1))
        print(f"  ... and {remaining} more chunks")
        break

    # Overlap check for chunks 2 and onward
    if i > 0:
        prev_chunk = chunks[i-1]
        current_start = chunk.page_content[:30]
        prev_end = prev_chunk.page_content[-30:]
        print(f"Overlap check - Previous chunk ended: '...{prev_end}'")
        print(f"Overlap check - Current chunk starts: '{current_start}...'")
    print()
print("="*60 + "\n")

```

    Chunking breakdown:
    --- CHUNK 1 ---
    Length: 597 characters
    Source: doc_1
    Text (preview): 'Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-o...'
    
    --- CHUNK 2 ---
    Length: 669 characters
    Source: doc_2
    Text (preview): 'Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervise...'
    Overlap check - Previous chunk ended: '...machine learning applications.'
    Overlap check - Current chunk starts: 'Machine Learning and Artificia...'
    
    --- CHUNK 3 ---
    Length: 629 characters
    Source: doc_3
    Text (preview): 'Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages li...'
      ... and 1 more chunks
    ============================================================
    



```python
# Cell 5: Extract chunk texts for embedding
chunks_texts = [chunk.page_content for chunk in chunks]
print(f"Extracted {len(chunks_texts)} chunk texts")

```

    Extracted 4 chunk texts


# Section 3: Creating Embeddings


In this step, we transform text chunks into **vector embeddings** that capture semantic meaning.  
These embeddings will later be stored in MongoDB and used for similarity-based retrieval.  

The workflow is as follows:  

1. **Load Embedding Model**  
   - We use **`all-MiniLM-L6-v2`** (from SentenceTransformers), a lightweight and fast model.  
   - It produces **384-dimensional embeddings**, inferred automatically from a test run.  

2. **Generate Embeddings for Chunks**  
   - The `model.encode()` method converts each text chunk into a dense vector.  
   - Batching (`batch_size=32`) ensures efficiency without overloading memory.  
   - The result is a NumPy array of shape `(num_chunks, embedding_dim)`.  

3. **Prepare Documents for MongoDB**  
   - Each chunk is packaged into a dictionary with:  
     - `_id` → numeric index  
     - `text` → chunk content  
     - `embedding` → embedding as a Python list  
     - `source` → document metadata  
   - Sanity checks ensure the embeddings all have the expected length (384).  

4. **Preview a Sample Document**  
   - The helper function `preview_document()` prints out metadata, text snippet, and embedding size for quick inspection.  

At the end of this section, we have a **list of documents (`documents_to_insert`)** ready to be stored in MongoDB with both text and embeddings.  



```python
# Choose a model

# all-MiniLM-L6-v2 is a good default (fast, 384-dim)
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
print("Loaded SentenceTransformer model:", MODEL_NAME)

# Quick check to infer embedding dimension
_sample = model.encode("test", convert_to_numpy=True)
EMBED_DIM = int(_sample.shape[0])
print("Inferred EMBED_DIM (sample):", EMBED_DIM)
```

    Loaded SentenceTransformer model: all-MiniLM-L6-v2
    Inferred EMBED_DIM (sample): 384



```python
# Creating Embedding for the chunk_texts (Batched)

assert 'chunks_texts' in globals() and isinstance(chunks_texts, list) and len(chunks_texts) > 0, \
    "chunks_texts missing or empty. Run Section 2 chunking cell first."

print(f"Creating embeddings for {len(chunks_texts)} chunks using model '{MODEL_NAME}' ...")

# Choose batch size by memory; 32 is usually safe.
batch_size = 32

# model.encode accepts a list and handles batching internally
doc_embeddings = model.encode(
    chunks_texts,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Convert to numpy & validate shape
doc_embeddings = np.asarray(doc_embeddings)
if doc_embeddings.ndim != 2:
    raise RuntimeError(f"Unexpected embeddings shape: {doc_embeddings.shape}")

EMBED_DIM = int(doc_embeddings.shape[1])
print(f"Created doc_embeddings with shape: {doc_embeddings.shape}")
print("EMBED_DIM (use this for Mongo index numDimensions):", EMBED_DIM)

# Display a small sample for sanity
print("First embedding sample (first 8 values):", doc_embeddings[0][:8].tolist())

```

    Creating embeddings for 4 chunks using model 'all-MiniLM-L6-v2' ...



    Batches:   0%|          | 0/1 [00:00<?, ?it/s]


    Created doc_embeddings with shape: (4, 384)
    EMBED_DIM (use this for Mongo index numDimensions): 384
    First embedding sample (first 8 values): [-0.04258284717798233, -0.013280070386826992, -0.02217341586947441, 0.01017401460558176, -0.01862110197544098, -0.10375279933214188, -0.022004667669534683, 0.02950233966112137]



```python
# Prepare documents for Mongo insertion (convert embeddings to lists)


assert 'chunks' in globals() and len(chunks) > 0, "Run chunking first (Section 2)."
assert 'doc_embeddings' in globals() and getattr(doc_embeddings, "shape", None), "Run embedding cell first."

documents_to_insert = []
for i, (chunk, emb) in enumerate(zip(chunks, doc_embeddings)):
    documents_to_insert.append({
        "_id": i,
        "text": chunk.page_content,
        "embedding": emb.tolist(),   # convert numpy array -> plain Python list
        "source": chunk.metadata.get("source", f"doc_{i+1}")
    })

print(f"Prepared {len(documents_to_insert)} documents for insertion into MongoDB.")

# Sanity checks: counts & embedding lengths
assert len(documents_to_insert) == doc_embeddings.shape[0], "Mismatch: docs vs embeddings"
bad = [ (i, len(d['embedding'])) for i,d in enumerate(documents_to_insert) if len(d['embedding']) != EMBED_DIM ]
if bad:
    print("Found docs with wrong embedding length (first 5):", bad[:5])
else:
    print("All prepared document embeddings have correct length:", EMBED_DIM)



```

    Prepared 4 documents for insertion into MongoDB.
    All prepared document embeddings have correct length: 384



```python
def preview_document(doc, limit=500):
    """
    Pretty-print a single document's metadata, text, and embedding info.

    Args:
        doc (dict): A document with keys "_id", "source", "text", "embedding".
        limit (int): Number of characters of text to preview (default: 500).
    """
    if not doc:
        print("⚠️ Empty or None document passed to preview_document")
        return

    text = doc.get("text", "")
    emb = doc.get("embedding", [])

    print("\n--- DOCUMENT PREVIEW ---")
    print(f"ID: {doc.get('_id')} | Source: {doc.get('source')}")
    print(f"Text length: {len(text)} chars | Embedding length: {len(emb)}")

    # Show repr to reveal whitespace/newlines
    print("\nrepr(text):")
    print(repr(text[:limit]))

    # Readable preview: replace newlines with ␤
    print("\nReadable preview (first {0} chars, newlines as ␤):".format(limit))
    print(text[:limit].replace("\n", "␤"))

documents_to_insert[0]
preview_document(documents_to_insert[0])
```

    
    --- DOCUMENT PREVIEW ---
    ID: 0 | Source: doc_1
    Text length: 597 chars | Embedding length: 384
    
    repr(text):
    'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web d'
    
    Readable preview (first 500 chars, newlines as ␤):
    Python Programming Language Overview:␤    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.␤    It emphasizes code readability with its notable use of significant whitespace.␤    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.␤    The language is widely used for web development, data analysis, artificial intelligence, and automation.␤    Popular Python frameworks include Django for web d


# Section 4: Create the MonogoDB Vector database


In this section, we set up **MongoDB Atlas** as the vector database to store and retrieve embeddings.  

The workflow covers five steps:  
1. **Imports & Config** → Load `pymongo`, set the vector index name (`INDEX_NAME`).  
2. **Validate Embedding Dimension** → Ensure `EMBED_DIM` from the embedding model is available for index creation.  
3. **Connect to MongoDB Atlas** → Read the connection string, connect to Atlas, and select the target database (`rag_db`) and collection (`chunks`).  
4. **Insert Documents & Define Index** → Insert all chunk embeddings into MongoDB, then output a JSON schema for creating an **Atlas Vector Search index** (cosine similarity, dimension = `EMBED_DIM`).  
5. **Search Helpers & Test** → Define helper functions for querying: Atlas vector search (preferred) and manual cosine similarity (fallback). Finally, run a sample query to validate retrieval.  

By the end of this section, the knowledge base is stored in MongoDB with vector search enabled, making it ready for retrieval in the RAG pipeline.  




```python
# Imports and config

from pymongo import MongoClient
import os
import json
import numpy as np

# Names / settings
INDEX_NAME = "vector_index"   # Atlas index name you will create
print("INDEX_NAME:", INDEX_NAME)

# EMBED_DIM
assert 'EMBED_DIM' in globals(), "Run embeddings section first to set EMBED_DIM."
print("EMBED_DIM (from embeddings):", EMBED_DIM)

```

    INDEX_NAME: vector_index
    EMBED_DIM (from embeddings): 384



```python
# Connecting to MongoDB Atlas

MONGODB_URL = os.getenv("MONGODB_URL") or None
if MONGODB_URL is None:
    try:
        from google.colab import userdata
        MONGODB_URL = userdata.get('MONGODB_URL')
    except Exception:
        pass

assert MONGODB_URL and MONGODB_URL.startswith("mongodb"), "MONGODB_URL missing or invalid. Set it in env or Colab secrets."

mongo_client = MongoClient(MONGODB_URL)
mongo_client.admin.command('ping')   # will raise if connection fails
print("Connected to MongoDB Atlas")

db = mongo_client["rag_db"]
collection = db["chunks"]
print("Using database:", db.name, "collection:", collection.name)

```

    Connected to MongoDB Atlas
    Using database: rag_db collection: chunks



```python
# Insert chunks and embeddings into Mongo

assert 'chunks' in globals() and len(chunks) > 0
assert 'doc_embeddings' in globals() and getattr(doc_embeddings, "shape", None)

# Build Mongo documents
documents_to_insert = [
    {
        "_id": i,
        "text": chunk.page_content,
        "embedding": emb.tolist(),
        "source": chunk.metadata.get("source", f"doc_{i+1}")
    }
    for i, (chunk, emb) in enumerate(zip(chunks, doc_embeddings))
]

print("Prepared documents:", len(documents_to_insert))

# Clear collection and insert
del_res = collection.delete_many({})
print("Cleared existing documents:", del_res.deleted_count)

ins = collection.insert_many(documents_to_insert)
print("Inserted documents:", len(ins.inserted_ids))

# Quick sanity sample
sample = collection.find_one({}, {"_id":1,"source":1,"text":1,"embedding":1})
print("Sample in DB →", {
    "_id": sample["_id"],
    "source": sample.get("source"),
    "text_len": len(sample["text"]),
    "embedding_len": len(sample["embedding"])
})
```

    Prepared documents: 4
    Cleared existing documents: 2
    Inserted documents: 4
    Sample in DB → {'_id': 0, 'source': 'doc_1', 'text_len': 597, 'embedding_len': 384}



```python
# Atlas vector index definition

vector_index_definition = {
  "fields": [
    {"type": "vector", "path": "embedding", "numDimensions": EMBED_DIM, "similarity": "cosine"}
  ]
}

print("Recommended Atlas Vector Search definition (use in Atlas UI):")
print(json.dumps(vector_index_definition, indent=2))

#------ Create this index in Atlas manually:
#   Cluster → Search → Create Search Index → Atlas Vector Search
#   Database: rag_db
#   Collection: chunks
#   Index Name: vector_index
#   Paste the JSON above, then wait until status = ACTIVE

```

    Recommended Atlas Vector Search definition (use in Atlas UI):
    {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "numDimensions": 384,
          "similarity": "cosine"
        }
      ]
    }



```python
# Search helpers (Atlas + fallback)

def mongodb_vector_search(query_text, top_k=3, debug=False):
    """Try Atlas Vector Search first. Returns [] if fails."""
    try:
        q_emb = model.encode([query_text], convert_to_numpy=True)[0].tolist()
        pipeline = [
            {"$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": q_emb,
                "numCandidates": top_k * 5,
                "limit": top_k
            }},
            {"$project": {"_id":1,"text":1,"source":1,"score":{"$meta":"vectorSearchScore"}}}
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        if debug:
            print("Atlas vector search failed:", str(e))
        return []


def fallback_search(query_text, top_k=3):
    """Manual cosine similarity over stored embeddings."""
    qemb = np.asarray(model.encode([query_text], convert_to_numpy=True)[0], dtype=np.float32)
    qnorm = np.linalg.norm(qemb)
    sims = []
    for d in collection.find({}, {"_id":1,"text":1,"source":1,"embedding":1}):
        d_emb = np.asarray(d["embedding"], dtype=np.float32)
        denom = qnorm * np.linalg.norm(d_emb)
        score = float(np.dot(qemb, d_emb) / denom) if denom != 0 else 0.0
        sims.append({"_id": d["_id"], "text": d["text"], "source": d["source"], "score": score})
    return sorted(sims, key=lambda x: x["score"], reverse=True)[:top_k]


def test_search(query="What is Python programming?", top_k=3):
    """Run a query, show Atlas results if available, else fallback."""
    print(f"\n🔎 Query: {query}")
    results = mongodb_vector_search(query, top_k=top_k)
    if results:
        print("[ATLAS] Results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. score={r.get('score'):.4f} | source={r.get('source')}")
            print("   preview:", r.get('text','')[:200].replace("\n"," "))
    else:
        print("⚠ [FALLBACK] Atlas not available, using cosine similarity:")
        results = fallback_search(query, top_k=top_k)
        for i, r in enumerate(results, 1):
            print(f"{i}. score={r['score']:.4f} | source={r['source']}")
            print("   preview:", r['text'][:200].replace("\n"," "))
    return results


# Run a sample query
test_search("What is Python programming?", top_k=3)
```

    
    🔎 Query: What is Python programming?
    [ATLAS] Results:
    1. score=0.9127 | source=doc_1
       preview: Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of signi
    2. score=0.7424 | source=doc_4
       preview: Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automa
    3. score=0.6612 | source=doc_3
       preview: Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and J





    [{'_id': 0,
      'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.',
      'source': 'doc_1',
      'score': 0.912684440612793},
     {'_id': 3,
      'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.",
      'source': 'doc_4',
      'score': 0.7424005270004272},
     {'_id': 2,
      'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.',
      'source': 'doc_3',
      'score': 0.6612247228622437}]



# Section 5: LLM integration and prompt engineering


In this section, we connect the pipeline to **Azure OpenAI GPT-3.5 Turbo** and design structured prompts to guide the model’s responses.  
The goal is to make the LLM answer strictly from retrieved context while avoiding hallucination.  

The workflow covers five steps:  

1. **Azure OpenAI Setup**  
   - Import the `AzureOpenAI` client.  
   - Load API credentials (`AZURE_OPENAI_KEY`, endpoint, deployment name, API version) from Colab secrets.  
   - Initialize the client to enable chat completions.  

2. **Prompt Templates**  
   - Define a **system prompt** (instructs the model to only use provided context and cite sources).  
   - Define a **user prompt template** with placeholders for `{context}` and `{question}`.  

3. **Context Builder**  
   - Implement `build_context_from_docs()` to assemble retrieved documents into a concise context string.  
   - Each snippet is prefixed with `[source:doc_X]` to support inline citations.  

4. **Azure Chat Wrapper**  
   - Create `call_azure_chat()`, a helper function that sends system and user prompts to Azure OpenAI.  
   - Handles parameters like `max_tokens`, `temperature`, and error cases with optional debug printing.  

5. **RAG Query Helper**  
   - Define `rag_query()` to tie everything together:  
     - Retrieve top-k documents (Atlas or fallback).  
     - Build context from docs.  
     - Format the user prompt.  
     - Call the LLM and extract inline sources.  
   - Returns a structured result with the answer, sources, and retrieved documents.  

By the end of this section, the chatbot can **retrieve context + generate grounded answers** using Azure OpenAI, with proper source citations.  



```python
# Imports & Azure OpenAI config

from openai import AzureOpenAI
import os, traceback
from google.colab import userdata

# Load Azure key

print("Azure OpenAI setup starting...")

os.environ['AZURE_OPENAI_KEY '] = userdata.get('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = "https://jibz3-mfxwvj2c-swedencentral.cognitiveservices.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2025-01-01-preview"  # From your screenshot
AZURE_OPENAI_KEY = userdata.get('AZURE_OPENAI_KEY')
AZURE_DEPLOYMENT_NAME = "gpt-35-turbo"
AZURE_API_VERSION = "2024-12-01-preview"

assert AZURE_OPENAI_KEY, "AZURE_OPENAI_KEY not found in env/Colab secrets."
assert AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT not found. Set your Azure OpenAI endpoint URL."

print("Azure OpenAI configuration loaded. Deployment:", AZURE_DEPLOYMENT_NAME)

# Initialize client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    base_url=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_API_VERSION
)
print("AzureOpenAI client initialized.")

```

    Azure OpenAI setup starting...
    Azure OpenAI configuration loaded. Deployment: gpt-35-turbo
    AzureOpenAI client initialized.



```python
# Prompt templates & context formatter

SYSTEM_PROMPT = (
    "You are an assistant that answers user questions using ONLY the provided CONTEXT. "
    "Cite sources inline using [source:doc_X]. If the answer cannot be found in the context, "
    "say 'I don't know' and do not hallucinate."
)

# This template will receive {context} and {question}

USER_PROMPT_TEMPLATE = (
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}\n\n"
    "INSTRUCTIONS:\n"
    "- Answer based only on the CONTEXT above.\n"
    "- Keep the answer concise and include source tags like [source:doc_1].\n"
    "- If context doesn't contain the answer, reply: 'I don't know'.\n\n"
    "ANSWER:"
)

def build_context_from_docs(docs, per_doc_chars=500):
    """
    Build a single context string from retrieved docs.
    Each doc is a dict with keys: _id, text, source, score (if present).
    Truncate each doc to per_doc_chars characters to keep prompts small.
    """
    parts = []
    for d in docs:
        src = d.get("source") or f"doc_{d.get('_id')}"
        text = d.get("text", "")
        # clean whitespace and truncate
        text_snippet = text.strip().replace("\n", " ")[:per_doc_chars].strip()
        parts.append(f"[source:{src}] {text_snippet}")
    return "\n\n".join(parts)

```


```python
#  Azure call wrapper

def call_azure_chat(prompt_system, prompt_user, max_tokens=350, temperature=0.0, debug=False):
    """
    Calls Azure OpenAI chat completion. Returns the assistant text.
    prompt_system: system message string
    prompt_user: user message string (full prompt including context + question)
    """
    try:
        messages = [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
        if debug:
            print("Calling Azure with messages (truncated):")
            print("SYSTEM:", prompt_system[:300])
            print("USER:", prompt_user[:800])

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Extract text safely
        text = response.choices[0].message.content
        return text.strip()
    except Exception as e:
        if debug:
            traceback.print_exc()
        return f"[LLM_ERROR] {str(e)}"

```


```python
#  RAG helper: retrieval + LLM generation

def rag_query(user_question, top_k=3, per_doc_chars=500, use_atlas=True, llm_debug=False):
    """
    End-to-end RAG:
      1) retrieve top_k docs (Atlas vector search )
      2) build context string
      3) call LLM with system + user prompt
    Returns: dict with keys: answer, sources (list), docs (retrieved)
    """
    # 1) Retreive: try Atlas first, fallback to manual cosine if needed
    docs = []
    if use_atlas:
        try:
            docs = mongodb_vector_search(user_question, top_k=top_k)
        except Exception:
            docs = []
    if not docs:
        docs = fallback_search(user_question, top_k=top_k)

    # normalize docs: ensure dicts have _id,text,source
    normalized = []
    for d in docs:
        # If Atlas returns Mongo docs (with score meta), keep keys consistent
        normalized.append({
            "_id": d.get("_id"),
            "text": d.get("text") or d.get("page_content") or "",
            "source": d.get("source") or f"doc_{d.get('_id')}",
            "score": d.get("score")
        })

    # 2) Build context
    context = build_context_from_docs(normalized, per_doc_chars=per_doc_chars)

    # 3) Build full prompt and call LLM
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=user_question)
    answer = call_azure_chat(SYSTEM_PROMPT, user_prompt, debug=llm_debug)

    # Extract used sources from answer heuristically (simple)
    used_sources = []
    for part in normalized:
        tag = f"[source:{part['source']}]"
        if tag in answer:
            used_sources.append(part['source'])

    # If model responded with hallucination marker or error, you might enforce fallback text
    if answer.strip().lower().startswith("[llm_error]"):
        final_answer = "Error from LLM: " + answer
    else:
        final_answer = answer

    return {"answer": final_answer, "sources": used_sources, "docs": normalized, "context": context}

```


```python
# Example uses

q = "Who created Python and when?"
res = rag_query(q, top_k=3, per_doc_chars=400, llm_debug=False)
print("\n=== RAG ANSWER ===")
print(res["answer"])
print("\nSources used (detected):", res["sources"])
print("\nRetrieved docs (ids & sources):")
for d in res["docs"]:
    print(d["_id"], d["source"], f"(score={d.get('score')})")

```

    
    === RAG ANSWER ===
    Python was created by Guido van Rossum in 1991 [source:doc_1].
    
    Sources used (detected): ['doc_1']
    
    Retrieved docs (ids & sources):
    0 doc_1 (score=0.8238394856452942)
    3 doc_4 (score=0.6892796754837036)
    2 doc_3 (score=0.5887914896011353)


# Section 6: RAG pipeline


In this section, we bring together all previous components (embeddings, MongoDB, and LLM) into a full **Retrieval-Augmented Generation (RAG) pipeline**.  
The pipeline handles retrieval, context-building, prompt creation, and LLM response in a single flow.  

The workflow covers four steps:  

1. **Retrieve Documents for a Query**  
   - `retrieve_docs_for_query()` encodes the user query with the SentenceTransformer model.  
   - Attempts **Atlas Vector Search** first; if it fails, falls back to manual cosine similarity.  
   - Returns a normalized list of docs with `_id`, `text`, `source`, and `score`.  

2. **Build RAG Prompt**  
   - `build_rag_prompt()` assembles the retrieved docs into a context string.  
   - Uses the system + user templates from Section 5.  
   - Returns a dict with: system message, user prompt, and context.  

3. **Call LLM**  
   - `call_llm_for_rag()` is a thin wrapper around the Azure helper.  
   - Keeps the pipeline modular by separating LLM calling logic from retrieval.  

4. **Run Full RAG Pipeline**  
   - `run_rag_pipeline()` orchestrates the entire workflow:  
     - Retrieve docs → Build prompt → Query LLM → Extract cited sources.  
   - Handles edge cases (e.g., if the LLM errors, a safe error message is returned).  
   - Returns a structured result with `question`, `answer`, `docs`, `sources`, and `context`.  
   - A quick test query (“What is Python programming?”) validates the flow end-to-end.  

By the end of this section, the system can answer user questions using **retrieved context + LLM reasoning** in one unified function.  



```python

#  Embed Query & Retrieve (Atlas first, fallback)

import numpy as np
from typing import List, Dict

def retrieve_docs_for_query(query: str, top_k: int = 3, debug: bool = False) -> List[Dict]:
    """
    1) Encode the query using the SentenceTransformer `model`.
    2) Attempt Atlas vector search. If it returns nothing or fails, use fallback cosine search.
    3) Return a list of retrieved doc dicts with keys: _id, text, source, score (score may be None).
    """
    # 1) embed query
    q_emb = model.encode([query], convert_to_numpy=True)[0].tolist()
    if debug:
        print(f"[retrieve] query embedded (len={len(q_emb)})")

    # 2) Try Atlas
    docs = []
    try:
        docs = mongodb_vector_search(query, top_k=top_k)
        if debug:
            print(f"[retrieve] Atlas returned {len(docs)} docs")
    except Exception as e:
        if debug:
            print("[retrieve] Atlas search exception:", e)
        docs = []

    # 3) Fallback if Atlas returned nothing
    if not docs:
        if debug:
            print("[retrieve] Using fallback_search (manual cosine)")
        docs = fallback_search(query, top_k=top_k)

    # Normalize returned docs to expected dict shape
    normalized = []
    for d in docs:
        normalized.append({
            "_id": d.get("_id"),
            "text": d.get("text") or d.get("page_content") or "",
            "source": d.get("source") or f"doc_{d.get('_id')}",
            "score": d.get("score")
        })
    return normalized

```


```python

# Prompt Builder (assemble retrieved chunks into a context)

def build_rag_prompt(question: str,
                     docs: List[Dict],
                     per_doc_chars: int = 500,
                     system_prompt: str = None,
                     user_template: str = None) -> Dict[str, str]:
    """
    Build the system + user messages for the LLM from the question and retrieved docs.
    Returns a dict: {'system': system_prompt, 'user': user_prompt, 'context': context_str}
    """
    # Use templates from Section 5 if not provided
    sys_prompt = system_prompt or SYSTEM_PROMPT
    user_tmpl = user_template or USER_PROMPT_TEMPLATE

    # Build context (reuse helper from Section 5)
    context_str = build_context_from_docs(docs, per_doc_chars=per_doc_chars)

    # Fill the user prompt template
    user_prompt = user_tmpl.format(context=context_str, question=question)

    return {"system": sys_prompt, "user": user_prompt, "context": context_str}

```


```python

# LLM Call

def call_llm_for_rag(system_msg: str, user_msg: str, max_tokens: int = 350, temperature: float = 0.0, debug: bool = False):
    """
    Thin wrapper around the Azure call helper. Keeps the interface clear for the pipeline.
    Returns the LLM string answer (raw).
    """
    # We reuse call_azure_chat from Section 5 which already handles errors and debug printing
    return call_azure_chat(system_msg, user_msg, max_tokens=max_tokens, temperature=temperature, debug=debug)

```


```python

# Full RAG pipeline wrapper (retrieve → prompt → LLM → return)

def run_rag_pipeline(question: str,
                     top_k: int = 3,
                     per_doc_chars: int = 500,
                     max_tokens: int = 350,
                     temperature: float = 0.0,
                     use_atlas: bool = True,
                     debug: bool = False) -> Dict:
    """
    End-to-end RAG pipeline.
    Returns a dictionary:
      {
        "question": str,
        "answer": str,
        "docs": [retrieved docs],
        "sources": [list of detected sources],
        "context": str
      }
    """
    # 1) Retrieve
    docs = retrieve_docs_for_query(question, top_k=top_k, debug=debug if debug else False)

    if debug:
        print(f"[RAG] Retrieved {len(docs)} docs. ids:", [d['_id'] for d in docs])

    # 2) Build prompt
    prompt_obj = build_rag_prompt(question, docs, per_doc_chars=per_doc_chars)
    if debug:
        print("[RAG] Context preview:", prompt_obj['context'][:400])

    # 3) Call LLM
    answer = call_llm_for_rag(prompt_obj['system'], prompt_obj['user'], max_tokens=max_tokens, temperature=temperature, debug=debug)

    # 4) Extract sources heuristically from answer (detect [source:doc_X] tags)
    used_sources = []
    for d in docs:
        tag = f"[source:{d['source']}]"
        if tag in answer:
            used_sources.append(d['source'])

    # 5) Safety: if LLM responded with an error marker, convert to safe message
    if isinstance(answer, str) and answer.lower().startswith("[llm_error]"):
        final_answer = "Error from LLM: " + answer
    else:
        final_answer = answer

    return {
        "question": question,
        "answer": final_answer,
        "docs": docs,
        "sources": used_sources,
        "context": prompt_obj["context"]
    }


# Quick test (example)

if __name__ == "__main__" or True:
    demo_q = "What is Python programming?"
    res = run_rag_pipeline(demo_q, top_k=3, per_doc_chars=400, debug=True)
    print("\n=== RAG RESULT ===")
    print("Answer:\n", res["answer"])
    print("\nSources:", res["sources"])
    print("\nRetrieved docs ids:", [d["_id"] for d in res["docs"]])

```

    [retrieve] query embedded (len=384)
    [retrieve] Atlas returned 3 docs
    [RAG] Retrieved 3 docs. ids: [0, 3, 2]
    [RAG] Context preview: [source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web developm
    Calling Azure with messages (truncated):
    SYSTEM: You are an assistant that answers user questions using ONLY the provided CONTEXT. Cite sources inline using [source:doc_X]. If the answer cannot be found in the context, say 'I don't know' and do not hallucinate.
    USER: CONTEXT:
    [source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analy
    
    [source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.p
    
    === RAG RESULT ===
    Answer:
     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It supports multiple programming paradigms and is widely used for web development [source:doc_1].
    
    Sources: ['doc_1']
    
    Retrieved docs ids: [0, 3, 2]


# Section 7: RAG Chatbot


In this section, we extend the RAG pipeline into a **multi-turn conversational chatbot**.  
Unlike the single-question pipeline in Section 6, this chatbot maintains **conversation memory**, allowing it to respond contextually across turns.  

The workflow includes four parts:  

1. **Conversation Memory Setup**  
   - Initialize `conversation_history` as a list of user/assistant messages.  
   - This memory lets the assistant recall prior context in an ongoing conversation.  

2. **Chatbot Turn Function**  
   - `_format_history_for_prompt()` compresses recent history into a short snippet for the LLM.  
   - `chatbot_turn()` handles a single interaction:  
     - Stores the user’s input.  
     - Adds conversation history to the query for context.  
     - Runs the **RAG pipeline** to retrieve docs and generate an answer.  
     - Stores the assistant’s reply back in history.  
   - Returns both the assistant’s answer and full pipeline details (sources, retrieved docs, context).  

3. **Single-Turn Demo**  
   - Runs `chatbot_turn("Who created Python and when?")` to test the chatbot flow.  
   - Prints the bot’s answer, retrieved document IDs, and detected sources.  

4. **Interactive Loop**  
   - Starts a live conversation where the user can continuously ask questions.  
   - The chatbot responds until the user types `"exit"` or `"quit"`.  

 **Why this section matters:**  
This step transforms the project from a **retrieval pipeline** into a fully usable **chat interface**.  
The chatbot can now handle **dialogue continuity** (memory of past turns) while grounding its answers in retrieved knowledge, making it practical for real-world applications.  



```python
# Conversation memory setup

conversation_history = []

```


```python

# chatbot_turn - one conversational turn using RAG

from typing import Tuple, Dict, Any

# Ensure conversation history exists (Cell 7.1 should create this, but safe-guard here)
try:
    conversation_history  # noqa: F821
except NameError:
    conversation_history = []  # each item: {"role": "user"|"assistant", "content": str}

def _format_history_for_prompt(history, max_turns=6) -> str:
    """
    Format the last few turns into a short text block to include in the query.
    Uses alternating User / Assistant lines to give context to the LLM.
    """
    if not history:
        return ""
    # keep only the last max_turns entries (counting both user+assistant as separate turns)
    snippet = history[-max_turns:]
    lines = []
    for h in snippet:
        role = h.get("role", "user")
        content = h.get("content", "").strip().replace("\n", " ")
        prefix = "User:" if role == "user" else "Assistant:"
        lines.append(f"{prefix} {content}")
    return "\n".join(lines)

def chatbot_turn(user_input: str,
                 top_k: int = 3,
                 per_doc_chars: int = 500,
                 max_history_turns: int = 6,
                 debug: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Handle one chat turn:
      - Add user turn to conversation_history
      - Build a question that includes a short history snippet
      - Run the RAG pipeline
      - Append assistant reply to conversation_history
    Returns:
      (assistant_text, pipeline_result_dict)
    """
    global conversation_history

    # 1) store user turn
    conversation_history.append({"role": "user", "content": user_input})

    # 2) build a short history snippet to include in retrieval/prompt
    history_snippet = _format_history_for_prompt(conversation_history, max_turns=max_history_turns)

    # Compose the effective question to send to RAG.
    # We keep the final natural question clear for embedding/retrieval, but include the snippet so LLM sees the dialog.
    if history_snippet:
        # Make sure not to create an enormous query — this is just a short summary of recent turns.
        effective_question = f"Conversation History:\n{history_snippet}\n\nUser question:\n{user_input}"
    else:
        effective_question = user_input

    if debug:
        print("=== chatbot_turn debug ===")
        print("Effective question sent to RAG (preview):", effective_question[:400])

    # 3) Run the RAG pipeline (retrieve -> prompt -> LLM)
    pipeline_result = run_rag_pipeline(
        effective_question,
        top_k=top_k,
        per_doc_chars=per_doc_chars,
        debug=debug
    )

    assistant_text = pipeline_result.get("answer", "").strip()

    # 4) Append assistant reply to history
    conversation_history.append({"role": "assistant", "content": assistant_text})

    if debug:
        print("Assistant preview:", assistant_text[:200])
        print("Sources returned:", pipeline_result.get("sources"))

    return assistant_text, pipeline_result

```


```python
# run one turn
reply, info = chatbot_turn("Who created Python and when?", top_k=3, debug=True)
print("Bot:", reply)
# view retrieved docs & sources:
print("Retrieved doc ids:", [d["_id"] for d in info["docs"]])
print("Detected sources in answer:", info["sources"])

```

    === chatbot_turn debug ===
    Effective question sent to RAG (preview): Conversation History:
    User: Who created Python and when?
    
    User question:
    Who created Python and when?
    [retrieve] query embedded (len=384)
    [retrieve] Atlas returned 3 docs
    [RAG] Retrieved 3 docs. ids: [0, 3, 2]
    [RAG] Context preview: [source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web developm
    Calling Azure with messages (truncated):
    SYSTEM: You are an assistant that answers user questions using ONLY the provided CONTEXT. Cite sources inline using [source:doc_X]. If the answer cannot be found in the context, say 'I don't know' and do not hallucinate.
    USER: CONTEXT:
    [source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d
    
    [source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API a
    Assistant preview: Python was created by Guido van Rossum in 1991 [source:doc_1].
    Sources returned: ['doc_1']
    Bot: Python was created by Guido van Rossum in 1991 [source:doc_1].
    Retrieved doc ids: [0, 3, 2]
    Detected sources in answer: ['doc_1']



```python
# Interactive loop

while True:
    q = input("You: ")
    if q.lower() in ["exit", "quit"]: break
    answer = chatbot_turn(q)
    print("Bot:", answer)

```

    You: What is python?
    Bot: ('Python is a high-level, interpreted programming language that emphasizes code readability and supports multiple programming paradigms [source:doc_1].', {'question': 'Conversation History:\nUser: Who created Python and when?\nAssistant: Python was created by Guido van Rossum in 1991 [source:doc_1].\nUser: What is python?\n\nUser question:\nWhat is python?', 'answer': 'Python is a high-level, interpreted programming language that emphasizes code readability and supports multiple programming paradigms [source:doc_1].', 'docs': [{'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.8085494041442871}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7318991422653198}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.584436297416687}], 'sources': ['doc_1'], 'context': "[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: What is special about python formatting?
    Bot: ('Python is notable for its use of significant whitespace, which emphasizes code readability [source:doc_1].', {'question': 'Conversation History:\nUser: Who created Python and when?\nAssistant: Python was created by Guido van Rossum in 1991 [source:doc_1].\nUser: What is python?\nAssistant: Python is a high-level, interpreted programming language that emphasizes code readability and supports multiple programming paradigms [source:doc_1].\nUser: What is special about python formatting?\n\nUser question:\nWhat is special about python formatting?', 'answer': 'Python is notable for its use of significant whitespace, which emphasizes code readability [source:doc_1].', 'docs': [{'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.8298357725143433}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7143781185150146}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6015096306800842}], 'sources': ['doc_1'], 'context': "[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: Does Python support multiple programming paradigms?
    Bot: ('Python supports multiple programming paradigms including procedural, object-oriented, and functional programming [source:doc_1].', {'question': 'Conversation History:\nAssistant: Python was created by Guido van Rossum in 1991 [source:doc_1].\nUser: What is python?\nAssistant: Python is a high-level, interpreted programming language that emphasizes code readability and supports multiple programming paradigms [source:doc_1].\nUser: What is special about python formatting?\nAssistant: Python is notable for its use of significant whitespace, which emphasizes code readability [source:doc_1].\nUser: Does Python support multiple programming paradigms?\n\nUser question:\nDoes Python support multiple programming paradigms?', 'answer': 'Python supports multiple programming paradigms including procedural, object-oriented, and functional programming [source:doc_1].', 'docs': [{'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.8404381275177002}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7073812484741211}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.5924005508422852}], 'sources': ['doc_1'], 'context': "[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: Name a Python library for scientific computing.
    Bot: ("I don't know", {'question': 'Conversation History:\nAssistant: Python is a high-level, interpreted programming language that emphasizes code readability and supports multiple programming paradigms [source:doc_1].\nUser: What is special about python formatting?\nAssistant: Python is notable for its use of significant whitespace, which emphasizes code readability [source:doc_1].\nUser: Does Python support multiple programming paradigms?\nAssistant: Python supports multiple programming paradigms including procedural, object-oriented, and functional programming [source:doc_1].\nUser: Name a Python library for scientific computing.\n\nUser question:\nName a Python library for scientific computing.', 'answer': "I don't know", 'docs': [{'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.8499234914779663}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7145059108734131}, {'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.6205575466156006}], 'sources': [], 'context': "[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains"})
    You: Is Python good for web development and machine learning?
    Bot: ("Python is good for web development [source:doc_1]. I don't know about machine learning specifically.", {'question': "Conversation History:\nAssistant: Python is notable for its use of significant whitespace, which emphasizes code readability [source:doc_1].\nUser: Does Python support multiple programming paradigms?\nAssistant: Python supports multiple programming paradigms including procedural, object-oriented, and functional programming [source:doc_1].\nUser: Name a Python library for scientific computing.\nAssistant: I don't know\nUser: Is Python good for web development and machine learning?\n\nUser question:\nIs Python good for web development and machine learning?", 'answer': "Python is good for web development [source:doc_1]. I don't know about machine learning specifically.", 'docs': [{'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.8422232866287231}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7274588942527771}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6667062044143677}], 'sources': ['doc_1'], 'context': "[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: What are the main types of machine learning?
    Bot: ('The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].', {'question': "Conversation History:\nAssistant: Python supports multiple programming paradigms including procedural, object-oriented, and functional programming [source:doc_1].\nUser: Name a Python library for scientific computing.\nAssistant: I don't know\nUser: Is Python good for web development and machine learning?\nAssistant: Python is good for web development [source:doc_1]. I don't know about machine learning specifically.\nUser: What are the main types of machine learning?\n\nUser question:\nWhat are the main types of machine learning?", 'answer': 'The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.8425145149230957}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7764991521835327}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7062056660652161}], 'sources': ['doc_2'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: What is supervised learning?
    Bot: ('Supervised learning uses labeled data to train models for prediction tasks [source:doc_2].', {'question': "Conversation History:\nAssistant: I don't know\nUser: Is Python good for web development and machine learning?\nAssistant: Python is good for web development [source:doc_1]. I don't know about machine learning specifically.\nUser: What are the main types of machine learning?\nAssistant: The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].\nUser: What is supervised learning?\n\nUser question:\nWhat is supervised learning?", 'answer': 'Supervised learning uses labeled data to train models for prediction tasks [source:doc_2].', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.8230359554290771}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7614139318466187}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7267137765884399}], 'sources': ['doc_2'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: What is unsupervised learning?
    Bot: ('Unsupervised learning finds patterns in data without labels, such as clustering similar items [source:doc_2].', {'question': "Conversation History:\nAssistant: Python is good for web development [source:doc_1]. I don't know about machine learning specifically.\nUser: What are the main types of machine learning?\nAssistant: The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].\nUser: What is supervised learning?\nAssistant: Supervised learning uses labeled data to train models for prediction tasks [source:doc_2].\nUser: What is unsupervised learning?\n\nUser question:\nWhat is unsupervised learning?", 'answer': 'Unsupervised learning finds patterns in data without labels, such as clustering similar items [source:doc_2].', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.8504395484924316}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7274320125579834}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.6946704387664795}], 'sources': ['doc_2'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: What is reinforcement learning in short?
    Bot: ('Reinforcement learning trains... [source:doc_2]', {'question': 'Conversation History:\nAssistant: The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].\nUser: What is supervised learning?\nAssistant: Supervised learning uses labeled data to train models for prediction tasks [source:doc_2].\nUser: What is unsupervised learning?\nAssistant: Unsupervised learning finds patterns in data without labels, such as clustering similar items [source:doc_2].\nUser: What is reinforcement learning in short?\n\nUser question:\nWhat is reinforcement learning in short?', 'answer': 'Reinforcement learning trains... [source:doc_2]', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.8395439982414246}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.620013415813446}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.600802481174469}], 'sources': ['doc_2'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: Name popular ML libraries mentioned.
    Bot: ("I don't know.", {'question': 'Conversation History:\nAssistant: Supervised learning uses labeled data to train models for prediction tasks [source:doc_2].\nUser: What is unsupervised learning?\nAssistant: Unsupervised learning finds patterns in data without labels, such as clustering similar items [source:doc_2].\nUser: What is reinforcement learning in short?\nAssistant: Reinforcement learning trains... [source:doc_2]\nUser: Name popular ML libraries mentioned.\n\nUser question:\nName popular ML libraries mentioned.', 'answer': "I don't know.", 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.835810661315918}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6274741888046265}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.6252289414405823}], 'sources': [], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: What do frontend developers do?
    Bot: ('Frontend developers focus on user interfaces using HTML, CSS, and JavaScript [source:doc_3].', {'question': "Conversation History:\nAssistant: Unsupervised learning finds patterns in data without labels, such as clustering similar items [source:doc_2].\nUser: What is reinforcement learning in short?\nAssistant: Reinforcement learning trains... [source:doc_2]\nUser: Name popular ML libraries mentioned.\nAssistant: I don't know.\nUser: What do frontend developers do?\n\nUser question:\nWhat do frontend developers do?", 'answer': 'Frontend developers focus on user interfaces using HTML, CSS, and JavaScript [source:doc_3].', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.7490824460983276}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.7232982516288757}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.6749754548072815}], 'sources': ['doc_3'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: What do backend developers do?
    Bot: ('Backend developers handle server-side logic, databases, and APIs using languages like Python, Java, or Node.js [source:doc_3].', {'question': "Conversation History:\nAssistant: Reinforcement learning trains... [source:doc_2]\nUser: Name popular ML libraries mentioned.\nAssistant: I don't know.\nUser: What do frontend developers do?\nAssistant: Frontend developers focus on user interfaces using HTML, CSS, and JavaScript [source:doc_3].\nUser: What do backend developers do?\n\nUser question:\nWhat do backend developers do?", 'answer': 'Backend developers handle server-side logic, databases, and APIs using languages like Python, Java, or Node.js [source:doc_3].', 'docs': [{'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.773979663848877}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7014768123626709}, {'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.6763972043991089}], 'sources': ['doc_3'], 'context': "[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains"})
    You: What is a REST API?
    Bot: ('A REST API provides a way for different systems to communicate over HTTP using standard methods [source:doc_3].', {'question': "Conversation History:\nAssistant: I don't know.\nUser: What do frontend developers do?\nAssistant: Frontend developers focus on user interfaces using HTML, CSS, and JavaScript [source:doc_3].\nUser: What do backend developers do?\nAssistant: Backend developers handle server-side logic, databases, and APIs using languages like Python, Java, or Node.js [source:doc_3].\nUser: What is a REST API?\n\nUser question:\nWhat is a REST API?", 'answer': 'A REST API provides a way for different systems to communicate over HTTP using standard methods [source:doc_3].', 'docs': [{'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.8092038631439209}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7358430624008179}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6686243414878845}], 'sources': ['doc_3'], 'context': "[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: Name modern frontend frameworks.
    Bot: ('Modern frontend frameworks include React, Vue.js, and Angular [source:doc_3].', {'question': 'Conversation History:\nAssistant: Frontend developers focus on user interfaces using HTML, CSS, and JavaScript [source:doc_3].\nUser: What do backend developers do?\nAssistant: Backend developers handle server-side logic, databases, and APIs using languages like Python, Java, or Node.js [source:doc_3].\nUser: What is a REST API?\nAssistant: A REST API provides a way for different systems to communicate over HTTP using standard methods [source:doc_3].\nUser: Name modern frontend frameworks.\n\nUser question:\nName modern frontend frameworks.', 'answer': 'Modern frontend frameworks include React, Vue.js, and Angular [source:doc_3].', 'docs': [{'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.8314011693000793}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7026718854904175}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6758211851119995}], 'sources': ['doc_3'], 'context': "[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: Name example database systems.
    Bot: ('Example database systems include MySQL, PostgreSQL, and MongoDB [source:doc_1].', {'question': 'Conversation History:\nAssistant: Backend developers handle server-side logic, databases, and APIs using languages like Python, Java, or Node.js [source:doc_3].\nUser: What is a REST API?\nAssistant: A REST API provides a way for different systems to communicate over HTTP using standard methods [source:doc_3].\nUser: Name modern frontend frameworks.\nAssistant: Modern frontend frameworks include React, Vue.js, and Angular [source:doc_3].\nUser: Name example database systems.\n\nUser question:\nName example database systems.', 'answer': 'Example database systems include MySQL, PostgreSQL, and MongoDB [source:doc_1].', 'docs': [{'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.7999935150146484}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7032430171966553}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.658782958984375}], 'sources': ['doc_1'], 'context': "[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: What can Discord bots do?
    Bot: ('Discord bots can respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].', {'question': 'Conversation History:\nAssistant: A REST API provides a way for different systems to communicate over HTTP using standard methods [source:doc_3].\nUser: Name modern frontend frameworks.\nAssistant: Modern frontend frameworks include React, Vue.js, and Angular [source:doc_3].\nUser: Name example database systems.\nAssistant: Example database systems include MySQL, PostgreSQL, and MongoDB [source:doc_1].\nUser: What can Discord bots do?\n\nUser question:\nWhat can Discord bots do?', 'answer': 'Discord bots can respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.8568935394287109}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6809566617012024}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6160430312156677}], 'sources': ['doc_4'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: Which Python library is commonly used to build Discord bots?
    Bot: ('Python developers often use the discord.py library to create bots with features like slash commands [source:doc_4].', {'question': 'Conversation History:\nAssistant: Modern frontend frameworks include React, Vue.js, and Angular [source:doc_3].\nUser: Name example database systems.\nAssistant: Example database systems include MySQL, PostgreSQL, and MongoDB [source:doc_1].\nUser: What can Discord bots do?\nAssistant: Discord bots can respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].\nUser: Which Python library is commonly used to build Discord bots?\n\nUser question:\nWhich Python library is commonly used to build Discord bots?', 'answer': 'Python developers often use the discord.py library to create bots with features like slash commands [source:doc_4].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.9063236713409424}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7247673869132996}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6685607433319092}], 'sources': ['doc_4'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: How do bots authenticate?
    Bot: ('Bots require proper authentication using bot tokens and must be invited to servers [source:doc_4].', {'question': 'Conversation History:\nAssistant: Example database systems include MySQL, PostgreSQL, and MongoDB [source:doc_1].\nUser: What can Discord bots do?\nAssistant: Discord bots can respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].\nUser: Which Python library is commonly used to build Discord bots?\nAssistant: Python developers often use the discord.py library to create bots with features like slash commands [source:doc_4].\nUser: How do bots authenticate?\n\nUser question:\nHow do bots authenticate?', 'answer': 'Bots require proper authentication using bot tokens and must be invited to servers [source:doc_4].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.8999583721160889}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6775244474411011}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6173231601715088}], 'sources': ['doc_4'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: Give examples of common bot features.
    Bot: ('Common bot features include responding to messages, moderating chat, playing music, and performing various automated tasks [source:doc_4].', {'question': 'Conversation History:\nAssistant: Discord bots can respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].\nUser: Which Python library is commonly used to build Discord bots?\nAssistant: Python developers often use the discord.py library to create bots with features like slash commands [source:doc_4].\nUser: How do bots authenticate?\nAssistant: Bots require proper authentication using bot tokens and must be invited to servers [source:doc_4].\nUser: Give examples of common bot features.\n\nUser question:\nGive examples of common bot features.', 'answer': 'Common bot features include responding to messages, moderating chat, playing music, and performing various automated tasks [source:doc_4].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.9228736758232117}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7007776498794556}, {'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.6225418448448181}], 'sources': ['doc_4'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains"})
    You: Is Python compiled or interpreted?
    Bot: ('Python is an interpreted programming language [source:doc_1].', {'question': 'Conversation History:\nAssistant: Python developers often use the discord.py library to create bots with features like slash commands [source:doc_4].\nUser: How do bots authenticate?\nAssistant: Bots require proper authentication using bot tokens and must be invited to servers [source:doc_4].\nUser: Give examples of common bot features.\nAssistant: Common bot features include responding to messages, moderating chat, playing music, and performing various automated tasks [source:doc_4].\nUser: Is Python compiled or interpreted?\n\nUser question:\nIs Python compiled or interpreted?', 'answer': 'Python is an interpreted programming language [source:doc_1].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.872445821762085}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7669037580490112}, {'_id': 2, 'text': 'Web Development Technologies:\n    Web development involves creating websites and web applications using various technologies.\n    Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.\n    Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.\n    REST APIs provide a way for different systems to communicate over HTTP using standard methods.\n    Modern web frameworks like React, Vue.js, and Angular help build interactive user interfaces.\n    Database systems like PostgreSQL, MongoDB, and Redis store and manage application data efficiently.', 'source': 'doc_3', 'score': 0.6227100491523743}], 'sources': ['doc_1'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_3] Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development focuses on user interfaces using HTML, CSS, and JavaScript.     Backend development handles server-side logic, databases, and APIs using languages like Python, Java, or Node.js.     REST APIs provide a way for different systems to communicate over HTTP using standard methods.     Modern web frameworks like React, Vue.js, and Angular help build int"})
    You: What are popular frameworks for machine learning tasks?
    Bot: ('Popular frameworks for machine learning tasks include TensorFlow and scikit-learn [source:doc_2].', {'question': 'Conversation History:\nAssistant: Bots require proper authentication using bot tokens and must be invited to servers [source:doc_4].\nUser: Give examples of common bot features.\nAssistant: Common bot features include responding to messages, moderating chat, playing music, and performing various automated tasks [source:doc_4].\nUser: Is Python compiled or interpreted?\nAssistant: Python is an interpreted programming language [source:doc_1].\nUser: What are popular frameworks for machine learning tasks?\n\nUser question:\nWhat are popular frameworks for machine learning tasks?', 'answer': 'Popular frameworks for machine learning tasks include TensorFlow and scikit-learn [source:doc_2].', 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7949889898300171}, {'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.7776491641998291}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7343425750732422}], 'sources': ['doc_2'], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d"})
    You: Which databases are mentioned for storing application data?
    Bot: ("I don't know.", {'question': 'Conversation History:\nAssistant: Common bot features include responding to messages, moderating chat, playing music, and performing various automated tasks [source:doc_4].\nUser: Is Python compiled or interpreted?\nAssistant: Python is an interpreted programming language [source:doc_1].\nUser: What are popular frameworks for machine learning tasks?\nAssistant: Popular frameworks for machine learning tasks include TensorFlow and scikit-learn [source:doc_2].\nUser: Which databases are mentioned for storing application data?\n\nUser question:\nWhich databases are mentioned for storing application data?', 'answer': "I don't know.", 'docs': [{'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.7179670929908752}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7077343463897705}, {'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.7075351476669312}], 'sources': [], 'context': "[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains"})
    You: What libraries can I use for deep learning?
    Bot: ('For deep learning tasks, you can use libraries like TensorFlow and PyTorch [source:doc_2].', {'question': "Conversation History:\nAssistant: Python is an interpreted programming language [source:doc_1].\nUser: What are popular frameworks for machine learning tasks?\nAssistant: Popular frameworks for machine learning tasks include TensorFlow and scikit-learn [source:doc_2].\nUser: Which databases are mentioned for storing application data?\nAssistant: I don't know.\nUser: What libraries can I use for deep learning?\n\nUser question:\nWhat libraries can I use for deep learning?", 'answer': 'For deep learning tasks, you can use libraries like TensorFlow and PyTorch [source:doc_2].', 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.726359486579895}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.7124402523040771}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.6623196005821228}], 'sources': ['doc_2'], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})
    You: Who created Java?
    Bot: ("I don't know.", {'question': "Conversation History:\nAssistant: Popular frameworks for machine learning tasks include TensorFlow and scikit-learn [source:doc_2].\nUser: Which databases are mentioned for storing application data?\nAssistant: I don't know.\nUser: What libraries can I use for deep learning?\nAssistant: For deep learning tasks, you can use libraries like TensorFlow and PyTorch [source:doc_2].\nUser: Who created Java?\n\nUser question:\nWho created Java?", 'answer': "I don't know.", 'docs': [{'_id': 1, 'text': 'Machine Learning and Artificial Intelligence:\n    Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.\n    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.\n    Supervised learning uses labeled data to train models for prediction tasks.\n    Unsupervised learning finds patterns in data without labels, such as clustering similar items.\n    Reinforcement learning trains agents to make decisions through trial and error with rewards and penalties.\n    Popular machine learning libraries include scikit-learn, TensorFlow, PyTorch, and Keras.', 'source': 'doc_2', 'score': 0.7160383462905884}, {'_id': 0, 'text': 'Python Programming Language Overview:\n    Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.\n    It emphasizes code readability with its notable use of significant whitespace.\n    Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.\n    The language is widely used for web development, data analysis, artificial intelligence, and automation.\n    Popular Python frameworks include Django for web development, NumPy for scientific computing,\n    and TensorFlow for machine learning applications.', 'source': 'doc_1', 'score': 0.6789410710334778}, {'_id': 3, 'text': "Discord Bot Development:\n    Discord bots are applications that can interact with Discord servers automatically.\n    They can respond to messages, moderate chat, play music, and perform various automated tasks.\n    Discord bots are built using Discord's API and can be developed in multiple programming languages.\n    Python developers often use the discord.py library to create bots with features like slash commands.\n    Bots require proper authentication using bot tokens and must be invited to servers with appropriate permissions.\n    Common bot features include welcome messages, role management, music playback, and custom commands.", 'source': 'doc_4', 'score': 0.659609854221344}], 'sources': [], 'context': "[source:doc_2] Machine Learning and Artificial Intelligence:     Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically.     There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.     Supervised learning uses labeled data to train models for prediction tasks.     Unsupervised learning finds patterns in data without labels, such as clustering similar items.     Reinforcement learning trains\n\n[source:doc_1] Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasizes code readability with its notable use of significant whitespace.     Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.     The language is widely used for web development, data analysis, artificial intelligence, and automation.     Popular Python frameworks include Django for web d\n\n[source:doc_4] Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, moderate chat, play music, and perform various automated tasks.     Discord bots are built using Discord's API and can be developed in multiple programming languages.     Python developers often use the discord.py library to create bots with features like slash commands.     Bots require proper authentication using bot tokens and must be invited to se"})



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /tmp/ipython-input-3875411250.py in <cell line: 0>()
          2 
          3 while True:
    ----> 4     q = input("You: ")
          5     if q.lower() in ["exit", "quit"]: break
          6     answer = chatbot_turn(q)


    /usr/local/lib/python3.12/dist-packages/ipykernel/kernelbase.py in raw_input(self, prompt)
       1175                 "raw_input was called, but this frontend does not support input requests."
       1176             )
    -> 1177         return self._input_request(
       1178             str(prompt),
       1179             self._parent_ident["shell"],


    /usr/local/lib/python3.12/dist-packages/ipykernel/kernelbase.py in _input_request(self, prompt, ident, parent, password)
       1217             except KeyboardInterrupt:
       1218                 # re-raise KeyboardInterrupt, to truncate traceback
    -> 1219                 raise KeyboardInterrupt("Interrupted by user") from None
       1220             except Exception:
       1221                 self.log.warning("Invalid Message:", exc_info=True)


    KeyboardInterrupt: Interrupted by user


# Section 8: Testing and evaluation


In this section, we validate the performance of the RAG chatbot by running different types of queries and checking if responses are accurate, relevant, and grounded in the knowledge base.  

The evaluation process includes four parts:  

1. **Smoke Tests (Simple Queries)**  
   - Run a set of straightforward questions such as *“Who created Python?”* or *“What is machine learning?”*.  
   - Verify that the chatbot provides correct answers with proper source citations.  

2. **Edge Case Queries**  
   - Test queries outside the knowledge base (e.g., *“Who is Elon Musk?”*).  
   - Check overlapping/multi-topic queries (e.g., *“Tell me about Python and Discord together”*).  
   - Confirm that the chatbot either answers correctly from available context or says *“I don’t know”* rather than hallucinating.  

3. **Atlas vs. Fallback Search Comparison**  
   - Compare retrieval quality between **Atlas Vector Search** and manual cosine similarity fallback.  
   - Ensures system resilience if Atlas is unavailable.  

4. **Evaluation Helper Function**  
   - `evaluate_queries()` automates testing against a set of queries with expected keywords.  
   - Returns pass/fail results for each query, helping quantify accuracy.  

 **Relevance of this section:**  
Evaluation ensures the RAG pipeline is **trustworthy, robust, and production-ready**.  
By testing both normal and edge cases, as well as validating fallback mechanisms, we confirm that the chatbot behaves consistently in real-world scenarios.  



```python

# Simple Queries Test (smoke test)


test_queries = [
    "Who created Python?",
    "What is machine learning?",
    "Tell me about Discord bots"
]

print("🔎 Running simple smoke test queries...\n")
for q in test_queries:
    answer, info = chatbot_turn(q, top_k=3)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
    print(f"Sources: {info['sources']}")
    print("-" * 60)

```

    🔎 Running simple smoke test queries...
    
    Q: Who created Python?
    A: Python was created by Guido van Rossum in 1991 [source:doc_1].
    
    Sources: ['doc_1']
    ------------------------------------------------------------
    Q: What is machine learning?
    A: Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically [source:doc_2].
    
    Sources: ['doc_2']
    ------------------------------------------------------------
    Q: Tell me about Discord bots
    A: Discord bots are applications that can interact with Discord servers automatically, respond to messages, moderate chat, play music, and perform various automated tasks. They are built using Discord's API and can be developed in multiple programming languages like Python with the discord.py library for features like slash commands [source:doc_4].
    
    Sources: ['doc_4']
    ------------------------------------------------------------



```python
# Edge Cases

edge_queries = [
    "Who is Elon Musk?",         # out of scope
    "Explain quantum physics",   # not in knowledge base
    "Tell me about Python and Discord together"  # overlapping topics
]

print("⚠️ Testing edge cases...\n")
for q in edge_queries:
    answer, info = chatbot_turn(q, top_k=3)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
    print(f"Sources: {info['sources']}")
    print("-" * 60)

```

    ⚠️ Testing edge cases...
    
    Q: Who is Elon Musk?
    A: I don't know.
    
    Sources: []
    ------------------------------------------------------------
    Q: Explain quantum physics
    A: I don't know.
    
    Sources: []
    ------------------------------------------------------------
    Q: Tell me about Python and Discord together
    A: Python is commonly used for developing Discord bots, with Python developers often utilizing the discord.py library to create bots with features like slash commands [source:doc_4].
    
    Sources: ['doc_4']
    ------------------------------------------------------------



```python
# Compare Atlas vs Fallback


def compare_search(query, top_k=3):
    print(f"\n🔎 Query: {query}")

    atlas_res = mongodb_vector_search(query, top_k=top_k, debug=False)
    if atlas_res:
        print("\n✅ Atlas Vector Search Results:")
        for i, r in enumerate(atlas_res, 1):
            print(f" {i}. score={r.get('score'):.4f} | source={r.get('source')}")
            print("    preview:", r.get("text","")[:150].replace("\n"," "))
    else:
        print("\n⚠️ Atlas returned no results.")

    fb_res = fallback_search(query, top_k=top_k)
    print("\n🟡 Fallback Cosine Similarity Results:")
    for i, r in enumerate(fb_res, 1):
        print(f" {i}. score={r['score']:.4f} | source={r['source']}")
        print("    preview:", r['text'][:150].replace("\n"," "))

# Example test
compare_search("What is Python programming?", top_k=3)

```

    
    🔎 Query: What is Python programming?
    
    ✅ Atlas Vector Search Results:
     1. score=0.9127 | source=doc_1
        preview: Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasi
     2. score=0.7424 | source=doc_4
        preview: Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, mod
     3. score=0.6612 | source=doc_3
        preview: Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development
    
    🟡 Fallback Cosine Similarity Results:
     1. score=0.8254 | source=doc_1
        preview: Python Programming Language Overview:     Python is a high-level, interpreted programming language created by Guido van Rossum in 1991.     It emphasi
     2. score=0.4848 | source=doc_4
        preview: Discord Bot Development:     Discord bots are applications that can interact with Discord servers automatically.     They can respond to messages, mod
     3. score=0.3224 | source=doc_3
        preview: Web Development Technologies:     Web development involves creating websites and web applications using various technologies.     Frontend development



```python
#  Evaluation Helper


def evaluate_queries(test_cases, top_k=3):
    """
    Runs multiple queries and checks if answer contains expected keywords.
    test_cases: list of dicts {query: str, expected_keywords: [list of str]}
    """
    results = []
    for case in test_cases:
        q = case["query"]
        expected = case.get("expected_keywords", [])
        ans, info = chatbot_turn(q, top_k=top_k)

        passed = all(kw.lower() in ans.lower() for kw in expected)
        results.append({"query": q, "expected": expected, "answer": ans, "pass": passed})

    # Print results
    print("\n📊 Evaluation Results:")
    for r in results:
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        print(f"Q: {r['query']}")
        print(f"Expected keywords: {r['expected']}")
        print(f"A: {r['answer'][:200]}...")
        print("Result:", status)
        print("-" * 60)
    return results

# Example evaluation set
test_cases = [
    {"query": "Who created Python?", "expected_keywords": ["Guido", "1991"]},
    {"query": "What is machine learning?", "expected_keywords": ["data", "learn"]},
    {"query": "What are Discord bots?", "expected_keywords": ["Discord", "automated"]}
]

evaluate_queries(test_cases)

```

    
    📊 Evaluation Results:
    Q: Who created Python?
    Expected keywords: ['Guido', '1991']
    A: Python was created by Guido van Rossum in 1991 [source:doc_1]....
    Result: ✅ PASS
    ------------------------------------------------------------
    Q: What is machine learning?
    Expected keywords: ['data', 'learn']
    A: Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically. It includes supervised learning, unsupervised learning, and reinforcement learning [sou...
    Result: ✅ PASS
    ------------------------------------------------------------
    Q: What are Discord bots?
    Expected keywords: ['Discord', 'automated']
    A: Discord bots are applications that can interact with Discord servers automatically, respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4]....
    Result: ✅ PASS
    ------------------------------------------------------------





    [{'query': 'Who created Python?',
      'expected': ['Guido', '1991'],
      'answer': 'Python was created by Guido van Rossum in 1991 [source:doc_1].',
      'pass': True},
     {'query': 'What is machine learning?',
      'expected': ['data', 'learn'],
      'answer': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data automatically. It includes supervised learning, unsupervised learning, and reinforcement learning [source:doc_2].',
      'pass': True},
     {'query': 'What are Discord bots?',
      'expected': ['Discord', 'automated'],
      'answer': 'Discord bots are applications that can interact with Discord servers automatically, respond to messages, moderate chat, play music, and perform various automated tasks [source:doc_4].',
      'pass': True}]




```python

```
