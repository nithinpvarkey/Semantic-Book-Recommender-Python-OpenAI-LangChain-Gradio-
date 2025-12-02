# Semantic-Book-Recommender-Python-OpenAI-LangChain-Gradio-


A semantic book recommendation system that combines vector embeddings, LangChain, and LLMs to provide intelligent book recommendations based on natural language descriptions.

## 🎯 Project Overview

This project implements an AI-powered book recommendation engine that:
- Uses semantic search to find books similar to user queries
- Analyzes emotional tones (joy, surprise, anger, fear, sadness) in book descriptions
- Filters recommendations by book categories
- Provides an interactive Gradio-based dashboard for easy access
- Leverages OpenAI's LLMs and Hugging Face embeddings for intelligent matching

### Key Features

- **Semantic Search**: Vector-based similarity search using Chroma vector database
- **Emotional Classification**: Analyzes emotional content in book descriptions
- **Multi-filter Recommendations**: Filter by category and emotional tone
- **Interactive Dashboard**: User-friendly Gradio interface for real-time recommendations
- **LLM Integration**: Uses LangChain to orchestrate LLM operations with OpenAI
- **Production-Ready**: Includes data cleaning, exploration, and sentiment analysis pipelines

## 🛠️ Tech Stack

- **Python 3.10+**
- **LangChain**: LLM framework for chains and integrations
- **OpenAI**: Language models for understanding and generation
- **Hugging Face Transformers**: State-of-the-art NLP models
- **Chroma**: Vector database for semantic search
- **Sentence Transformers**: Generate semantic embeddings (all-MiniLM-L6-v2)
- **Gradio**: Build interactive web interfaces
- **Pandas**: Data manipulation and analysis
- **PyTorch**: Deep learning backend

## 📊 Dataset

The project uses a curated dataset of **5,197 books** with:
- Book metadata (ISBN, title, authors, categories)
- Full descriptions (cleaned and filtered for quality)
- Ratings and rating counts
- Emotional analysis labels (joy, surprise, anger, fear, sadness)
- Book thumbnails/cover images

## 📁 Project Structure

Semantic-Book-Recommender/
├── dashboard.py # Production Gradio dashboard application
├── requirements.txt # Python dependencies
├── books_cleaned.csv # Processed book dataset (5,197 records)
├── books_with_emotions.csv # Books with emotional classifications
├── explore_data.ipynb # Data exploration and analysis
├── text_classification.ipynb # Text classification workflows
├── sentiment_analysis.ipynb # Emotion and sentiment analysis
├── vector_search.ipynb # Vector embedding and search exploration
└── README.md # This file



The dashboard will start a local web server (default: http://127.0.0.1:7860) with:
- **Text Input**: Enter a description of the book you're looking for
- **Category Filter**: Select book categories (Fiction, Mystery, Self-Help, etc.)
- **Tone Filter**: Choose emotional tones (Happy, Surprising, Angry, Suspenseful, Sad)
- **Output**: Gallery of recommended books with covers, titles, and authors

### Example Query


Returns book recommendations matching the semantic query with the selected emotional tone.

### Jupyter Notebooks

The project includes several exploration notebooks:

1. **explore_data.ipynb**
   - Load and explore the book dataset
   - Data quality analysis
   - Missing value handling
   - Statistical summaries

2. **text_classification.ipynb**
   - Text classification pipelines
   - Category classification approaches
   - Feature extraction techniques

3. **sentiment_analysis.ipynb**
   - Emotion classification (joy, surprise, anger, fear, sadness)
   - Sentiment analysis workflows
   - Emotion extraction from book descriptions

4. **vector_search.ipynb**
   - Vector embedding generation
   - Similarity search implementation
   - Semantic matching exploration

## 🔧 Architecture

### Recommendation Pipeline

User Query
↓
[Text Processing]
↓
[Generate Embeddings] - Sentence Transformers (all-MiniLM-L6-v2)
↓
[Vector Search] - Chroma Vector Database
↓
[Apply Filters] - Category & Emotional Tone
↓
[Sort & Format] - With book metadata and covers
↓
Recommended Books (Gallery View)


### Key Components

1. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
   - Fast, efficient semantic embeddings
   - 384-dimensional vectors
   - Optimized for semantic search

2. **Vector Database**: Chroma
   - In-memory vector storage
   - Similarity search with configurable k (initial_top_k=50, final_top_k=16)
   - Fast retrieval for real-time recommendations

3. **Filtering System**
   - Category-based filtering
   - Emotional tone ranking (sorts by emotion scores)
   - Fallback to top-k results if filters limit output

4. **UI Layer**: Gradio
   - 8-column gallery display
   - 2-row layout for book covers
   - Real-time interactive recommendations

## 📈 Data Processing

The cleaned dataset (`books_cleaned.csv`) includes:
- **5,197 records** with complete descriptions (25+ words minimum)
- **13 columns**: isbn13, isbn10, title, authors, categories, thumbnail, description, published_year, average_rating, num_pages, ratings_count, title_and_subtitle, tagged_description
- **Emotion scores**: joy, surprise, anger, fear, sadness
- **Quality filters**: Removed entries with missing descriptions or metadata

## 🔍 How Recommendations Work

1. **Semantic Understanding**: Converts user queries to  embedding vectors
2. **Similarity Matching**: Finds most similar book descriptions using cosine similarity
3. **Ranking**: Retrieves top 50 candidates, then:
   - Filters by selected category (if specified)
   - Sorts by emotion score (if emotional tone selected)
   - Returns top 16 recommendations
4. **Display**: Shows book covers, titles, authors, and truncated descriptions


## 🛠️ Configuration

### Dashboard Settings (in `dashboard.py`)

Vector search parameters
initial_top_k = 50 # Initial retrieval candidates
final_top_k = 16 # Final recommendation count
chunk_size = 800 # Text chunk size for processing
chunk_overlap = 0 # Overlap between chunks

Model settings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
device = 'cpu' # Change to 'cuda' for GPU
normalize_embeddings = True



## 📦 Dependencies

### Core ML & NLP
- `transformers`: Hugging Face transformer models
- `sentence-transformers`: Semantic embedding generation
- `langchain-community`: LangChain integrations
- `langchain-openai`: OpenAI integration for LangChain
- `langchain-chroma`: Chroma vector database integration
- `torch`: PyTorch deep learning backend

### Data & Visualization
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `seaborn`: Statistical visualization

### UI & Interaction
- `gradio`: Interactive web interfaces
- `jupyter`: Notebook environment
- `notebook`: Jupyter notebook support
- `ipywidgets`: Interactive widgets

### Development & Configuration
- `python-dotenv`: Environment variable management
- `kagglehub`: Dataset downloading

See `requirements.txt` for full version specifications.

