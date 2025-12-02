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

## 🚀 Performance Considerations

- **Embedding Generation**: ~50-100ms per query (CPU with all-MiniLM-L6-v2)
- **Vector Search**: <10ms for similarity lookup
- **Total Latency**: ~100-150ms for full recommendation pipeline
- **Memory**: ~2GB for complete dataset with vectors loaded

### Optimization Tips

- Use GPU acceleration (change `device='cuda'`) for faster embeddings
- Batch process queries for higher throughput
- Adjust `initial_top_k` to balance speed vs. recall
- Use `chunk_overlap=0` for faster processing

## 🔄 Workflow Pipeline

### Data Preparation
1. Load raw book dataset
2. Clean and validate entries
3. Remove records with missing critical fields
4. Generate semantic embeddings
5. Store in vector database

### Model Training/Loading
1. Load pre-trained sentence transformer (all-MiniLM-L6-v2)
2. Initialize Chroma vector database
3. Load cleaned book metadata
4. Index all book descriptions

### Recommendation Generation
1. Accept user query + filters
2. Generate query embedding
3. Search vector database (k=50)
4. Filter by category if specified
5. Sort by emotion score if specified
6. Return top-k recommendations

## 📝 Notebooks Guide

### explore_data.ipynb
- Downloads book dataset from Kaggle
- Performs exploratory data analysis
- Shows missing value patterns
- Generates statistical summaries
- Visualizes data distributions

### text_classification.ipynb
- Implements text classification pipelines
- Categorizes books by genre/type
- Demonstrates classification techniques
- Evaluates model performance

### sentiment_analysis.ipynb
- Analyzes emotional content in descriptions
- Generates emotion scores (joy, surprise, anger, fear, sadness)
- Creates emotions CSV with labels
- Explores emotion distributions across genres

### vector_search.ipynb
- Generates semantic embeddings
- Demonstrates vector similarity search
- Shows nearest neighbor retrieval
- Explores embedding space properties

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Additional emotion detection models
- Multi-language support
- Advanced filtering options
- Performance optimization
- UI/UX improvements
- Additional datasets

## 📄 License

This project is open source and available under the MIT License.

## 🙋 Support & Issues

For questions, issues, or suggestions:
1. Check existing GitHub issues
2. Review notebook examples
3. Refer to documentation
4. Submit new issues with detailed descriptions

## 🎓 Learning Resources

This project demonstrates:
- Semantic search with embeddings
- Vector database integration
- LangChain orchestration
- LLM integration with OpenAI
- Production UI with Gradio
- End-to-end ML application development

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Chroma Vector Database](https://www.trychroma.com/)
- [Gradio Documentation](https://gradio.app/)
- [Sentence Transformers](https://www.sbert.net/)

---

**Project Status**: Active Development

**Last Updated**: December 2024

**Maintainer**: [nithinpvarkey](https://github.com/nithinpvarkey)
