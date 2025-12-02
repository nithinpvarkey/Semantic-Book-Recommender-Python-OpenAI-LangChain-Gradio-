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



