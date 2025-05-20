# imdb-sentiment-analysis-nlp
A Natural Language Processing (NLP) project for sentiment analysis of IMDB movie reviews using deep learning techniques. This project involves data preprocessing, model training with LSTM-based architectures, evaluation, and a Streamlit-powered web app for real-time sentiment prediction.

Features
--------
- **Advanced Text Preprocessing**: Cleans and normalizes text with specialized handling for negations and context-specific terms
- **Deep Learning Model**: Utilizes Bidirectional LSTM architecture for capturing sequential dependencies in text
- **Nuanced Sentiment Scale**: Rates reviews on a 0-10 scale rather than binary classification
- **Interactive Web Interface**: User-friendly Streamlit app for real-time sentiment analysis
- **High Accuracy**: Achieves ~88% accuracy on IMDB review dataset

Technology Stack
----------------
- Python: Core programming language
- TensorFlow/Keras: Deep learning framework for model development
- NLTK: Natural language processing for text preprocessing
- Streamlit: Web application framework for the user interface
- Pandas/NumPy: Data manipulation and numerical operations

Project Structure
-----------------
MovieMood/
├── app/
│   ├── main.py                # Streamlit web application
│   └── assets/
│       └── style.css          # Custom styling for the app
├── models/
│   ├── sentiment_classifier.h5  # Trained sentiment analysis model
│   └── tokenizer.json         # Text tokenizer for preprocessing
├── notebooks/
│   └── model_training.ipynb   # Jupyter notebook with model development
├── data/
│   ├── raw/                   # Original IMDB dataset
└── ReadMe.md                  # Project documentation

Getting Started
---------------
### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- NLTK with stopwords corpus
- Streamlit

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/MovieMood.git
cd MovieMood

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('stopwords')"

# Run the Streamlit app
streamlit run app/main.py
```

How It Works
------------
1. **Text Preprocessing**: Reviews undergo cleaning, stopword removal, and specialized handling for negations
2. **Tokenization**: Processed text is converted to sequences of tokens
3. **Neural Processing**: Bidirectional LSTM network analyzes the sequences
4. **Sentiment Prediction**: Model outputs a sentiment score on a 0-10 scale
5. **Classification**: Scores above 5 are considered positive, below 5 negative

Model Performance
-----------------
| Metric     | Score |
|------------|-------|
| Accuracy   | 88%   |
| Precision  | 0.87  |
| Recall     | 0.89  |
| F1 Score   | 0.88  |

Future Improvements
-------------------
- Implement aspect-based sentiment analysis to identify specific movie elements (acting, plot, visuals)
- Add support for multiple languages
- Integrate with movie databases for contextual analysis
- Deploy as a public API for third-party applications

References
----------
- IMDB Dataset of 50K Movie Reviews
- "Attention Is All You Need" - Vaswani et al.
- "Effective Approaches to Attention-based Neural Machine Translation" - Luong et al.

Author
------
Your Name - [https://github.com/Harshajevs](https://github.com/Harshajevs)

License
-------
This project is licensed under the MIT License - see the LICENSE file for details.
