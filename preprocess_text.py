import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from nltk.stem.porter import PorterStemme

# Download required NLTK resources (Run this ONCE before calling preprocess_text)
nltk.download('stopwords')
def preprocess_text(text):
    """
    Preprocesses the given text by:
    - Converting to lowercase
    - Removing URLs, special characters, numbers, and extra spaces
    - Removing stopwords
    - Tokenizing
    - Lemmatizing words
    """
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\[.*?\]', '', text)  # Remove text inside square brackets
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words with numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)  # Tokenize text
    #Stemming
    filtered_words = [word for word in words if word not in stop_words]
    #ps=PorterStemmer()
    #Stemmed_words = [ps.stem(word) for word in filtered_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)