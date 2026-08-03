import nltk
import string
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
# Download required NLTK resources inside code
nltk.download('stopwords')
nltk.download('punkt')

def text_cleaning(user_text):
    text = user_text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    filtered_text = " ".join([word for word in text.split() if word not in stop_words])
    print("Processed Text:", filtered_text)

def word_frequency(user_text):
    w = user_text.lower().split()
    f = Counter(w)
    print("Word Frequencies:", f)
    print("Most Common Terms:", f.most_common(3))

def sentence_segmentation(user_text):
    sentences = nltk.sent_tokenize(user_text)
    word_counts = [len(word_tokenize(s)) for s in sentences]
    max_sentence = sentences[word_counts.index(max(word_counts))]
    min_sentence = sentences[word_counts.index(min(word_counts))]
    print("Sentence with Max Words:", max_sentence)
    print("Sentence with Min Words:", min_sentence)

def bag_of_words(corpus):
    v = CountVectorizer()
    X = v.fit_transform(corpus)
    df = pd.DataFrame(X.toarray(), columns = v.get_feature_names_out())
    print(df)

def lexical_analysis(user_text):
    t = word_tokenize(user_text)
    unigrams = list(ngrams(t, 1))
    bigrams = list(ngrams(t, 2))
    trigrams = list(ngrams(t, 3))
    print("Unigrams:", unigrams)
    print("Bigrams:", bigrams)
    print("Trigrams:", trigrams)

# Menu system
def main():
    print("\nText Processing Toolkit")
    print("Choose an option:")
    print("1. Text Cleaning")
    print("2. Word Frequency Analysis")
    print("3. Sentence Segmentation & Word Count")
    print("4. Bag-of-Words Model")
    print("5. Lexical Analysis (Unigrams, Bigrams, Trigrams)")
    choice = input("Enter choice (1-10): ")

    if choice in ["1","2","3","5","6","9","10"]:
        user_text = input("Enter your text: ")
        if choice == "1": text_cleaning(user_text)
        elif choice == "2": word_frequency(user_text)
        elif choice == "3": sentence_segmentation(user_text)
        elif choice == "5": lexical_analysis(user_text)

    elif choice == "4":
        print("Enter multiple documents (type 'END' to finish):")
        corpus = []
        while True:
            doc = input()
            if doc.strip().upper() == "END":
                break
            corpus.append(doc)
        bag_of_words(corpus)

    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()
