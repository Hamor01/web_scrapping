import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

def load_custom_stopwords(stopwords_path):
    with open(stopwords_path, "r") as stop_words_file:
        custom_stop_words = stop_words_file.read().splitlines()
    return set(stopwords.words('english')).union(custom_stop_words)

def clean_text(text, stop_words):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

def analyze_sentiment(page, stop_words):
    response = requests.get(page)
    if response.status_code != 200:
        print(f"Error! - The website was not scrapped correctly: {page}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [headline.get_text() for headline in soup.find_all('h2')]
    cleaned_headlines = [clean_text(headline, stop_words) for headline in headlines]

    return cleaned_headlines

def generate_wordcloud(cleaned_data):
    bigrams = []
    for headline in cleaned_data:
        words = headline.split()
        bigrams.extend([f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)])

    total_words = ' '.join(cleaned_data) + ' ' + ' '.join(bigrams)

    wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(total_words)

    return wordcloud

# URLs
link_one = "https://www.travelchannel.com/interests/holidays/photos/christmas-in-nyc"
link_two = "https://theplanetd.com/new-york-city-christmas/"

# Path to custom stopwords file
stopwords_path = r"C:\Users\Hamor\Desktop\DIGITAL_WORK\ANA_PROJECTS\assignments\stopwords_en.txt"

# Load custom stopwords
stop_words = load_custom_stopwords(stopwords_path)

# Analyze sentiment for each page
cleaned_headlines_1 = analyze_sentiment(link_one, stop_words)
cleaned_headlines_2 = analyze_sentiment(link_two, stop_words)

# Combine the words and bigrams from both pages
all_cleaned_headlines = cleaned_headlines_1 + cleaned_headlines_2

# Generate word cloud for the combined data
wordcloud_combined = generate_wordcloud(all_cleaned_headlines)

# Calculate overall sentiment for the combined data
overall_sentiment_combined = TextBlob(' '.join(all_cleaned_headlines)).sentiment.polarity

# Print the sentiment of the combined data
print(f"\nOverall Sentiment of the Combined Data: {overall_sentiment_combined:.2f} (Positive)" if overall_sentiment_combined > 0 else "Overall Sentiment of the Combined Data: 0.00 (Neutral)")

# Display the word cloud for the combined data
plt.imshow(wordcloud_combined, interpolation='bilinear')
plt.axis('off')
plt.show()
