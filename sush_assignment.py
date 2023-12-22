import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


class WebPageAnalyzer:
    def __init__(self, page_url, stop_words_path):
        self.page_url = page_url
        self.stop_words_path = stop_words_path

    def fetch_page_content(self):
        response = requests.get(self.page_url)
        if response.status_code != 200:
            print(f"Error! - The website was not scrapped correctly: {self.page_url}")
            return None
        return response.text

    def load_stopwords(self):
        stop_words = set(stopwords.words('english'))
        with open(self.stop_words_path, "r") as stop_words_file:
            custom_stop_words = stop_words_file.read().splitlines()
            stop_words.update(custom_stop_words)
        return stop_words

    def process_headlines(self, headlines, stop_words):
        if headlines is None:
            return []

        def final_text(text):
            return ' '.join([word for word in text.split() if word.lower() not in stop_words])

        return [final_text(headline.get_text()) for headline in headlines]

    def analyze_sentiment(self, final_headline):
        s_scores = [TextBlob(headline).sentiment.polarity for headline in final_headline]
        return s_scores

    def display_sentiment_scores(self, final_headline, s_scores):
        for i, (headline, score) in enumerate(zip(final_headline, s_scores), start=1):
            print(f"Sentiment Score for Headline {i}: {score:.2f}\n{headline}\n")

    def display_high_low_sentiment(self, final_headline, s_scores):
        print("\nThese are the headlines with the highest sentiment:")
        for i, score in sorted(enumerate(s_scores), key=lambda x: x[1], reverse=True)[:3]:
            print(f"Headline {i + 1}: {final_headline[i]} (Sentiment Score: {score:.2f})")

        print("\nThese are the headlines with the lowest sentiment:")
        for i, score in sorted(enumerate(s_scores), key=lambda x: x[1])[:3]:
            print(f"Headline {i + 1}: {final_headline[i]} (Sentiment Score: {score:.2f})")

    def classify_sentiment(self, s_scores):
        return ['Positive' if score > 0 else 'Negative' if score < 0 else 'Neutral' for score in s_scores]

    def extract_bigrams(self, final_headline):
        bigrams = []
        for headline in final_headline:
            words = headline.split()
            bigrams.extend([f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)])
        return bigrams

    def generate_wordcloud(self, total_words):
        return WordCloud(width=800, height=400, max_words=100, background_color='white').generate(total_words)


# Scrapped links
link_one = "https://www.travelchannel.com/interests/holidays/photos/christmas-in-nyc"
link_two = "https://theplanetd.com/new-york-city-christmas/"

# Initialize analyzers
analyzer_one = WebPageAnalyzer(link_one,
                               "C:\\Users\\Hamor\\Desktop\\DIGITAL_WORK\\ANA_PROJECTS\\assignments\\stopwords_en.txt")
analyzer_two = WebPageAnalyzer(link_two,
                               "C:\\Users\\Hamor\\Desktop\\DIGITAL_WORK\\ANA_PROJECTS\\assignments\\stopwords_en.txt")

# Processing both web pages
content_1 = analyzer_one.fetch_page_content()
if content_1 is not None:
    headlines_1 = BeautifulSoup(content_1, 'html.parser').find_all('h2')
    s_scores_1 = analyzer_one.analyze_sentiment(
        analyzer_one.process_headlines(headlines_1, analyzer_one.load_stopwords()))
    analyzer_one.display_sentiment_scores(analyzer_one.process_headlines(headlines_1, analyzer_one.load_stopwords()),
                                          s_scores_1)
    analyzer_one.display_high_low_sentiment(analyzer_one.process_headlines(headlines_1, analyzer_one.load_stopwords()),
                                            s_scores_1)

content_2 = analyzer_two.fetch_page_content()
if content_2 is not None:
    headlines_2 = BeautifulSoup(content_2, 'html.parser').find_all('h2')
    s_scores_2 = analyzer_two.analyze_sentiment(
        analyzer_two.process_headlines(headlines_2, analyzer_two.load_stopwords()))
    analyzer_two.display_sentiment_scores(analyzer_two.process_headlines(headlines_2, analyzer_two.load_stopwords()),
                                          s_scores_2)
    analyzer_two.display_high_low_sentiment(analyzer_two.process_headlines(headlines_2, analyzer_two.load_stopwords()),
                                            s_scores_2)

# Combine sentiment labels from both pages
all_sentiment_labels = analyzer_one.classify_sentiment(s_scores_1) + analyzer_two.classify_sentiment(s_scores_2)

# Combining the words and bigrams from both pages
total = ' '.join(analyzer_one.process_headlines(headlines_1, analyzer_one.load_stopwords())) + ' ' + ' '.join(
    analyzer_two.process_headlines(headlines_2, analyzer_two.load_stopwords()))

# Word cloud and sentiment for the final combined data
w_c_combined = analyzer_one.generate_wordcloud(total)
s_combined = TextBlob(total).sentiment.polarity

# Print the sentiment of the combined data
print(
    f"\nOverall Sentiment of the Combined Data: {s_combined:.2f} (Positive)" if s_combined > 0 else "Overall Sentiment of the Combined Data: 0.00 (Neutral)")

# One final word cloud for both
plt.imshow(w_c_combined, interpolation='bilinear')
plt.axis('off')
plt.show()
