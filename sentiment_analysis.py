import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

# Load Dataset
df = pd.read_csv("reviews.csv")

# Function to classify sentiment
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["Review"].apply(get_sentiment)

# Display Results
print(df)

# -------------------------------
# Graph 1: Sentiment Count
# -------------------------------
plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Sentiment",
              order=["Positive","Neutral","Negative"],
              palette="Set2")
plt.title("Sentiment Count")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# -------------------------------
# Graph 2: Sentiment Percentage
# -------------------------------
plt.figure(figsize=(6,6))
df["Sentiment"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["green","gold","red"]
)
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.show()

# -------------------------------
# Graph 3: Word Cloud
# -------------------------------
text = " ".join(df["Review"])
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.show()

# -------------------------------
# Graph 4: Sentiment Score Distribution
# -------------------------------
df["Polarity"] = df["Review"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

plt.figure(figsize=(8,5))
sns.histplot(df["Polarity"], bins=10, kde=True, color="skyblue")
plt.title("Sentiment Score Distribution")
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Graph 5: Review Length Distribution
# -------------------------------
df["Review_Length"] = df["Review"].apply(len)

plt.figure(figsize=(8,5))
sns.histplot(df["Review_Length"], bins=10, kde=True, color="purple")
plt.title("Review Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.show()

print("\nSentiment Analysis Completed Successfully!")