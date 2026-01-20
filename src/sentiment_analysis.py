positive_words = {
    "good", "great", "excellent", "amazing", "love", "nice", "awesome",
    "perfect", "best", "wonderful", "fantastic", "positive"
}

negative_words = {
    "bad", "worst", "poor", "terrible", "awful", "hate", "boring",
    "negative", "waste", "disappointed", "problem"
}

def get_sentiment(text):
    text = str(text).lower()
    pos = sum(word in text for word in positive_words)
    neg = sum(word in text for word in negative_words)

    if pos > neg:
        return "Positive"
    elif neg > pos:
        return "Negative"
    else:
        return "Neutral"
