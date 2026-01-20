positive_words = {
    "good", "great", "excellent", "amazing", "awesome", "fantastic",
    "love", "loved", "lovely", "nice", "perfect", "best", "wonderful",
    "satisfied", "happy", "pleased", "delightful", "positive",
    "recommend", "worth", "value", "reliable", "smooth", "fast",
    "comfortable", "beautiful", "impressive", "brilliant"
}

negative_words = {
    "bad", "worst", "poor", "terrible", "awful", "hate", "hated",
    "boring", "disappointed", "waste", "problem", "negative",
    "slow", "delay", "damaged", "broken", "useless", "cheap",
    "defective", "refund", "return", "complaint", "unhappy",
    "frustrating", "annoying", "pathetic", "failure"
}

def get_sentiment(text):
    text = str(text).lower()

    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)

    if pos_score > neg_score:
        return "Positive"
    elif neg_score > pos_score:
        return "Negative"
    else:
        return "Neutral"

