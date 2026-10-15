"""
ML Framework - Hugging Face Sentiment Monitor
Subject          : AI ML 1 - CCA2
Class            : S Y BCom FinTech
Assignment       : Sentiment analysis using Hugging Face Transformers
                   + Multiple problem statements

Student Name     : <YOUR FULL NAME>
PRN              : <YOUR PRN NUMBER>
Branch           : <YourFirstName>_Dev

Instructions:
1. Replace <YOUR FULL NAME> and <YOUR PRN NUMBER> below.
2. Install required packages (once):
      pip install transformers torch
   (or on Colab: !pip install transformers torch -q)
3. Commit this file to your Dev branch.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. STUDENT DETAILS  ← UPDATE THESE TWO LINES
# ============================================================
STUDENT_NAME = "<YOUR FULL NAME>"
STUDENT_PRN  = "<YOUR PRN NUMBER>"


# ============================================================
# 3. LOAD HUGGING FACE SENTIMENT MODEL
# ============================================================
print("Loading Hugging Face Sentiment Analysis model...")
print("(This may take a few seconds on first run)\n")

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of a single text"""
    result = sentiment_pipeline(text)[0]
    return {
        "text": text,
        "label": result["label"],
        "score": round(result["score"], 4)
    }


def print_result(result: dict):
    """Pretty print the result"""
    emoji = "😊" if result["label"] == "POSITIVE" else "😞"
    print(f"  Text      : {result['text']}")
    print(f"  Sentiment : {result['label']} {emoji}")
    print(f"  Confidence: {result['score']}")
    print("-" * 60)


# ============================================================
# 5. MULTIPLE PROBLEM STATEMENTS
# ============================================================

print("=" * 70)
print("  HUGGING FACE SENTIMENT MONITOR")
print(f"  Student Name : {STUDENT_NAME}")
print(f"  PRN          : {STUDENT_PRN}")
print("=" * 70)


# ----------------------------------------------------------
# PROBLEM STATEMENT 1
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PROBLEM STATEMENT 1: Customer Reviews of Digital Payment Apps")
print("Use Case: Analyse user feedback of PhonePe / GPay / Paytm")
print("=" * 70)

reviews_payment = [
    "PhonePe is very fast and easy to use. I love the cashback offers!",
    "The app keeps crashing while making payments. Very frustrating experience.",
    "Good interface but customer support is terrible and never responds.",
    "I transferred money successfully within seconds. Highly recommended!",
    "Transaction failed three times and money got deducted. Worst experience."
]

print("\nAnalyzing reviews...\n")
for review in reviews_payment:
    result = analyze_sentiment(review)
    print_result(result)


# ----------------------------------------------------------
# PROBLEM STATEMENT 2
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PROBLEM STATEMENT 2: Sentiment Analysis of Tweets about a Bank")
print("Use Case: Monitor public opinion about a bank or FinTech company")
print("=" * 70)

tweets = [
    "HDFC Bank has the best mobile banking app in India. Super smooth!",
    "Waiting in queue for 2 hours because the bank server is down again.",
    "Got my personal loan approved in just 10 minutes. Amazing service!",
    "Hidden charges everywhere. Completely disappointed with this bank.",
    "The new UPI feature is convenient and secure. Great job!"
]

print("\nAnalyzing tweets...\n")
for tweet in tweets:
    result = analyze_sentiment(tweet)
    print_result(result)


# ----------------------------------------------------------
# PROBLEM STATEMENT 3
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PROBLEM STATEMENT 3: Feedback from Loan Applicants")
print("Use Case: Understand customer satisfaction during loan process")
print("=" * 70)

loan_feedback = [
    "The loan process was quick and the interest rate is reasonable.",
    "Documents were rejected multiple times without clear reason. Very stressful.",
    "Relationship manager was helpful and guided me through every step.",
    "Too many unnecessary formalities. Wasted my entire day.",
    "Received the loan amount in my account within 24 hours. Excellent!"
]

print("\nAnalyzing loan feedback...\n")
for feedback in loan_feedback:
    result = analyze_sentiment(feedback)
    print_result(result)


# ----------------------------------------------------------
# PROBLEM STATEMENT 4
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PROBLEM STATEMENT 4: Product Reviews of Credit Card / Investment App")
print("Use Case: Analyse sentiment of product reviews for a FinTech product")
print("=" * 70)

product_reviews = [
    "This credit card gives great rewards on every spend. Loving it!",
    "Annual fees are too high and benefits are almost zero. Not worth it.",
    "The investment app is simple and perfect for beginners.",
    "App freezes every time I try to redeem points. Need urgent fix.",
    "Portfolio tracking feature is accurate and very useful."
]

print("\nAnalyzing product reviews...\n")
for review in product_reviews:
    result = analyze_sentiment(review)
    print_result(result)


# ----------------------------------------------------------
# PROBLEM STATEMENT 5
# ----------------------------------------------------------
print("\n" + "=" * 70)
print("PROBLEM STATEMENT 5: Customer Support Chat Sentiment")
print("Use Case: Detect negative sentiment in support conversations early")
print("=" * 70)

support_chats = [
    "Thank you so much for resolving my issue so quickly!",
    "I have been waiting for 3 days and still no response from your team.",
    "The agent was polite and solved my problem in one call.",
    "This is the worst customer service I have ever experienced.",
    "My query was handled professionally. Appreciate the support."
]

print("\nAnalyzing support chats...\n")
for chat in support_chats:
    result = analyze_sentiment(chat)
    print_result(result)


# ============================================================
# 6. INTERACTIVE MODE
# ============================================================
print("\n" + "=" * 70)
print("INTERACTIVE MODE – Enter your own text for sentiment analysis")
print("Type 'exit' to quit")
print("=" * 70)

while True:
    user_text = input("\nEnter text: ").strip()
    if user_text.lower() in ["exit", "quit", "q"]:
        print("\nThank you! Exiting Hugging Face Sentiment Monitor.")
        break
    if user_text:
        result = analyze_sentiment(user_text)
        print()
        print_result(result)
