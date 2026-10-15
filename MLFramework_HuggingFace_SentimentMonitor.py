# Name: Your Full Name
# PRN: Your PRN Number
from transformers import pipeline
# Load free Hugging Face model
sentiment_analyzer = pipeline("sentiment-analysis")
print("===== Hugging Face Sentiment Monitor =====\n")
# ========== PROBLEM STATEMENTS (more than basic ones) ==========
problem_statements = [
    # 1. Product review
    "I absolutely love this new smartphone, the camera is amazing!",
    
    # 2. Customer service complaint
    "The customer support was terrible and they never resolved my issue.",
    
    # 3. Social media post
    "Just finished the best concert of my life! What a night!",
    
    # 4. College / education feedback
    "The AI & ML classes at MIT-WPU are very practical and helpful for FinTech students.",
    
    # 5. Movie / entertainment review
    "The movie was so boring that I left the theatre halfway.",
    
    # 6. Banking / FinTech related
    "The new mobile banking app is smooth and secure. Highly recommended!",
    
    # 7. Neutral feedback
    "The delivery arrived on time but the packaging was average.",
    
    # 8. Food / restaurant
    "Worst pizza I have ever tasted. Cold and tasteless."
]
for i, text in enumerate(problem_statements, 1):
    result = sentiment_analyzer(text)[0]
    print(f"Problem {i}: {text}")
    print(f"→ Sentiment: {result['label']}  |  Confidence: {result['score']:.2%}")
    print("-" * 70) 
