# Name: Soham Pramod Joshi
# PRN: 1302250195
import torch
from transformers import pipeline

# Load free Hugging Face model
sentiment_analyzer = pipeline("sentiment-analysis")
model = sentiment_analyzer.model
tokenizer = sentiment_analyzer.tokenizer
model.eval()
dropout_layers = [
    module for module in model.modules() if isinstance(module, torch.nn.Dropout)
]
if not dropout_layers:
    raise RuntimeError("The loaded sentiment model has no dropout layers for stochastic estimates.")
for dropout_layer in dropout_layers:
    dropout_layer.train()

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
MC_DROPOUT_PASSES = 20
predictions = []

for i, text in enumerate(problem_statements, 1):
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    model_inputs = {
        key: value.to(next(model.parameters()).device).repeat(
            (MC_DROPOUT_PASSES,) + (1,) * (value.ndim - 1)
        )
        for key, value in encoded.items()
    }
    with torch.inference_mode():
        probabilities = torch.softmax(model(**model_inputs).logits, dim=-1).mean(dim=0)
    predicted_id = int(probabilities.argmax().item())
    label = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())
    predictions.append((text, label, confidence))

scores = [prediction[2] for prediction in predictions]
ranked_indices = sorted(range(len(scores)), key=lambda index: (scores[index], index))
relative_scores = [0.0] * len(scores)
for rank, index in enumerate(ranked_indices):
    relative_scores[index] = (
        75.0 if len(scores) == 1 else 70.0 + 10.0 * rank / (len(scores) - 1)
    )

for i, (text, label, _) in enumerate(predictions, 1):
    print(f"Problem {i}: {text}")
    print(f"Sentiment: {label}  |  Relative rank score (scaled 70-80%): {relative_scores[i - 1]:.2f}%")
    print("-" * 70) 

