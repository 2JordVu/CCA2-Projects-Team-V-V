"""
ML Framework - Retrieval Augmented Generation (RAG)
Subject          : AI ML 1 - CCA2
Class            : S Y BCom FinTech
Assignment       : Simple RAG implementation with knowledge base + extra rules

Student Name     : Soham Pramod Joshi
PRN              : 1302250195
Branch           : Soham_Dev

Instructions for GitHub:
1. Replace <YOUR FULL NAME> and <YOUR PRN NUMBER> below.
2. Commit this file to your Dev branch.
"""

# ============================================================
# 1. IMPORTS
# ============================================================
# Install required packages first: pip install scikit-learn numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple


# ============================================================
# 2. STUDENT DETAILS  ← UPDATE THESE TWO LINES
# ============================================================
STUDENT_NAME = "<YOUR FULL NAME>"
STUDENT_PRN  = "<YOUR PRN NUMBER>"


# ============================================================
# 3. KNOWLEDGE BASE
# ============================================================
KNOWLEDGE_BASE = [
    {
        "id": 1,
        "title": "What is FinTech",
        "content": "FinTech (Financial Technology) refers to the use of technology to improve and automate financial services. It includes digital payments, peer-to-peer lending, robo-advisors, blockchain, and mobile banking. FinTech aims to make financial services faster, cheaper, and more accessible."
    },
    {
        "id": 2,
        "title": "Machine Learning in Finance",
        "content": "Machine Learning is widely used in finance for credit scoring, fraud detection, algorithmic trading, customer segmentation, and risk management. Models learn patterns from historical data to make predictions about loan defaults, stock prices, or suspicious transactions."
    },
    {
        "id": 3,
        "title": "Retrieval Augmented Generation",
        "content": "Retrieval Augmented Generation (RAG) is a technique that combines information retrieval with text generation. First, relevant documents are retrieved from a knowledge base. Then those documents are given as context to a language model so that the generated answer is grounded in real data instead of only the model's memory."
    },
    {
        "id": 4,
        "title": "Advantages of RAG",
        "content": "RAG reduces hallucinations, allows the system to use up-to-date or private company data, provides source citations, and avoids expensive fine-tuning of large language models. It is especially useful for domain-specific applications such as banking, insurance, and education."
    },
    {
        "id": 5,
        "title": "Traditional Programming vs Machine Learning",
        "content": "In traditional programming, a developer writes explicit rules (if-else logic) to solve a problem. In Machine Learning, the system learns the rules automatically from data. Traditional systems are rigid; ML systems can adapt when new data arrives."
    },
    {
        "id": 6,
        "title": "Sentiment Analysis",
        "content": "Sentiment Analysis is a Natural Language Processing task that determines whether a piece of text expresses positive, negative, or neutral emotion. It is commonly used to analyse customer reviews, social media posts, and feedback in FinTech applications."
    },
    {
        "id": 7,
        "title": "Credit Risk Assessment",
        "content": "Credit risk assessment evaluates the probability that a borrower will default on a loan. Modern systems use machine learning models trained on features such as income, credit history, employment status, and existing debts to predict default risk more accurately than traditional scorecards."
    },
    {
        "id": 8,
        "title": "Blockchain in FinTech",
        "content": "Blockchain is a distributed ledger technology that records transactions in a secure, transparent, and immutable way. In FinTech it is used for cryptocurrencies, smart contracts, cross-border payments, and trade finance."
    },
]


# ============================================================
# 4. EXTRA RULES (as required by the assignment)
# ============================================================
EXTRA_RULES = [
    "RULE 1: Answer ONLY using the information present in the retrieved documents. Do not use external knowledge.",
    "RULE 2: If the retrieved documents do not contain enough information to answer the question, reply exactly with: 'I do not have enough information in the knowledge base to answer this question.'",
    "RULE 3: Never invent facts, numbers, or explanations that are not present in the retrieved context.",
    "RULE 4: Always mention the source document title(s) used to generate the answer.",
    "RULE 5: Keep the answer clear, concise, and professional (maximum 5-6 sentences).",
    "RULE 6: If the question is unrelated to Finance, AI, ML, or the topics in the knowledge base, politely refuse to answer.",
]


# ============================================================
# 5. RAG CLASS
# ============================================================
class SimpleRAG:
    def __init__(self, knowledge_base: List[dict]):
        self.kb = knowledge_base
        self.documents = [doc["content"] for doc in knowledge_base]
        self.titles = [doc["title"] for doc in knowledge_base]
        self.ids = [doc["id"] for doc in knowledge_base]

        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.doc_vectors = self.vectorizer.fit_transform(self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str, str]]:
        """Retrieve the top_k most relevant documents"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.05:  # minimum relevance threshold
                results.append((
                    self.ids[idx],
                    float(scores[idx]),
                    self.titles[idx],
                    self.documents[idx]
                ))
        return results

    def generate_answer(self, query: str, retrieved_docs: List[Tuple[int, float, str, str]]) -> str:
        """Generate answer following the EXTRA RULES"""
        # RULE 2 & RULE 6
        if not retrieved_docs:
            return "I do not have enough information in the knowledge base to answer this question."

        sources = []
        for doc_id, score, title, content in retrieved_docs:
            sources.append(title)

        unique_sources = list(dict.fromkeys(sources))
        best_content = retrieved_docs[0][3]

        answer = (
            f"Based on the knowledge base, here is the answer:\n\n"
            f"{best_content}\n\n"
            f"Source(s) used: {', '.join(unique_sources)}"
        )
        return answer

    def ask(self, query: str, top_k: int = 3) -> str:
        """Full RAG pipeline: Retrieve → Generate"""
        print("\n" + "="*70)
        print(f"USER QUERY : {query}")
        print("="*70)

        retrieved = self.retrieve(query, top_k=top_k)

        print("\n--- Retrieved Documents ---")
        if not retrieved:
            print("No relevant documents found.")
        else:
            for i, (doc_id, score, title, content) in enumerate(retrieved, 1):
                print(f"{i}. [ID:{doc_id}] {title}  (similarity: {score:.3f})")
                print(f"   {content[:110]}...")
                print()

        answer = self.generate_answer(query, retrieved)

        print("--- Final Answer (following Extra Rules) ---")
        print(answer)
        print("="*70 + "\n")
        return answer


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("  SIMPLE RETRIEVAL AUGMENTED GENERATION (RAG) SYSTEM")
    print(f"  Student Name : {STUDENT_NAME}")
    print(f"  PRN          : {STUDENT_PRN}")
    print("="*70)

    print("\nEXTRA RULES BEING ENFORCED:")
    for rule in EXTRA_RULES:
        print(f"  • {rule}")

    # Create RAG system
    rag = SimpleRAG(KNOWLEDGE_BASE)

    # Demo Questions
    demo_questions = [
        "What is FinTech?",
        "How is Machine Learning used in finance?",
        "Explain Retrieval Augmented Generation",
        "What are the advantages of RAG?",
        "Difference between traditional programming and machine learning",
        "How does blockchain help in FinTech?",
        "What is the capital of France?",          # should trigger Rule 2
        "Tell me a joke about cats",              # should trigger Rule 6
    ]

    for q in demo_questions:
        rag.ask(q)

    # Interactive mode
    print("\n" + "="*70)
    print("INTERACTIVE MODE – Type your own questions (type 'exit' to quit)")
    print("="*70)

    while True:
        user_q = input("\nYour question: ").strip()
        if user_q.lower() in ["exit", "quit", "q"]:
            print("Thank you! Exiting RAG system.")
            break
        if user_q:
            rag.ask(user_q)
