from transformers import AutoTokenizer, AutoModelForSequenceClassification  # HuggingFace Transformers components
import torch  # PyTorch for tensor operations and model execution
from typing import Tuple  # Type hinting for function signatures

# Determine computation device: GPU if available, else CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load FinBERT tokenizer and classification model for financial sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

# Define sentiment labels corresponding to model outputs
labels = ["positive", "negative", "neutral"]


def estimate_sentiment(news: list) -> Tuple[torch.Tensor, str]:
    """
    Estimate overall sentiment of a list of news headlines using FinBERT.

    Parameters:
    - news (list of str): List of textual headlines or sentences.

    Returns:
    - probability (torch.Tensor): Confidence score of the predicted sentiment.
    - sentiment (str)         : One of 'positive', 'negative', or 'neutral'.
    """
    # If no news provided, default to neutral with 0 probability
    if not news:
        return torch.tensor(0.0), labels[-1]

    # Tokenize the list of headlines into model-ready input tensors
    tokens = tokenizer(
        news,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Run the model to get raw logits for each label per headline
    outputs = model(
        tokens["input_ids"],
        attention_mask=tokens["attention_mask"]
    )["logits"]  # Shape: (batch_size, num_labels)

    # Sum logits across all headlines to get an aggregated score
    summed_logits = torch.sum(outputs, dim=0)

    # Convert summed logits to probabilities with softmax across label dimension
    probs = torch.nn.functional.softmax(summed_logits, dim=-1)

    # Identify the predicted label index and its probability
    idx = torch.argmax(probs)
    probability = probs[idx]
    sentiment = labels[idx]

    return probability, sentiment


if __name__ == "__main__":
    # Example usage: analyze two sample headlines
    example_news = [
        'markets responded negatively to the news!',
        'traders were displeased!'
    ]
    prob, sentiment = estimate_sentiment(example_news)
    print(f"Predicted sentiment: {sentiment} (confidence {prob:.4f})")
    # Check if CUDA GPU is available for acceleration
    print(f"CUDA available: {torch.cuda.is_available()}")
