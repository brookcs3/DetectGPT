"""
This version uses Groq-hosted LLaMA 3 models for fast perplexity estimation and AI detection.

Originally based on Hugging Face's perplexity example:
https://huggingface.co/docs/transformers/perplexity

Converted for Groq API integration by Cameron Brooks.
"""

from model import GroqLLaMA3 as GroqPPL

# initialize the model
model = GroqPPL()

sentence = "your text here"

score, diff, std = model.getScore(sentence)
print(f"Score: {score:.3f} | Diff: {diff:.3f} | Std Dev: {std:.3f}")
