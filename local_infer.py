"""
This version uses Groq-hosted LLaMA 3 models for fast perplexity estimation and AI detection.
Originally based on Hugging Face's perplexity example:
https://huggingface.co/docs/transformers/perplexity

Converted for Groq API integration by Cameron Brooks.
"""

from model import GroqLLaMA3 as GroqPPL
 
# initialize the model
model = GroqPPL()

print("Please enter your sentence: (Press Enter twice to start processing)")
contents = []
while True:
    line = input()
    if len(line) == 0:
        break
    contents.append(line)
sentence = "\n".join(contents)

score, diff, std = model.getScore(sentence)
print(f"Score: {score:.3f} | Diff: {diff:.3f} | Std Dev: {std:.3f}")
