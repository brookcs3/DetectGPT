"""
This version uses Groq-hosted LLaMA 3 models for AI text detection and analysis.
Originally based on Hugging Face's perplexity demo:
https://huggingface.co/docs/transformers/perplexity

Code modified and extended to work with the Groq API for ultra-fast inference.

Original authors: Burhan Ul Tayyab and Nicholas Chua
Groq-based integration and refactor by Cameron Brooks.
"""

from model import GPT2PPLV2 as GPT2PPL
from fastapi import FastAPI, Form, Request
import gradio as gr
from database import DB
from HTML_MD_Components import bannerHTML, emailHTML, discordHTML

CUSTOM_PATH = "/"

app = FastAPI()

# initialize the model
model = GPT2PPL()
database = DB()

@app.post("/postdb")
def uploadDataBase(request: Request, email: str = Form()):
    if request and request.client:
        database.set(request.client.host, email)
        return "Email Sent"
    return "Invalid Request"

@app.get("/infer")
def infer(sentence: str):
    return model(sentence, 512, "v1.1")

def inference(*args):
    return model(*args)

with gr.Blocks(title="Groq Detector", css="#discord {text-align: center} #submit {background-color: #FF8C00} #advertisment {text-align: center;} #email {height:120%; background-color: LightSeaGreen} #blank {margin:150px} #code_feedback { margin-left:-0.3em;color:gray;text-align: center;margin-bottom:-100%;padding-bottom:-100%}") as io:
    with gr.Row():
        gr.HTML(bannerHTML, visible=True)
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=8):
            gr.Markdown('<h1 style="text-align: center;">Groq Detector <span style="font-size:small">(LLaMA 3, Groq-hosted)</span></h1>')
        with gr.Column(scale=1, elem_id="discord"):
            gr.HTML(discordHTML, visible=True)
    with gr.Row():
        gr.Markdown("Use Groq Detector to evaluate whether the text was written by a human or AI, using Groq-hosted LLaMA 3.")
    with gr.Row():
        with gr.Column(scale=1):
            InputTextBox = gr.Textbox(lines=7, placeholder="Please Insert your text(s) here", label="Texts")
            submit_btn = gr.Button("Submit", elem_id="submit")
        with gr.Column(scale=1):
            OutputLabels = gr.JSON(label="Output")
            OutputTextBox = gr.Textbox(show_label=False)

    submit_btn.click(lambda x: inference(x, 512, "v1.1"), inputs=[InputTextBox], outputs=[OutputLabels, OutputTextBox], api_name="infer")

    with gr.Row():
        gr.Markdown('# <span style="color:#006400">Register</span> here for updates.')
    with gr.Row():
        with gr.Column(scale=5):
            emailTextBox = gr.HTML(emailHTML)
        with gr.Column(scale=5):
            pass

    with gr.Row():
        gr.Markdown('For <a style="text-decoration:none;color:gray" href="mailto:groqdetector@yourdomain.com" target="_blank">feedback</a>, contact us at groqdetector@yourdomain.com', elem_id="code_feedback")

app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)
