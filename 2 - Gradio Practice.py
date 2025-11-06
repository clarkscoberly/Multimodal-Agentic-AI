
import gradio as gr
import torch
import requests


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

def process_inputs(text, name):
    return f"Hello world! - {text} {name}"

demo = gr.Interface(
    fn = process_inputs,
    inputs = [
        gr.Textbox(label = 'Enter Text'),
        gr.Textbox(label = 'Enter your name')
    ],
    outputs =  gr.Textbox(label = 'Output')

)

demo.launch()


# pip install transformers