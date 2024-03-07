import gradio as gr
import os
import time
from multi_agent_chatbot import multi_agent_chatbot

# Chatbot demo with multimodal input (text, markdown, LaTeX, code blocks, image, audio, & video). Plus shows support for streaming text.

multi_agent_chatbot = multi_agent_chatbot()

def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    # print(history[-1][0][0])
    # print(history[-1][0])
    # print(type(history[-1][0]))
    # response = "**That's cool!**"
    response = multi_agent_chatbot.respond(history[-1][0])
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center; color: #ffffff; font-size: 40px;'> 🏡 🤖 Garden Whisperer AI - Your personal kitchen gardening assistant (POC)")
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        height=800,
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "assets//avatar.png"))),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)


demo.queue()
if __name__ == "__main__":
    demo.launch()
