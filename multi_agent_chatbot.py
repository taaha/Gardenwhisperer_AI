from llama_index.legacy.multi_modal_llms.openai_utils import (
    generate_openai_multi_modal_chat_message,
)
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

class multi_agent_chatbot():
    def __init__(self):
        conversational_flow_prompt = """
You are a helpful assistant named 'Garden Whisperer AI' specialising in helping users set up kitchen gardens.
You will first on board a person in following steps
1. ask them to share a picture of their space. analyse the place regarding sunlight, water drainage, etc
2. ask them about weather and suggest plants
3. Once user is convinced on a plant. Give him suggestions on how to grow it and where to place it in space
4. Give user schedule on when to water it
4. Also then help user if he wants to know anything

Keep it brief, simple and fun
"""
        chat_query = generate_openai_multi_modal_chat_message(
            prompt=conversational_flow_prompt,
            role="system",
        )
        self.chat_messages = [chat_query]
        self.openai_mm_llm = OpenAIMultiModal(
            model="gpt-4-vision-preview", api_key= os.environ["OPENAI_API_KEY"] , max_new_tokens=300
        )

    def respond(self, query):
        if type(query) == str:
            chat_query = generate_openai_multi_modal_chat_message(
                prompt=query,
                role="user",
            )
        else:
            image_documents = SimpleDirectoryReader(input_files=query).load_data()
            chat_query = generate_openai_multi_modal_chat_message(
                prompt="",
                role="user",
                image_documents=image_documents,
            )

        self.chat_messages.append(chat_query)
        response = self.openai_mm_llm.chat(
            messages=self.chat_messages,
        )
        chat_response = generate_openai_multi_modal_chat_message(
            prompt=response.message.content,
            role="assistant",
        )
        self.chat_messages.append(chat_response)

        return response.message.content
        # else:
        #     # put your local directory here
        #     image_documents = SimpleDirectoryReader(input_files=query).load_data()
        #
        #     response = self.openai_mm_llm.complete(
        #         prompt="Describe the images as an alternative text",
        #         image_documents=image_documents,
        #     )
        #     return response
