from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import boto3
import os
import streamlit as st

# AWS Profile
os.environ["AWS_PROFILE"] = "bedrock"

# Create Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# Model ID
modelID = "amazon.titan-text-premier-v1:0"

# Initialize LLM
llm = BedrockLLM(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={
        "maxTokenCount": 2000,
        "temperature": 0.9
    }
)

def my_chatbot(language, freeform_text):
    prompt = PromptTemplate(
        input_variables=["language", "freeform_text"],
        template="You are a chatbot. Respond in {language}.\n\n{freeform_text}"
    )

    # New chaining method
    chain = prompt | llm

    # Use invoke instead of ()
    response = chain.invoke({'language': language, 'freeform_text': freeform_text})

    return response

#print(my_chatbot("english", "what is operator?"))



st.title("Bedrock Chatbot")

language = st.sidebar.selectbox("Language", ["english", "spanish"])

if language:
    freeform_text = st.sidebar.text_area(label="what is your question?",
    max_chars=100)

if freeform_text:
    response = my_chatbot(language,freeform_text)
    st.write(response)
