import langchain
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet") ])
    | ChatOpenAI(model="ft:gpt-3.5-turbo-0613:langchain::7qqjIosa")
    | StrOutputParser()
)

def generate_tweet():
    result = chain.invoke({})
    return result

def main():
    st.title("Tweet Generator")

    if st.button("Generate Tweet"):
        tweet = generate_tweet()
        st.write(tweet)

if __name__ == "__main__":
    main()
