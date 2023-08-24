import langchain
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig
from langsmith import Client
from streamlit_feedback import streamlit_feedback

client = Client()

if "run_id" not in st.session_state:
    st.session_state.run_id = None

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(
    callbacks=[run_collector],
    tags=["Streamlit Chat"],
)

chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet about {topic}") ])
    | ChatOpenAI(model="ft:gpt-3.5-turbo-0613:langchain::7qqjIosa")
    | StrOutputParser()
)

def generate_tweet(topic):
    result = chain.invoke({"topic": topic}, config=runnable_config)
    run = run_collector.traced_runs[0]
    run_collector.traced_runs = []
    st.session_state.run_id = run.id
    wait_for_all_tracers()
    return result

def main():
    col1, col2 = st.columns([1, 6])  # Adjust the ratio for desired layout

    # Display the smaller image in the first column
    col1.image("elon.jpeg")  # Adjust width as needed

    # Display the title in the second column
    col2.title("Elon Musk Tweet Generator")
    st.info("This generator was finetuned on tweets by Elon Musk to imitate his style. Source code [here](https://github.com/langchain-ai/twitter-finetune)")

    topic = st.text_input("Enter a topic:")

    if st.button("Generate Tweet"):
        if topic:
            tweet = generate_tweet(topic)
            st.markdown("### Generated Tweet:")
            st.write(f"🐦: {tweet}")
            st.markdown("---")  # Add a horizontal line for separation
        else:
            st.warning("Please enter a topic before generating a tweet.")
    if st.session_state.get("run_id"):
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            key=f"feedback_{st.session_state.run_id}",
        )
        scores = {"👍": 1, "👎": 0}
        if feedback:
            score = scores[feedback["score"]]
            feedback = client.create_feedback(st.session_state.run_id, "user_score", score=score)
            st.session_state.feedback = {"feedback_id": str(feedback.id), "score": score}


if __name__ == "__main__":
    main()
