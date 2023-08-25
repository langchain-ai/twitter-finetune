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

normal_chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet about {topic} in the style of Elon Musk") ])
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    ChatPromptTemplate.from_messages([("system", "write a tweet about {topic}") ])
    | ChatOpenAI(model="ft:gpt-3.5-turbo-0613:langchain::7qqjIosa")
    | StrOutputParser()
)

def generate_tweet_normal(topic):
    result = normal_chain.invoke({"topic": topic})
    wait_for_all_tracers()
    return result

def generate_tweet(topic):
    result = chain.invoke({"topic": topic}, config=runnable_config)
    run = run_collector.traced_runs[0]
    run_collector.traced_runs = []
    st.session_state.run_id = run.id
    wait_for_all_tracers()
    return result


col1, col2 = st.columns([1, 6])  # Adjust the ratio for desired layout

# Display the smaller image in the first column
col1.image("elon.jpeg")  # Adjust width as needed

# Display the title in the second column
col2.title("Elon Musk Tweet Generator")
st.info("This generator was finetuned on tweets by Elon Musk to imitate his style. Source code [here](https://github.com/langchain-ai/twitter-finetune)\n\nTwo tweets will be generated: one using a finetuned model, one useing a prompted model. Afterwards, you can provide feedback about whether the finetuned model performed better!")

topic = st.text_input("Enter a topic:")
if 'show_tweets' not in st.session_state:
    st.session_state.show_tweets = None

if st.button("Generate Tweets"):
    if topic:
        col3, col4 = st.columns([6, 6])
        tweet = generate_tweet(topic)
        col3.markdown("### Finetuned Tweet:")
        col3.write(f"🐦: {tweet}")
        col3.markdown("---")  # Add a horizontal line for separation
        feedback = streamlit_feedback(
            feedback_type="thumbs",
            key=f"feedback_{st.session_state.run_id}",
            align="flex-start"
        )
        scores = {"👍": 1, "👎": 0}
        if feedback:
            score = scores[feedback["score"]]
            feedback = client.create_feedback(st.session_state.run_id, "user_score", score=score)
            st.session_state.feedback = {"feedback_id": str(feedback.id), "score": score}
        tweet = generate_tweet_normal(topic)
        col4.markdown("### Prompted Tweet:")
        col4.write(f"🐦: {tweet}")
        col4.markdown("---")  # Add a horizontal line for separation
    else:
        st.warning("Please enter a topic before generating a tweet.")
    