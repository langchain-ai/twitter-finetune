import json
from langchain.schema import AIMessage
from langchain.adapters.openai import convert_message_to_dict
import time
import openai
from io import BytesIO


if __name__ == "__main__":
    with open('dataset_twitter-scraper_2023-08-23_22-13-19-740.json') as f:
        data = json.load(f)

    tweets = [d["full_text"] for d in data if "t.co" not in d['full_text']]
    messages = [AIMessage(content=t) for t in tweets]
    system_message = {"role": "system", "content": "write a tweet"}
    data = [[system_message, convert_message_to_dict(m)] for m in messages]


    my_file = BytesIO()
    for m in data:
        my_file.write((json.dumps({"messages": m}) + "\n").encode('utf-8'))

    my_file.seek(0)
    training_file = openai.File.create(
      file=my_file,
      purpose='fine-tune'
    )
    while True:
        try:
            job = openai.FineTuningJob.create(training_file=training_file.id, model="gpt-3.5-turbo")
            break
        except Exception as e:
            print(e)
            print("Trying again in ten seconds....")
            time.sleep(10)

    start = time.time()

    while True:
        ftj = openai.FineTuningJob.retrieve(job.id)
        if ftj.fine_tuned_model is None:
            print(f"Waiting for fine-tuning to complete... Elapsed: {time.time() - start}", end="\r", flush=True)
            time.sleep(10)
        else:
            print("\n")
            print(ftj.fine_tuned_model, flush=True)
            break
