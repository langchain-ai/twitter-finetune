# Twitter Finetune

This repo shows how to finetune GPT-3.5-Turbo on tweets. Specifically, Elon Musk's tweets.

In this example, data is first exported from Twitter using [Apify](https://apify.com/).
A copy of this data can be found in [dataset_twitter-scraper_2023-08-23_22-13-19-740.json](dataset_twitter-scraper_2023-08-23_22-13-19-740.json).

This data is then loaded to a format with which it can be used to finetune a GPT-3.5-Turbo model, and is then used to do exactly that. This can be done by running `python ingest.py`.

A Streamlit app is this created to compare this finetuned model to a prompted GPT-3.5-Turbo model.
This can be run with `streamlit run app.py`.

Access the final app hosted on Streamlit [here](https://elon-twitter-clone.streamlit.app/).
