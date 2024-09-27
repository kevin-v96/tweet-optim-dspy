# %% [markdown]
# # Data

# %%
import json
with open('./data/hormozi_tweets.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# make a list of 300 tweets
tweets = [item['tweet'] for item in data][:300]

# %% [markdown]
# # Baseline Tweet Generator

# %%
import random
import openai
import os
from dotenv import load_dotenv

load_dotenv()

random_tweets = random.sample(tweets, 10)

prompt = f"""
Past Tweets:
{random_tweets}

Create a new tweet based on the past tweets.
"""

openai.api_key = os.environ.get('OPENAI_API_KEY')

def generate_tweet(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates tweets similar to past tweets from the user."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=280,
    )
    return response.choices[0].message.content.strip().lower()

# Create a new tweet
new_tweet = generate_tweet(prompt)

# %%
new_tweet

# %% [markdown]
# # First experiment
# 
# DSPy recommends that you start with the simplest solution and add complexity, so that is what we'll do. For our first experiement, we'll just use a ChainOfThought module to generate tweets based on all the tweets we've previously seen.

# %%
import json
with open('./data/hormozi_tweets.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# make a list of 300 tweets
tweets = [{'tweet': item['tweet'], 'engagement': item['replies'] + item['retweets'] +item['likes']} for item in data][:300]

# %%
tweets[2]

# %%
import dspy
from dspy.primitives import Example
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot


# %%
tweets

# %% [markdown]
# ## Annotating tweets
# Our tweets don't have any topic or label associated with them that might help us optimize and generate more tweets. We can annotate them by getting an LLM to find out the topic they fall under.

# %%
import dspy

# Convert tweets to Example objects
dataset = [dspy.Example(tweet=tweet).with_inputs("tweet", "engagement") for tweet in tweets]

# %%
class TopicPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("tweet -> topic")

    def forward(self, tweet):
        return self.prog(tweet=tweet)

# %%
mini = dspy.LM(model='gpt-4o')
dspy.configure(lm = mini)

# %%
# Initialize the topic predictor
topic_predictor = TopicPredictor()

# Annotate each tweet with a predicted topic
annotated_dataset = []
for example in dataset:
    response = topic_predictor(example.tweet)
    annotated_example = example.copy()
    annotated_example.topic = response.topic
    annotated_dataset.append(annotated_example)

# Print annotated dataset
for example in annotated_dataset:
    print(f"Tweet: {example.tweet}, Topic: {example.topic}")

# %%
annotated_dataset[0]

# %% [markdown]
# We now have our annotated data. We can save it to use later and train a tweet writer.

# %%
class TweetWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("topic -> tweet")

    def forward(self, topic):
        return self.prog(topic=topic)

# %%
class Assess(dspy.Signature):
    """Assess the creativity of a tweet."""
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Score between 1 and 5")

def creativity_metric(gold, pred, trace=None):
    tweet = pred.tweet
    creativity_question = "Rate the creativity of the following tweet on a scale of 1 to 5."
    creativity_score = dspy.Predict(Assess)(
            assessed_text=tweet,
            assessment_question=creativity_question
        )
    print(creativity_score)
    score = int(creativity_score.assessment_answer.strip())
    return score

# %%
uncompiled_tweet_writer = TweetWriter()

# %%
uncompiled_tweet_writer("Business")

# %% [markdown]
# Let's evaluate our uncompiled tweet writer

# %%
flattened_annotated_dataset = [
    dspy.Example(tweet=item.tweet["tweet"], engagement = item.tweet["engagement"], topic=item["topic"]).with_inputs("topic")
    for item in annotated_dataset
]

# %%
import pickle

with open('data/annotated_dataset.pkl', 'wb') as f:
    pickle.dump(flattened_annotated_dataset, f) 

# %%
dev_set_n = 250
trainset = flattened_annotated_dataset[:dev_set_n]
devset = flattened_annotated_dataset[dev_set_n:]

# %%
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=4, display_progress=True, display_table=5, provide_traceback=True)
evaluator(uncompiled_tweet_writer) 

# %% [markdown]
# ### For unoptimized tweet writer, our score on the devset is: 320

# %% [markdown]
# # BootstrapFewShot

# %%
config = dict(max_bootstrapped_demos=25, max_labeled_demos=4)
teleprompter = BootstrapFewShot(metric=creativity_metric, **config)
optimized_tweet_writer = teleprompter.compile(TweetWriter(), trainset=trainset[:10])

# %%
dspy.settings.configure(backoff_time=5)
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(optimized_tweet_writer)

# %% [markdown]
# It seems like BootstrapFewShots isn't giving us the results we want since it didn't improve on an uncompiled program. Let's try a few more examples and BootstrapFewShotWithRandomSearch
# 
# # BootstrapFewShotWithRandomSearch

# %% [markdown]
# DSPy suggests that you use BootstrapFewShotWithRandomSearch for upto 50 examples.

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

dspy.settings.configure(backoff_time=10)
config = dict(max_labeled_demos = 1, max_bootstrapped_demos=1, num_candidate_programs=2)
teleprompter = BootstrapFewShotWithRandomSearch(metric = creativity_metric, **config)

rs_optimized_tweet_writer = teleprompter.compile(TweetWriter(), trainset=trainset[:20])

# %%
dspy.settings.configure(backoff_time=5)
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(rs_optimized_tweet_writer)

# %% [markdown]
# Let's try MIPROv2 and see if we can improve on this, then we could try out maybe changing the metric to better optimize our writer.

# %%
optimized_tweet_writer.save(path="fewshot")

# %%
rs_optimized_tweet_writer.save(path="fewshotwithrs")

# %% [markdown]
# # MIPROv2

# %%
# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize optimizer
teleprompter = MIPROv2(
    metric=creativity_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=2,
)

mipro_optimized_program = teleprompter.compile(
    TweetWriter(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_trials=10,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=True, 
)

# Save optimize program for future use
mipro_optimized_program.save("mipro_optimized")

# %%
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(mipro_optimized_program)

# %% [markdown]
# # MIPROv2 Results: 330
# 
# MIPROv2 did increase our score from 320 -> 330.
# 
# But as we can see, most of the scores for creativity turn out to be 3 - this is becoming a problem. The LLM judge is probably having a problem defining 'creativity', especially with topics as 'dull' as business. We need a multitude of 'yes/no' metrics and we need to ensemble them to get a good sense. Plus, from the examples in the table above it seems like the tweets generated are really long. We need a conciseness metric as well - or in extreme cases, an assertion that the tweet generated be within 280 characters. It's possible the LLM has issues because we're generating + judging with the same LM, whereas the way it's generally done is a more capable model judging the generation of a smaller/less capable model. Let's try a few changes:
# 
# 1) Changing the metric to encompass more measures of a tweet's 'quality'
# 2) using 4o-mini for generation and 4o for judging
# 3) Playing around with the hyperparameters of DSPy optimizers a bit more
# 4) Manually checking a few examples of generation.
# 5) Using a DSPy program that we compile to judge the quality of tweets as a metric. i.e., something like second-order optimization.
# 6) Using the tweets dataset for RAG since we can't train on the whole thing. (due to time constraints, Rate limit, etc.)
# 7) Trying a different form of annotation/labels than just topics
# 8) Dividing the devset into val and test sets to keep the test set for final testing and preventing data leakage

# %% [markdown]
# # Experiment 2: without RAG

# %%
flattened_annotated_dataset[:5]

# %%
dev_set_n = 250
# Tell DSPy that the 'topic' field is the input. Any other fields are labels and/or metadata.
trainset = [x.without('engagement').with_inputs('topic') for x in flattened_annotated_dataset[:dev_set_n]]
devset = [x.without('engagement').with_inputs('topic') for x in flattened_annotated_dataset[dev_set_n:]]

# %%
trainset[0]

# %%
gpt4o = dspy.LM(model = 'gpt-4o', max_tokens=1000, model_type='chat')
mini = dspy.LM(model = 'gpt-4o-mini', max_tokens=1000, model_type='chat')
dspy.configure(lm = mini)

# %%
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc='ignore if N/A')
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def tweet_ensemble_metric(gold, pred, trace=None):
    topic, tweet = gold.topic, pred.tweet

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    concise = "Does the assessed text make for a concise, cogent tweet?"
    creative = "Does the assessed text make for a creative tweet?"
    relevant = f"The text above should be relevant `{topic}`. The gold answer is `{tweet}`."
    relevant = f"{relevant} Does the assessed text above contain the gold answer?"
    
    with dspy.context(lm=gpt4o):
        relevant =  dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=relevant)
        engaging = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=engaging)
        creative = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=creative)
        concise = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=concise)

    relevant, engaging, creative, concise = (m.assessment_answer.split()[0].lower() == 'yes' for m in [relevant, engaging, creative, concise])
    #takes care of length
    score = (relevant + engaging + creative + concise) if (len(tweet) <= 280) else 0

    if trace is not None: 
        return score >= 4
    return score / 4.0

# %%
devset, testset = devset[:-10], devset[-10:]

# %%
evaluator = Evaluate(devset=devset, metric=tweet_ensemble_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(uncompiled_tweet_writer) 

# %% [markdown]
# ## Uncompiled: 96.25

# %%
config = dict(max_bootstrapped_demos=15, max_labeled_demos=5)
optimizer = BootstrapFewShot(metric = tweet_ensemble_metric, **config)
optimized_tweet_writer = optimizer.compile(TweetWriter(), trainset=trainset[:10])
optimized_tweet_writer.save(path="fewshotv2")

# %%
evaluator(optimized_tweet_writer) 

# %% [markdown]
# ## BootstrapFewShot: 100

# %%
config = dict(max_labeled_demos = 1, max_bootstrapped_demos=1, num_candidate_programs=2)
teleprompter = BootstrapFewShotWithRandomSearch(metric = tweet_ensemble_metric, **config)
rs_optimized_tweet_writer = teleprompter.compile(TweetWriter(), trainset=trainset[:20])
rs_optimized_tweet_writer.save(path="fewshotwithrsv2")

# %%
evaluator(rs_optimized_tweet_writer) 

# %% [markdown]
# ## BootstrapFewShotWithRandomSearch: 95

# %%
# Initialize optimizer
teleprompter = MIPROv2(
    metric=tweet_ensemble_metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=2,
)

mipro_optimized_program = teleprompter.compile(
    TweetWriter(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_trials=10,
    minibatch_size=25,
    minibatch_full_eval_steps=10,
    minibatch=True, 
)

# Save optimize program for future use
mipro_optimized_program.save("mipro_optimizedv2")

# %%
evaluator(mipro_optimized_program)

# %% [markdown]
# # MIPROv2 score: 100
# Given that the eval for this was more robust, this might be a more 'useful' 100 than a 100 from BootstrapFewShot. I'd like to explore how to normalize and juxtapose these scores but leave it up for further work in the interest of time.
# 
# Also during this run, I realised that some of the higher hyperparams in both of my MIPRO runs were completely unneccessary. As such, I'll reduce them for the next experiment.

# %% [markdown]
# # Experiment 3: with RAG

# %%
import pickle 

with open('data/annotated_dataset.pkl', 'rb') as pickle_file:
    annotated_dataset = pickle.load(pickle_file)

# %%
rag_entries = 200
#200 entries
set_for_rag = annotated_dataset[:rag_entries]
#100 entries
set_for_training = annotated_dataset[rag_entries:]
#75 for training, 25 for val
dev_set_n = 75
trainset = [x.without('engagement').with_inputs('topic') for x in set_for_training[:dev_set_n]]
devset = [x.without('engagement').with_inputs('topic') for x in set_for_training[dev_set_n:]]

# %%
import weaviate
from dotenv import load_dotenv
import os

load_dotenv()
  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = weaviate.connect_to_local(host='localhost', port=8080, headers={
    'X-Openai-Api-Key': OPENAI_API_KEY
})

print(client.is_ready())

# %%
import weaviate.classes.config as wvcc
  
client.collections.delete_all()
collection = client.collections.create(
    name = "DspyTweets",
    vectorizer_config=wvcc.Configure.Vectorizer.text2vec_openai(),
    properties=[
        wvcc.Property(name = "content", data_type=wvcc.DataType.TEXT),
    ],
)

# %%
import re

def chunk_list(lst, chunk_size):
    """Break a list into chunks of the specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def split_into_sentences(text):
    """Split text into sentences using regular expressions."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def chunk_tweets():
    """Read index.md files from subfolders, split into sentences, and chunk every 5 sentences."""
    tweet_chunks = []
    for tweet in set_for_rag:
        content = tweet.tweet
        sentences = split_into_sentences(content)
        sentence_chunks = chunk_list(sentences, 5)
        sentence_chunks = [' '.join(chuck) for chuck in sentence_chunks]
        tweet_chunks.extend(sentence_chunks)
    return tweet_chunks

tweet_chunks = chunk_tweets()

# %%
len(tweet_chunks)

# %%
tweet_chunks[0]

# %%
tweets_collection = client.collections.get("DspyTweets")

for idx, tweet_chunk in enumerate(tweet_chunks):
    upload = tweets_collection.data.insert(
        properties={
            "content": tweet_chunk
        }
    )

# %%
import dspy
from dspy.retrieve.weaviate_rm import WeaviateRM

retriever_model = WeaviateRM(
    weaviate_collection_name="DspyTweets",
    weaviate_client=client,
)

results = retriever_model("entrepreneurship", k=5)

for result in results:
    print("Document:", result.long_text, "\n")

# %%
dspy.configure(rm = retriever_model)

# %%
# Define the signature for automatic assessments.
gpt4o = dspy.LM(model= 'gpt-4o', max_tokens=1000, model_type='chat')
retrieve = dspy.Retrieve(k=5)

class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    context = dspy.InputField(desc='ignore if N/A')
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def metric(gold, pred, trace=None):
    topic, tweet = gold.topic, pred.tweet
    context = retrieve(topic).passages

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    faithful = "Is the assessed text grounded in the context? Say no if it includes significant facts not in the context."
    concise = "Does the assessed text make for a concise, cogent tweet?"
    creative = "Does the assessed text make for a creative tweet?"
    relevant = f"The this is the tweet: `{tweet}`. It should be relevant to: `{topic}`."
    
    with dspy.context(lm=gpt4o):
        faithful = dspy.Predict(Assess)(context=context, assessed_text=tweet, assessment_question=faithful)
        relevant =  dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=relevant)
        engaging = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=engaging)
        creative = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=creative)
        concise = dspy.Predict(Assess)(context='N/A', assessed_text=tweet, assessment_question=concise)

    relevant, engaging, creative, concise, faithful = (m.assessment_answer.split()[0].lower() == 'yes' for m in [relevant, engaging, creative, concise, faithful])
    #takes care of length
    score = (relevant + engaging + creative + concise + faithful) if (len(tweet) <= 280) else 0

    if trace is not None: 
        return score >= 5
    return score / 5.0

# %%
class GenerateTweet(dspy.Signature):
    """Generate a tweet based on the context"""

    context = dspy.InputField(desc = "May contain relevant facts")
    topic = dspy.InputField()
    tweet = dspy.OutputField()

class TweetRAG(dspy.Module):
    def __init__(self, num_passages = 3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k = num_passages)
        self.generate_tweet = dspy.ChainOfThought(GenerateTweet)

    def forward(self, topic):
        context = self.retrieve(topic).passages
        prediction = self.generate_tweet(context = context, topic = topic)
        return dspy.Prediction(tweet = prediction.tweet)

# %%
mini = dspy.LM(model = 'gpt-4o-mini')
dspy.configure(lm=mini)

# %%
from dspy.evaluate.evaluate import Evaluate
uncompiled_tweet_rag = TweetRAG()
evaluator = Evaluate(devset=devset, metric=metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(uncompiled_tweet_rag) 

# %% [markdown]
# ## Uncompiled Tweet-RAG: 91.2

# %%
from dspy.teleprompt import BootstrapFewShot
config = dict(max_bootstrapped_demos=15, max_labeled_demos=5)
optimizer = BootstrapFewShot(metric = metric, **config)
optimized_tweet_writer = optimizer.compile(TweetRAG(), trainset=trainset[:10])
optimized_tweet_writer.save(path="fewshotv3")

# %%
evaluator(optimized_tweet_writer)

# %% [markdown]
# ## BootstrapFewShot: 99.2

# %%
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
config = dict(max_labeled_demos = 2, max_bootstrapped_demos=2, num_candidate_programs=3)
teleprompter = BootstrapFewShotWithRandomSearch(metric = metric, **config)
rs_optimized_tweet_writer = teleprompter.compile(TweetRAG(), trainset=trainset[:20])
rs_optimized_tweet_writer.save(path="fewshotwithrsv3")

# %%
mini.inspect_history(n=1)

# %%
evaluator(rs_optimized_tweet_writer)

# %% [markdown]
# ## BootstrapFewShotWithRandomSearch: 99.2

# %%
from dspy.teleprompt import MIPROv2
# Initialize optimizer
teleprompter = MIPROv2(
    metric=metric,
    num_candidates=7,
    init_temperature=0.5,
    verbose=False,
    num_threads=2,
)

mipro_optimized_program = teleprompter.compile(
    TweetRAG(),
    trainset=trainset,
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_trials=5,
    minibatch=True,
    minibatch_size=5, 
    minibatch_full_eval_steps=2
)

# Save optimize program for future use
mipro_optimized_program.save("mipro_optimizedv3")

# %%
mini.inspect_history(n=1)

# %%
evaluator(mipro_optimized_program)

# %% [markdown]
# ## MIPROv2 with RAG: 91.2

# %% [markdown]
# # Experiment 4: Second Order Optimization
# 
# DSPy allows us to use a DSPy program as a metric. Thus, we can optimize this program itself by passing it some examples of judgement that it should learn from. This is a complete shift in paradigm compared to the evaluation we've used above. Let's try this out.

# %%
#50 for training, 25 for second-order-optimization (named devset here), 25 for eval
dev_set_n = 50
trainset = [x.without('engagement').with_inputs('topic') for x in set_for_training[:50]]
devset = [x.without('engagement').with_inputs('topic') for x in set_for_training[50:75]]
valset = [x.without('engagement').with_inputs('topic') for x in set_for_training[75:]]

# %%
class UnoptimisedMetric(dspy.Module):
    """Metric that will judge whether tweets are good in quality"""
    def __init__(self, metric_function):
        super().__init__()
        self.metric = metric_function

    def forward(self, first_order_gold, first_order_pred, expected_score):
        prediction = self.metric(first_order_gold, first_order_pred)
        #print("\nThe prediction for this pair is: ", prediction)
        return dspy.Prediction(answer = prediction)

# %%
import numpy as np

def metric_of_metric(second_order_gold, second_order_pred, trace=None):
    """Determine whether the metric judges the similarity of the texts well enough"""
    score_by_metric = second_order_pred.answer
    expected_score = second_order_gold.expected_score
    score_diff = float(np.abs(score_by_metric - expected_score))
    # print("\nExpected score: ", expected_score)
    # print("\nMetric score: ", score_by_metric)
    # print("\nScore difference is: ", score_diff)

    return 1 - score_diff

# %% [markdown]
# ## Exploring the possibility for a dataset of metric generations

# %%
gold = devset[0]
pred = mipro_optimized_program(devset[0].tweet)
print("Example of input/gold label for metric: ", gold)
print("\nExample of the output/prediction for metric: ", pred)
print("\n output of metric for the above inputs: ", metric(gold, pred))

# %% [markdown]
# We need to have a training set for the metric of metric that looks like the above. i.e., we need to optimize for the output according to how we want our generation to look.

# %%
for example in devset:
    print(f"""Example(first_order_gold = Example(topic = \"\"\"{example.topic}\"\"\", tweet = \"\"\"{example.tweet}\"\"\"), 
            first_order_pred = Example(tweet = "PUT PREDICTION HERE"),
            expected_score = "PUT SCORE HERE").with_inputs('first_order_gold', 'first_order_pred'),""")

# %% [markdown]
# Let's add about half positive and half negative examples. i.e., the 'expected score' should be distributed equally over the 4 values values (0.0, 0.3333, 0.6666, 1.0)

# %%
for idx in range(6, 12):
    topic = devset[idx].topic
    print("Topic: ", topic)
    tweet = mipro_optimized_program(topic).tweet
    print("\ntweet: ", tweet)

# %%
for idx in range(12, 18):
    topic = devset[idx].topic
    print("Topic: ", topic)
    tweet = rs_optimized_tweet_writer(topic).tweet
    print("\ntweet: ", tweet)

# %% [markdown]
# ## Manual dataset for metric optimization
# I've carefully curated the below dataset to align the metric. It's divided in generations with 4 scores - 0, that don't have anythine to do with the topic, 0.33, which are somewhat relevant, 0.66, which are pretty relevant and good generations, and 1.0, which are the exact same tweets. I'm hoping that feeding this to the optimizer for the metric will result in a metric more aligned with what we want.

# %%
metric_trainset = [
    #generations with score 0
Example(first_order_gold = Example(topic = """Learning and Experience""", tweet = """Doing a thing increases your ability to learn about it.
In order words, do the thing, then read about it. The experience will give you a framework to lay the new ideas on. 
Otherwise you’re mentally masturbating to the idea of doing it rather than learning to do it better."""), 
            first_order_pred = Example(tweet = "Sure, what would you like?"),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Everyday Life/Humor""", tweet = """Anyone else ever accidentally get in the shower with a hat still on?"""), 
            first_order_pred = Example(tweet = "What a tragedy! A major fire in Dagenham last month which had the same cause as the Grenfell Tower tragedy proves that fire fighters and the public are still at risk from a major blaze, a union boss has warned."),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Decision Making""", tweet = """If you have all the information to make a perfect decision, you missed the opportunity."""), 
            first_order_pred = Example(tweet = """I was in a sheer dress the day that we met
We were both in a rush, we talked for a sec
Your friend hit me up so we could connect
And what are the odds? You send me a text"""),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Habit Formation""", tweet = """I’m bad at starting new habits. I’m also pretty bad at keeping habits. In the last ten years, the only small habits I’ve been able to stick with came from modifying my environment. Here’s what I did: I identified my ‘watering holes’

The places in my house I sit for extended…"""), 
            first_order_pred = Example(tweet = """The Hobbit, or There and Back Again is a children's fantasy novel by the English author J. R. R. Tolkien. It was published in 1937 to wide critical acclaim, being nominated for the Carnegie Medal and awarded a prize from the New York Herald Tribune for best juvenile fiction. The book is recognized as a classic in children's literature and is one of the best-selling books of all time, with over 100 million copies sold."""),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Mental Health""", tweet = """The reason you are stressed is you have decisions to make and you’re not making them."""), 
            first_order_pred = Example(tweet = """Teletubbies is a British children's television series created by Anne Wood and Andrew Davenport for the BBC. The programme focuses on four differently coloured characters known as the Teletubbies, named after the television screens on their bellies. Recognised throughout popular culture for the uniquely shaped antenna protruding from the head of each character, the Teletubbies communicate through gibberish and were designed to bear resemblance to toddlers."""),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Personal Development""", tweet = """Winning builds your circumstances. 
Losing builds your character."""), 
            first_order_pred = Example(tweet = """Fight Club is a 1999 American film directed by David Fincher, and starring Brad Pitt, Edward Norton and Helena Bonham Carter. It is based on the 1996 novel by Chuck Palahniuk. Norton plays the unnamed narrator, who is discontented with his white-collar job. He forms a "fight club" with soap salesman Tyler Durden (Pitt), and becomes embroiled in a relationship with an impoverished but beguilingly attractive woman, Marla Singer (Bonham Carter)."""),
            expected_score = 0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
#generations with score 0.33
Example(first_order_gold = Example(topic = """Business Strategy""", tweet = """Best reason to do bigger deals:

Same work. More zeroes."""), 
            first_order_pred = Example(tweet = "Great entrepreneurs see beyond profits; they create value and partnerships that elevate their business. Shift your mindset: be the marketing engine for bigger players and watch your impact multiply! #BusinessStrategy #Entrepreneurship"),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Sales and Marketing""", tweet = """You know how to get a prospect to believe you’ll provide them value after they buy…provide them value before they buy."""), 
            first_order_pred = Example(tweet = "Great salesmen know their prospects inside out, not just their products! 🎯 Marketing thrives on creativity, not copies. And did you know? Audio accounts for 50% of book sales! 📚🔊 #Sales #Marketing #Innovation"),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Future of Human-Computer Interaction""", tweet = """We will quickly have no need for mouses or fingers to interact with computers. 

We’ll just say what we want and the AI will interpret it for the software or site.

After all, the only reason we had to point or click was to get it to understand what we wanted it to do. No more."""), 
            first_order_pred = Example(tweet = "🚀 The future of human-computer interaction is bright! With AI evolving rapidly, we might just be at the MySpace stage of a tech revolution. Imagine the possibilities in 24 months! #AI #FutureTech #HumanComputerInteraction"),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Personal Experience, Life Lessons, Fraternity Life""", tweet = """When I was 19 years old I pledged a fraternity, and I learned one of the most important lessons of my life on how to deal with women (and others)…
*context*
So when you’re a pledge you have to drive drunk brothers around. You’re the designated driver.
One night, I was driving an…"""), 
            first_order_pred = Example(tweet = """In life, there are no winners or losers, just players and quitters. Sometimes, the hardest choice is leaving the sure thing for the best thing. Remember: "Be too good to fail." Embrace the journey! #LifeLessons #FraternityLife #PersonalGrowth"""),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """COVID-19 Pandemic / Face Masks""", tweet = """It’s like a permanent face mask."""), 
            first_order_pred = Example(tweet = "Amid the COVID-19 chaos, some say 'word of mouth is dead,' but community support for face masks is more vital than ever! Remember, if it won’t matter in 10 million years, don’t let it stress you today. Let’s protect each other! 😷💪 #COVID19 #FaceMasks #Community"),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Personal grooming and appearance""", tweet = """If you’re an ugly dude, consider a beard. Leaves half your face to their imagination."""), 
            first_order_pred = Example(tweet = "Personal grooming isn't just about looking good; it's about being prepared and confident. Remember, it's better to be envied than pitied. #PersonalResponsibility #SelfConfidence"),
            expected_score = 1 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
#generations with score 0.66
Example(first_order_gold = Example(topic = """Personal Responsibility and Empowerment""", tweet = """The moment I realized I had given the person I resented the most power over my life, I realized that whoever carries the blame carries the power. 

So until your finger of blame points at your chest rather than someone else’s, you stay a victim of your own making."""), 
            first_order_pred = Example(tweet = """"Better to be envied than pitied." Embrace personal responsibility and empower yourself! Focus on being too good to fail and cultivate the skills that lead to success. #PersonalResponsibility #Empowerment"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Personal Growth and Individualism""", tweet = """Sometimes you have to let your parents’ dream die for yours to live."""), 
            first_order_pred = Example(tweet = """"It's better to be envied than pitied." A powerful reminder of personal responsibility! In a world of self-love, let’s not forget the value of humility and the skills that drive true growth. #PersonalGrowth #Individualism"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Societal Expectations and Age""", tweet = """30 isn’t the new 20.

It’s an excuse to take ten years longer to accomplish the same thing."""), 
            first_order_pred = Example(tweet = """Chase your dreams for YOU, not for others. In the end, how much will it matter to them? Focus on building skills like patience and honesty that truly enrich your life. #SocietalExpectations #PersonalGrowth"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Emotional Fear and Commitment""", tweet = """You like them more than being alone.
But not enough to say ‘only you.’

That’s not a commitment issue. 
It’s fear.

Fear of being alone.
Fear of not finding someone better.

The same reason people stay in jobs they hate. They don’t have a commitment issue. Theyre afraid to leave."""), 
            first_order_pred = Example(tweet = """Embrace your fears as fuel for commitment! Use pain, fear, and anger to drive your entrepreneurial journey. Remember: "Be too good to fail" and focus on what you can control. #Entrepreneurship #EmotionalStrength"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Relationships""", tweet = """You don’t have commitment issues. 
You just don’t like them that much."""), 
            first_order_pred = Example(tweet = """Remember, there's always another opportunity around the corner in relationships! Don't let ego hold you back from making the right choices. If you can't end something, you shouldn't start it. #Relationships #SelfAwareness"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Customer Service""", tweet = """Also works with “…the best tip you can give me is to bring a friend next time.”"""), 
            first_order_pred = Example(tweet = """In customer service, remember: there's always another opportunity around the corner! Focus on understanding your customers, not just your product. Choose wisely when seeking education or service—quality matters! #CustomerService #SalesStrategy"""),
            expected_score = 2 / 3.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
#generations with score 1.0
Example(first_order_gold = Example(topic = """Customer Service""", tweet = """I heard a great line today:

“On the receipt when it says to leave a tip, please don’t. The best tip you can give me is to come back and see me again.”

Love it."""), 
            first_order_pred = Example(tweet = """I heard a great line today:

“On the receipt when it says to leave a tip, please don’t. The best tip you can give me is to come back and see me again.”

Love it."""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Productivity and Decision-Making""", tweet = """If you think of “yeses” as withdrawals and “nos” as deposits, you’ll do fewer things and make more money from them."""), 
            first_order_pred = Example(tweet = """If you think of “yeses” as withdrawals and “nos” as deposits, you’ll do fewer things and make more money from them."""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Business Strategy""", tweet = """Hiring talented people is one of the easiest 10x-100x ROI opportunities.

And there’s always underpriced talent."""), 
            first_order_pred = Example(tweet = """Hiring talented people is one of the easiest 10x-100x ROI opportunities.

And there’s always underpriced talent."""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Relationships and Trust""", tweet = """Enemies are better than fake friends. 

At least you know who has your back rather than who plans on putting a knife in it."""), 
            first_order_pred = Example(tweet = """Enemies are better than fake friends. 

At least you know who has your back rather than who plans on putting a knife in it."""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Entrepreneurship Skills""", tweet = """Four skill deficiencies of an entrepreneur—they don’t know:
1) How to do something
2) How to get other people to do something
3) How to get them selves to do something
4) Something is possible"""), 
            first_order_pred = Example(tweet = """Four skill deficiencies of an entrepreneur—they don’t know:
1) How to do something
2) How to get other people to do something
3) How to get them selves to do something
4) Something is possible"""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Trust and Relationships""", tweet = """Let me save you a bunch of headache: 
If someone steals from you and promises they won’t. They’re lying, they’ll steal again…just more effectively.

And even if they never do, there are buckets of other great people you can choose from without that track record.

Cut and move on"""), 
            first_order_pred = Example(tweet = """Let me save you a bunch of headache: 
If someone steals from you and promises they won’t. They’re lying, they’ll steal again…just more effectively.

And even if they never do, there are buckets of other great people you can choose from without that track record.

Cut and move on"""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
Example(first_order_gold = Example(topic = """Customer Service Challenges""", tweet = """When you have bad customers, more help doesn’t help more."""), 
            first_order_pred = Example(tweet = """When you have bad customers, more help doesn’t help more."""),
            expected_score = 1.0).with_inputs('first_order_gold', 'first_order_pred', 'expected_score'),
]

# %%
import pickle 

with open('data/metric_trainset.pkl', 'wb') as f:
    pickle.dump(metric_trainset, f) 

# %% [markdown]
# ## Optimizing the metric

# %%
teleprompter = BootstrapFewShotWithRandomSearch(metric=metric_of_metric)
unoptimized_metric = UnoptimisedMetric(metric)
optimized_metric = teleprompter.compile(unoptimized_metric, trainset=metric_trainset)

# %%
config = dict(max_bootstrapped_demos=25, max_labeled_demos=len(metric_trainset))
teleprompter = BootstrapFewShot(metric=metric_of_metric, **config)
unoptimized_metric_for_bs = UnoptimisedMetric(metric)
optimized_metric_from_bs = teleprompter.compile(unoptimized_metric_for_bs, trainset=metric_trainset)

# %%
from dspy.teleprompt import BootstrapFewShot
config = dict(max_bootstrapped_demos=15, max_labeled_demos=5)
optimizer = BootstrapFewShot(metric = unoptimized_metric, **config)
unoptimized_tweet_writer = optimizer.compile(TweetRAG(), trainset=trainset[:10])
unoptimized_tweet_writer.save(path="fewshotv41")

# %%
evaluator(unoptimized_tweet_writer)

# %%
from dspy.teleprompt import BootstrapFewShot
config = dict(max_bootstrapped_demos=15, max_labeled_demos=5)
optimizer = BootstrapFewShot(metric = optimized_metric, **config)
optimized_tweet_writer = optimizer.compile(TweetRAG(), trainset=trainset[:10])
optimized_tweet_writer.save(path="fewshotv42")
evaluator(optimized_tweet_writer)

# %%
from dspy.teleprompt import BootstrapFewShot
config = dict(max_bootstrapped_demos=15, max_labeled_demos=5)
optimizer = BootstrapFewShot(metric = optimized_metric_from_bs, **config)
optimized_tweet_writer_from_bs = optimizer.compile(TweetRAG(), trainset=trainset[:10])
optimized_tweet_writer_from_bs.save(path="fewshotv43")
evaluator(optimized_tweet_writer_from_bs)

# %%



