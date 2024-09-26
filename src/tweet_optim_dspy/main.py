import json
import dspy
from dspy.primitives import Example
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2
import pickle

with open('./data/hormozi_tweets.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# make a list of 300 tweets
tweets = [{'tweet': item['tweet'], 'engagement': item['replies'] + item['retweets'] +item['likes']} for item in data][:300]

# Convert tweets to Example objects
dataset = [dspy.Example(tweet=tweet).with_inputs("tweet", "engagement") for tweet in tweets]

class TopicPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("tweet -> topic")

    def forward(self, tweet):
        return self.prog(tweet=tweet)
    
llm = dspy.LM(model='gpt-4o')
dspy.configure(lm = llm)

# Initialize the topic predictor
topic_predictor = TopicPredictor()

# Annotate each tweet with a predicted topic
annotated_dataset = []
for example in dataset:
    response = topic_predictor(example.tweet)
    annotated_example = example.copy()
    annotated_example.topic = response.topic
    annotated_dataset.append(annotated_example)
flattened_annotated_dataset = [
    dspy.Example(tweet=item.tweet["tweet"], engagement = item.tweet["engagement"], topic=item["topic"]).with_inputs("topic")
    for item in annotated_dataset
]

with open('data/annotated_dataset.pkl', 'wb') as f:
    pickle.dump(flattened_annotated_dataset, f) 

dev_set_n = 250
trainset = flattened_annotated_dataset[:dev_set_n]
devset = flattened_annotated_dataset[dev_set_n:]

class TweetWriter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("topic -> tweet")

    def forward(self, topic):
        return self.prog(topic=topic)
    
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

#uncompiled
uncompiled_tweet_writer = TweetWriter()
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=4, display_progress=True, display_table=5, provide_traceback=True)
evaluator(uncompiled_tweet_writer) 

#BootstrapFewShot
config = dict(max_bootstrapped_demos=25, max_labeled_demos=4)
teleprompter = BootstrapFewShot(metric=creativity_metric, **config)
optimized_tweet_writer = teleprompter.compile(TweetWriter(), trainset=trainset[:10])
dspy.settings.configure(backoff_time=5)
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(optimized_tweet_writer)
optimized_tweet_writer.save(path="fewshot")

#BootstrapFewShotWithRandomSearch
config = dict(max_labeled_demos = 1, max_bootstrapped_demos=1, num_candidate_programs=2)
teleprompter = BootstrapFewShotWithRandomSearch(metric = creativity_metric, **config)
rs_optimized_tweet_writer = teleprompter.compile(TweetWriter(), trainset=trainset[:20])
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(rs_optimized_tweet_writer)
rs_optimized_tweet_writer.save(path="fewshotwithrs")

#MIPRO

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
evaluator = Evaluate(devset=devset[:10], metric=creativity_metric, num_threads=2, display_progress=True, display_table=5, provide_traceback=True)
evaluator(mipro_optimized_program)

