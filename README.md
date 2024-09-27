Using DSPy to optimize some custom LLM-judged metrics on auto-generated tweets.

The verbose outputs in the notebook might be hard to sift through, so I've also put the instrumental code in main.py (you might need to move around some code to run it) and am putting the results here:

## Annotation
Since we just had the tweets and no labels of any sort, I chose to annotate the tweets with a topic using gpt-4o so that we'd have context for generation.

## Experiment 1: simple 'creativity' metric (1-5)
Using a 'creativity' metric with DSPy, simply instructing it to rate the creativity of the generated tweets from 1 to 5. 
Resulted in most generations being scored 3, making it hard to differentiate between methods. But these were the outputs:

1) Uncompiled program: 320
2) BootstrapFewShot: 320 `(max_bootstrapped_demos=25, max_labeled_demos=4, trainset[:10], running time ~= 3 minutes)`
3) BootstrapFewShotWithRandomSearch: 290 `(max_labeled_demos = 1, max_bootstrapped_demos=1, num_candidate_programs=2, trainset[:20], running time ~= 10 minutes)`
4) MIPROv2: 330 `(num_candidates=7, init_temperature=0.5, max_bootstrapped_demos=3, max_labeled_demos=4, num_trials=10 minibatch_size=25, running time ~= 20 minutes)`

Here are some example generations:
| Type                             | Input                   | Output                                                                                                                                                                          | Score |
|----------------------------------|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| BootstrapFewShot                 | Learning and Experience | Learning gives you knowledge, but experience gives you wisdom. Embrace both, and you'll navigate life with confidence and insight. 📚🌍 #Learning #Experience                     | ✔️ [3] |
| BootstrapFewShotWithRandomSearch | Learning and Experience | Never stop learning and seeking new experiences! They are the building blocks of growth and success. Embrace every opportunity to expand your horizons. 🌟📚 #LifelongLearning... | ✔️ [3] |
| MIPROv2                          | Learning and Experience | Learning gives you the map, but experience is the journey. 🌍📚 Embrace both to navigate life’s challenges and grow continuously. #LifelongLearning #ExperienceMatters            | ✔️ [4] |

## Experiment 2: ensemble metric of binary relevant + engaging + creative + concise
Used a more normalized metric - and ensemble of binary 'yes/no' between each of `relevant + engaging + creative + concise`.

1) Uncompiled program: 96.25
2) BootstrapFewShot: 100 `(max_bootstrapped_demos=15, max_labeled_demos=5, trainset[:10], running time ~= 1 minute)`
3) BootstrapFewShotWithRandomSearch: 95 `(max_labeled_demos = 1, max_bootstrapped_demos=1, num_candidate_programs=2, trainset[:20], running time ~= 2 minutes)`
4) MIPROv2: 100 `(num_candidates=7, init_temperature=0.5, max_bootstrapped_demos=3, max_labeled_demos=4, num_trials=10 minibatch_size=25, running time ~= 30 minutes)`

| Type                             | Input               | Output                                                                                                                                          | Score   |
|----------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| BootstrapFewShot                 | Everyday Life/Humor | Why did the scarecrow win an award? Because he was outstanding in his field! 🌾😄 Remember to find humor in everyday life—it makes the mundane... | ✔️ [1.0]   |
| BootstrapFewShotWithRandomSearch | Everyday Life/Humor | Why do we park in driveways and drive on parkways? 🤔 Life is full of these little mysteries! Embrace the humor in everyday life—it makes... | ✔️ [1.0] |
| MIPROv2                          | Everyday Life/Humor | Why do we park in driveways and drive on parkways? 🤔 Life is full of these little mysteries! Embrace the humor in everyday life—it makes...     | ✔️ [1.0] |

## Experiment 3: ensemble metric of binary relevant + engaging + creative + concise, with RAG
For the third experiment, I decided to set aside some of our dataset and use it for RAG instead of use the whole thing for dev. Tweets for RAG: 200. Tweets for training: 75. Tweets for val: 25.

1) Uncompiled program: 91.2
2) BootstrapFewShot: 99.2 `(max_bootstrapped_demos=15, max_labeled_demos=5, trainset[:10], running time ~= 1 minute)`
3) BootstrapFewShotWithRandomSearch: 95 `(max_labeled_demos = 2, max_bootstrapped_demos=2, num_candidate_programs=3, trainset[:20], running time ~= 2 minutes)`
4) MIPROv2: 91.2 `(max_bootstrapped_demos=3, max_labeled_demos=4, num_trials=5, minibatch=True, minibatch_size=5,minibatch_full_eval_steps=2, running time ~= 15 minutes)`

## Experiment 4: Second-order optimization, or metric of metric
For the fourth experiment, I manually defined some examples of how I would like the evaluator to work, then compiled the metric as a DSPy program so that it works more in line with what I expect from it.