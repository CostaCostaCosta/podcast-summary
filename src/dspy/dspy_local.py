import os

OPENAI_API_HOST = "http://127.0.0.1:5000"
OPENAI_API_KEY = "sk-111111111111111111111111111111111111111111111111"
OPENAI_API_BASE = "http://127.0.0.1:5000/v1"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_HOST"] = OPENAI_API_HOST
os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate

# Set up the LM
# turbo = dspy.OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=250)
turbo = dspy.OpenAI(
    model="TheBloke/dolphin-2.0-mistral-7B-GGUF", api_key=OPENAI_API_KEY, max_tokens=250
)
dspy.settings.configure(lm=turbo)

# Load math questions from the GSM8K dataset
gms8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gms8k.train[:10], gms8k.dev[:10]


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

# Optimize! Use the `gms8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(
    CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset
)

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(
    devset=gsm8k_devset,
    metric=gsm8k_metric,
    num_threads=4,
    display_progress=True,
    display_table=0,
)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

turbo.inspect_history(n=1)
