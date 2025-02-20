#%%
from utils_api import OpenAIModel
import openai
policy_client = openai.OpenAI(
        api_key="API_KEY",
        base_url = "http://0.0.0.0:9554/v1"
    )

models = policy_client.models.list()
model = models.data[0].id

policy_sampling_params = {
    "temperature": 0.8,
    "top_p": 1,
    # "use_beam_search": True,
    "best_of": 5,
    "max_tokens": 1024,
    "n": 5,
    "stop": [],
    "echo": True,
    "logprobs": 1

}

ref_sampling_params = {
    "temperature": 0,
    # "top_k": config.top_k,
    "top_p": 1,
    # "use_beam_search": config.use_beam_search,
    "best_of": 1,
    "max_tokens": 1,
    "echo": True,
    "logprobs": 1,
    "n": 1,
    "stop": ['\n\n']
}


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
        "/zhanghongbo/Super_MARIO_STEVEN/build_deepstepmath/deepseekmath-instruct-step-sft/deepseekmath-instruct-pattern-sft",
        trust_remote_code=True
    )

# policy = OpenAIModel(policy_client, model_name=model,sampling_params=ref_sampling_params, tokenizer=tokenizer)

llm = OpenAIModel(policy_client, model_name=model,sampling_params=ref_sampling_params, tokenizer=tokenizer)

#%%
prompts = ["[INST]who are you?[/INST]I'm a robot."]

result = llm.generate(prompts)
print(result)
print(len(result.choices))
print(result.choices[0].text)


#%%
a = {'a': [1,2,1,1,1,1,1], 'b': [2,2,1,2,2,2,2,2,2,22]}
from argparse import Namespace
a = Namespace(**{'a': [1,2,1,1,1,1,1], 'b': [2,2,1,2,2,2,2,2,2,22]})
print(a.b)

# %%
