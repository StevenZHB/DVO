import os
import time

from typing import Optional, Any, Dict, List, Callable, Type, Tuple

from vllm import LLM, SamplingParams
from vllm.outputs import CompletionOutput, RequestOutput


def local_vllm(
    prompt: str,
    llm: LLM,
    sampling_params: SamplingParams,
    n: int,
    temperature: float,
) -> List[str]:
    """
    This one is not for batch inference.

    Don't use this function for value
    """
    # update args
    sampling_params.n = n
    sampling_params.temperature = temperature
    # n samples for each prompt
    prompts = [prompt]
    outputs = llm.generate(prompts, sampling_params=sampling_params)    # return List[RequestOutput]
    # len(prompts) = 1,  we take the first one RequestOutput.
    output = outputs[0]
    completion_outputs = output.outputs                                 # return List[CompletionOutput], where len() = sampling_params.n
    return [co.text for co in completion_outputs]


def server_generator(
    prompts: List[str],
    engine: Any,
):
    vllm_outputs = []
    for prompt in prompts:
        responses = engine(prompt)
        output = RequestOutput(request_id=str(time.time()),
                               prompt=prompt,
                               prompt_token_ids=[],
                               prompt_logprobs=-1,
                               outputs=[CompletionOutput(index=idx, text=response, token_ids=[], cumulative_logprob=-1, logprobs=-1)
                                        for idx, response in enumerate(responses)],
                               finished=True)
        vllm_outputs.append(output)
    return vllm_outputs



def local_server_generator(
    prompts: List[str],
    sampling_params: SamplingParams,
    port: int=8000
):

    from openai import OpenAI
    pi_api_key = "EMPTY"
    pi_api_base = f"http://0.0.0.0:{port}/v1"

    client = OpenAI(
        api_key=pi_api_key,
        base_url=pi_api_base,
    )

    n = sampling_params.n
    max_tokens = sampling_params.max_tokens
    temperature = sampling_params.temperature
    vllm_models = client.models.list()
    pi_model = vllm_models.data[0].id
    completion = client.completions.create(
        model=pi_model,
        prompt=prompts,
        n=n,
        max_tokens=max_tokens,
        logprobs=0,
        temperature=temperature
    )
    return completion



