import backoff  # for exponential backoff
import openai
from omegaconf import OmegaConf, ListConfig, DictConfig
import concurrent.futures

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(client,**kwargs):
    return client.completions.create(**kwargs)

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def chat_completions_with_backoff(client,**kwargs):
    return client.chat.completions.create(**kwargs)


class OpenAIModel:
    def __init__(self, client, model_name, sampling_params, tokenizer=None) -> None:
        self.client = client
        self.model_name = model_name
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        # self.stop = sampling_params.get('stop', ['\n\n'])
        # self.stop_token_ids = [self.tokenizer.tokenize(step_delim) for step_delim in self.stop]
        if isinstance(self.sampling_params, (ListConfig, DictConfig)):
            self.sampling_params = OmegaConf.to_container(self.sampling_params, resolve=True)
        print(f'Started OpenAI API with {client}, {model_name}, {sampling_params}')


    def test_connection(self):
        available_models = self.client.models.list()
        model_ids = [model.id for model in available_models.data]
        self.model_name = model_ids[0]
        assert self.model_name in model_ids, f"{self.model_name} not in available model list: {model_ids}"
        print("Check connection successful.")

    def generate(self, input_string):
        response = completions_with_backoff(self.client,
            model = self.model_name,
            prompt = input_string,
            **self.sampling_params
        )
        return response

    def generate_mini_batch(self, input_string_list):
        response = completions_with_backoff(self.client,
                                            model=self.model_name,
                                            prompt = input_string_list,
                                            **self.sampling_params)
        return response

    def thread_generate(self, messages_list):
        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
            responses = list(executor.map(self.generate, messages_list))
        return responses

    def batch_generate(self, messages_list):
        # with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        #     responses = list(executor.map(self.generate, messages_list))
        # # input("batch_generate--")

        """
        for data = [1,2,3,4,5,6,7,8,9,0], we'll get:
        conpletion_list = [
            Namespace(choices = [...](len() = n))
            ....
        ],len() = 10
        """
        from argparse import Namespace
        n = self.sampling_params['n']
        response_choices_list = []
        response = self.generate_mini_batch(messages_list)
        choices = response.choices
        for i in range(len(choices)//n):
            response_choices_list.append(choices[i*n: i*n + n])

        completion_list = []
        for choice_list in response_choices_list:
            completion_list.append(Namespace(**{"choices": choice_list}))

        return completion_list

    def ref_generate(self, input, outputs):
        query_all = []
        choices_all = []
        choice_idx = [0]
        for index, o in enumerate(outputs):
            choices_all.extend(o.choices)
            query_all.extend([input[index]] * len(o.choices))
            choice_idx.append(choice_idx[-1] + len(o.choices))

        inputs_list = []
        for query,choice in zip(query_all,choices_all):
            response = choice.text
            # tokens = choice.logprobs.tokens

            choice.q_pi = sum(choice.logprobs.token_logprobs)
            response_tokens_len = len(choice.logprobs.tokens)
            # for s_ids in self.stop_token_ids:
            #     if tokens[-len(s_ids):] == s_ids:
            #         choice.q_pi = sum(choice.logprobs.token_logprobs[:-len(s_ids)])
            #         response_tokens_len = len(choice.logprobs.tokens)-len(s_ids)
            inputs_list.append(query + response)

        # inputs_list = inputs_list[0]
        # reference_output = self.generate(inputs_list)
        # print(reference_output)
        # input()
        reference_outputs = self.batch_generate(inputs_list)
        for choice,reference_output in zip(choices_all,reference_outputs):
            ref_choice = reference_output.choices[0]
            reference_logprobs = ref_choice.logprobs.token_logprobs
            reference_logprobs = reference_logprobs[(-1) * response_tokens_len:-1]
            reference_logprobs = [-200 if i is None or i < -200 else i for i in reference_logprobs]

            q_ref = sum(reference_logprobs)
            choice.q_ref = q_ref
            choice.value_estimate = choice.q_pi - choice.q_ref

        for i in range(len(choice_idx)-1):
            outputs[i].choices = choices_all[choice_idx[i]: choice_idx[i+1]]

        return outputs
