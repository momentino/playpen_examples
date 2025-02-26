import re
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from playpen.agents.base_agent import Agent
from playpen.backends.utils import ensure_alternating_roles, ContextExceededError
from src.utils.logger import file_logger, out_logger

class HFAgent(Agent):
    def __init__(self,
                 model_name: str,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 **gen_kwargs: Dict[str, Any]):
        super().__init__(name=model_name.split("/")[-1])
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", torch_dtype="auto",
                                                  verbose=False)
        self.gen_kwargs = gen_kwargs
        self.device = 'auto'

    def _check_context_limit(self, context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
        """
        Internal context limit check to run in generate_response.
        :param prompt_tokens: List of prompt token IDs.
        :param max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
        :return: Tuple with
                Bool: True if context limit is not exceeded, False if too many tokens
                Number of tokens for the given messages and maximum new tokens
                Number of tokens of 'context space left'
                Total context token limit
        """
        prompt_size = len(prompt_tokens)
        tokens_used = prompt_size + int(max_new_tokens)  # context includes tokens to be generated
        tokens_left = context_size - tokens_used
        fits = tokens_used <= context_size
        return fits, tokens_used, tokens_left, context_size

    def generate_response(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param return_full_text: If True, whole input context is returned.
        :param log_messages: If True, raw and cleaned messages passed will be logged.
        :return: the continuation
        """


        # log current given messages list:
        if log_messages:
            file_logger.info(f"Raw messages passed: {messages}")

        current_messages = ensure_alternating_roles(messages)

        # log current flattened messages list:
        if log_messages:
            file_logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)

        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {"inputs": prompt_text, "max_new_tokens": self.max_new_tokens,
                  "temperature": self.temperature, "return_full_text": return_full_text}

        """# check context limit:
        context_check = self._check_context_limit(self.context_size, prompt_tokens[0],
                                             max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            file_logger.info(f"Context token limit for {self.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])"""

        # greedy decoding:
        do_sample: bool = False
        if float(self.temperature) > 0.0:
            do_sample = True

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            """if 'output_split_prefix' in self.model_spec:
                response_text = model_output.rsplit(self.model_spec['output_split_prefix'], maxsplit=1)[1]

            # remove eos token string:
            eos_to_cull = self.model_spec['eos_to_cull']
            response_text = re.sub(eos_to_cull, "", response_text)"""
        else:
            response_text = model_output.strip()

        if log_messages:
            file_logger.info(f"Response message: {response_text}")
        print(" RESPONSE TEXT ", response_text)
        return prompt, response, response_text

    def act(self) -> Tuple[Any, Any, str]:
        prompt, response, response_text = self.generate_response(self.observations)
        return prompt, response, response_text

    def observe(self, observation, reward, termination, truncation, info):
        self.observations.append(observation)

    def shutdown(self):
        pass