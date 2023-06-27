from typing import Any

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # type: ignore


class Model:
    def __init__(self, device: str = "cuda"):
        self.model = None
        self.device = device

    def __call__(self, prompt: str) -> str:
        if self.model is None:
            raise ValueError("model is not loaded")

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                # pad_token_id=self.tokenizer.pad_token_id,
                # bos_token_id=self.tokenizer.bos_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
            )

        output = self.tokenizer.decode(output_ids.tolist()[0])
        return output

    def load_rinna(self):
        logger.info("loading rinna/japanese-gpt-neox-3.6b")
        self.tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b").to(self.device)

    def load_mpt30b(self, load_in_4bit=False, load_in_8bit=False):
        logger.info("loading mofumofu/mpt30b")
        self.tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-30b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "mosaicml/mpt-30b",
            trust_remote_code=True,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )

    def load_orca_mini_3b(self, load_in_4bit=False, load_in_8bit=False):
        logger.info("loading orca-mini-3b")
        self.tokenizer = AutoTokenizer.from_pretrained("psmathur/orca_mini_3b")
        self.model = AutoModelForCausalLM.from_pretrained(
            "psmathur/orca_mini_3b",
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map="auto",
        )


if __name__ == "__main__":
    model = Model()
    model.load_orca_mini_3b()

    p_system = "You are an AI assistant that follows instruction extremely well. Help as much as you can."
    p_instruction = (
        "You broadcast on youtube. Provide interesting trivias to viewer. Occasionally pose questions to viewer"
    )
    p_input = ""

    if len(p_input) != 0:
        prompt = f"### System:\n{p_system}\n\n### User:\n{p_instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{p_system}\n\n### User:\n{p_instruction}\n\n### Response:\n"

    print(model(prompt))
