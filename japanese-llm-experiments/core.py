import os
from typing import Any

import torch
from loguru import logger
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
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

    def load_rwkv4_world(self, use_fp16=False):
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "0"

        d_type = "fp16" if use_fp16 else "fp32"

        self.model = RWKV(
            model="japanese-llm-experiments/models/RWKV-4-World-7B-v1-20230626-ctx4096", strategy=f"cuda {d_type}"
        )
        self.pipeline = PIPELINE(self.model, "rwkv_vocab_v20230424")

    def rwkv_inference(self, instruction, input=None):
        args = PIPELINE_ARGS(
            temperature=1.0,
            top_p=0.3,
            top_k=100,
            alpha_frequency=0.25,
            alpha_presence=0.25,
            token_ban=[],
            token_stop=[0],
            chunk_len=256,
        )

        if input:
            prompt = f"""Instruction: {instruction}

Input: {input}

Response: """
        else:
            prompt = f"""Question: {instruction}

Answer: """

        result = self.pipeline.generate(prompt, token_count=200, args=args)
        return result


if __name__ == "__main__":
    model = Model()
    model.load_rwkv4_world()

    instruct = "面白いトリビアを教えてください？"
    result = model.rwkv_inference(instruct)
    print(result)
