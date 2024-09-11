import torch
from vllm import LLM, SamplingParams
from config import CKPT_ID, PROMPT


def load_vllm_engine(max_tokens=120):
    devices = torch.cuda.device_count()
    vllm_engine = LLM(CKPT_ID, tensor_parallel_size=devices)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    return vllm_engine, sampling_params


def infer(vllm_engine, inputs):
    sampling_params = inputs.pop("sampling_params")
    vllm_inputs = [{"prompt": PROMPT, "multi_modal_data": {"image": image}} for image in inputs["original_images"]]
    outputs = vllm_engine.generate(vllm_inputs, sampling_params)
    return outputs
