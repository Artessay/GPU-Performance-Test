import os
import torch
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_and_tokenizer(model_name_or_path):
    """
    load model and tokenizer
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        # low_cpu_mem_usage=True,
        # attn_implementation='flash_attention_2',
    )

    return model, tokenizer