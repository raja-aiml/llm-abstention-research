import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer_and_model(model_name: str):
    """
    Shared loader for tokenizer/model with consistent pad token handling.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as pad for decoder-only models

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return tokenizer, model
