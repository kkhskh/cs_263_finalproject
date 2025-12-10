# victim_service/model_backend.py

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class HFModelBackend:
    def __init__(self, model_name: str, quant_mode: str):

        print(f"[ModelBackend] Loading model {model_name!r} ({quant_mode}) onto GPU...")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # load model onto GPU
        self.model = self._load_model(model_name, quant_mode)
        self.model.eval()

        print("[ModelBackend] Model loaded successfully.")

    def generate(self, prompt: str, steps: int | None = None) -> str:
        # tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        # run inference
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                use_cache=True # important to use KV-cache
            )

        # decode output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _load_model(self, model_name: str, quant_mode: str):

        # support fp16 or 8bit quantization
        if quant_mode == "fp16":
            bnb_config = None
        elif quant_mode == "q-8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unknown quantization mode: {quant_mode}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        # only move to cuda if not using bitsandbytes
        if bnb_config is None:
            model = model.to("cuda")

        return model
        