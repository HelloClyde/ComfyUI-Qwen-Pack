import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from folder_paths import models_dir
from comfy import model_management, model_patcher

# init
qwen_model_folder_path = Path(models_dir) / 'qwen'
qwen_model_folder_path.mkdir(parents=True, exist_ok=True)

class QwenModel:
    def __init__(self, tokenizer, model, patcher):
        self.tokenizer = tokenizer
        self.model = model
        self.patcher = patcher
        
        # hook modelClass.device setter
        def set_value(self, new_value):
            pass
        model.__class__.device = property(fget=model.__class__.device.fget, fset=set_value)
        

class QwenPackModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_lst = []
        for folder in qwen_model_folder_path.iterdir():
            if folder.is_dir():
                config_file = folder / 'config.json'
                if config_file.is_file():
                    relative_path = str(folder.relative_to(qwen_model_folder_path))
                    model_lst.append(relative_path)
        return {
            "required": {
                "model_name": (model_lst, {}),
            }
        }
        
    RETURN_TYPES = ("QWEN_MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load_model"
    CATEGORY = "qwen_pack"
  
    def load_model(self, model_name):
        offload_device = torch.device('cpu')
        load_device = model_management.get_torch_device()
        model = AutoModelForCausalLM.from_pretrained(
            qwen_model_folder_path / model_name, 
            device_map=offload_device, 
            torch_dtype="auto", 
        )
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_folder_path / model_name)
        patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        
        return (QwenModel(tokenizer, model, patcher), )

class QwenPackQA:

    @classmethod
    def INPUT_TYPES(cls):
        DEFAULT_INSTRUCT = """你是一个翻译，将给定的中文翻译成英文。有如下要求：
1. 翻译要简明准确，不要说多余的废话
2. 如果输入的文本是空的，或者没有输入，你应该返回空
3. 如果输入的文本已经是英文了，你应该原样输出
4. 除了翻译结果外，不需要回答其他"""
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "system_instruction": ("STRING", {"default": DEFAULT_INSTRUCT, "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "a": ("STRING", {"defaultInput": True,}),
                "b": ("STRING", {"defaultInput": True,}),
                "c": ("STRING", {"defaultInput": True,}),
                "d": ("STRING", {"defaultInput": True,}),
                "e": ("STRING", {"defaultInput": True,}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("answer",)
    FUNCTION = "generate"
    CATEGORY = "qwen_pack"


    def generate(self, qwen_model, system_instruction, user_prompt, **kwargs):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt.format(**kwargs)},
        ]
        tokenizer = qwen_model.tokenizer
        model = qwen_model.model
        patcher = qwen_model.patcher
        model_management.load_model_gpu(patcher)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return (response, )




        

NODE_CLASS_MAPPINGS = {
    "QwenPackModelLoader": QwenPackModelLoader,
    "QwenPackQA": QwenPackQA,
}

