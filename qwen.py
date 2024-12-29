import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from folder_paths import models_dir
from comfy import model_management, model_patcher
from .utils import set_seed, tensor_to_pil_images, resize_tensor_images

# init
qwen_model_folder_path = Path(models_dir) / 'qwen'
qwen_model_folder_path.mkdir(parents=True, exist_ok=True)
qwen_vl_model_folder_path = Path(models_dir) / 'qwen-vl'
qwen_vl_model_folder_path.mkdir(parents=True, exist_ok=True)

class QwenModel:
    def __init__(self, model, patcher, tokenizer=None, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
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
        
        return (QwenModel(model, patcher, tokenizer=tokenizer), )

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 512, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
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


    def generate(self, qwen_model, system_instruction, user_prompt, seed=0, temperature=1.0, max_tokens=512, top_k=50, top_p=1.0, **kwargs):
        set_seed(seed % 9999999)
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
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return (response, )


class QwenPackVLModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        model_lst = []
        for folder in qwen_vl_model_folder_path.iterdir():
            if folder.is_dir():
                config_file = folder / 'config.json'
                if config_file.is_file():
                    relative_path = str(folder.relative_to(qwen_vl_model_folder_path))
                    model_lst.append(relative_path)
        return {
            "required": {
                "model_name": (model_lst, {}),
            }
        }
        
    RETURN_TYPES = ("QWEN_VL_MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load_model"
    CATEGORY = "qwen_pack"
  
    def load_model(self, model_name):
        offload_device = torch.device('cpu')
        load_device = model_management.get_torch_device()
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            qwen_vl_model_folder_path / model_name, 
            device_map=offload_device, 
            torch_dtype="auto", 
        )
        processor = AutoProcessor.from_pretrained(qwen_vl_model_folder_path / model_name)
        patcher = model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        
        return (QwenModel(model, patcher, processor=processor), )
    

class QwenPackVQA:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_vl_model": ("QWEN_VL_MODEL",),
                "images": ("IMAGE", {}),
                "user_prompt": ("STRING", {"default": "Describe this image.", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_resolution": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "max_tokens": ("INT", {"default": 512, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
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


    def generate(self, qwen_vl_model, user_prompt, images, seed=0, max_resolution=512, temperature=1.0, max_tokens=512, top_k=50, top_p=1.0, **kwargs):
        set_seed(seed % 9999999)
        processor = qwen_vl_model.processor
        model = qwen_vl_model.model
        patcher = qwen_vl_model.patcher
        model_management.load_model_gpu(patcher)
        
        msg_content = []
        pil_imgs = []
        
        if images is not None:
            N, H, W, C = images.shape
            for i in range(N):
                msg_content.append({ "type": "image"})
            resize_imgs = resize_tensor_images(images, max_resolution)
            pil_imgs += tensor_to_pil_images(resize_imgs)
            
        
        msg_content.append({"type": "text", "text": user_prompt.format(**kwargs)})
        conversation = [
            {
                "role": "user",
                "content":msg_content
            }
        ]

        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(
            text=[text_prompt], images=pil_imgs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=max_tokens, top_k=top_k, top_p=top_p, temperature=temperature)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return (output_text, )

        

NODE_CLASS_MAPPINGS = {
    "QwenPackModelLoader": QwenPackModelLoader,
    "QwenPackQA": QwenPackQA,
    "QwenPackVLModelLoader": QwenPackVLModelLoader,
    "QwenPackVQA": QwenPackVQA,
}

