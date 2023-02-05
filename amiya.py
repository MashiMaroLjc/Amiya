import re

from transformers import T5Tokenizer, T5ForConditionalGeneration

import config


def preprocess_dialogue_hist(dialogue_hist: list):
    """
    模型总共能大概能处理400个token字符后面表现不行
    保证prompt能有256个字符的空间
    所以对话历史的字符要保证在1000-256个内
    :param dialogue_hist:
    :return:
    """
    # 先直接暴力切割，保留16轮
    if len(dialogue_hist) > 16:
        dialogue_hist = dialogue_hist[-16:]
    max_word = 1000 - 256
    total_length = 0
    new_dialogue_hist = []
    for text in dialogue_hist:
        new_dialogue_hist.append(text)
        total_length += len(text)
        while total_length > max_word:
            # 一次弹出一对，保证对话角色不会错
            for _ in range(2):
                drop = new_dialogue_hist.pop(0)
                total_length -= len(drop)
    return new_dialogue_hist


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")

    return text


def postprocess(text):
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    return re.sub("\n+", "\n", text)


def generate(input_text, tokenizer, model, temperature=0.7, top_k=0):
    text = preprocess(input_text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt")
    if config.device > -1:
        encoding = encoding.to(device=config.device)
    # print("[log]: input size:", list(encoding["input_ids"].size()))
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=256, min_length=1,
                         do_sample=True, top_p=1, repetition_penalty=1.1, top_k=top_k, temperature=temperature,
                         num_return_sequences=1)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    # print(out_text)
    response = postprocess(out_text[0])
    # if len(response) < 2:
    #     print("#debug:")
    #     print("input:", input_text)
    #     print("gen:", out_text)
    #     print("===========")
    return response


def get_response(prompt, dialogue_hist, tokenizer, model):
    character_name = config.character_name
    conversation = ["{tag}:{text}".format(tag="用户" if i % 2 == 0 else character_name, text=text) for i, text in
                    enumerate(prompt)] + \
                   ["{tag}:{text}".format(tag="用户" if i % 2 == 0 else character_name, text=text) for i, text in
                    enumerate(dialogue_hist)]
    input_text = "\n".join(conversation) + "\n{character_name}:".format(character_name=character_name)
    # print(f"input:\n{input_text}", )
    response = generate(input_text, tokenizer=tokenizer, model=model)
    return response


def postprocess_dialogue_hist(dialogue_hist: list):
    """
    模型总共能大概能处理400个token,约1000个字符，太长了表现不行
    保证prompt能有256个字符的空间
    所以对话历史的字符要保证在1000-256个内
    :param dialogue_hist:
    :return:
    """
    # 先直接暴力切割，保留16轮，大概够了
    if len(dialogue_hist) > 16:
        dialogue_hist = dialogue_hist[-16:]
    max_word = 1000 - 256
    total_length = 0
    new_dialogue_hist = []
    for text in dialogue_hist:
        new_dialogue_hist.append(text)
        total_length += len(text)
        while total_length > max_word:
            # 一次弹出一对，保证对话角色不会错
            for _ in range(2):
                drop = new_dialogue_hist.pop(0)
                total_length -= len(drop)
    return new_dialogue_hist


class Amiya:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(config.spm_model)
        self.model = T5ForConditionalGeneration.from_pretrained(config.model_checkpoint)
        if config.device > -1:
            self.model.to(device=config.device)
        self.model.eval()
        self.dialogue_hist = []
        self.preset_dialogue = []

    def get_response(self, text):
        self.dialogue_hist.append(text)
        response = get_response(self.preset_dialogue, self.dialogue_hist, self.tokenizer, self.model)
        self.dialogue_hist.append(response)
        self.dialogue_hist = postprocess_dialogue_hist(self.dialogue_hist)
        return response
