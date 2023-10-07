

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import sys
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser,default_data_collator
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import logging
from modeling_chatglm import ChatGLMForConditionalGeneration
from data import InstrutionDataset,InstrutionCollator
from arguments import ModelArguments, DataTrainingArguments, FinetuneArguments as TrainingArguments
from transformers.trainer_utils import is_main_process
import transformers
from datasets import load_dataset
from tokenization_chatglm import ChatGLMTokenizer


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

tokenizer = AutoTokenizer.from_pretrained('./chatGLM-6B', trust_remote_code=True)


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))     # 保存训练的参数args,其中TRAINING_ARGS_NAME是一个特定的文件名,用于保存训练的参数设置。
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }       # 仅保存adapter_model.bin是为了让结果模型更加轻量
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))


def main():
    writer = SummaryWriter()


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 从chatGLM-6B预训练模型加载ChatGLMForConditionalGeneration模型
    model = ChatGLMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    tokenizer = ChatGLMTokenizer.from_pretrained(model_args.tokenizer_name, trust_remote_code=True)
    model.gradient_checkpointing_enable()   # 设置gradient checkpoint来减少显存
    model.enable_input_require_grads()
    model.is_parallelizable = True          # why
    model.model_parallel = True
    model.lm_head = CastOutputToFloat(model.lm_head)
    model.config.use_cache = (
        False  
    )

    # setup peft
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        lora_alpha=32,  
        target_modules=["query_key_value"],
        inference_mode=False,
        r=training_args.lora_rank,
        lora_dropout=0.1,
        bias="none",
        fan_in_fan_out = False
    )
    # lora_alpha越小,表示使用更多原始表示的成分。这会保留更多模型原始的表达能力,但蒸馏效果较弱
    model = get_peft_model(model, peft_config)
    model = model.half()        # 将模型的数据类型转换为半精度浮点数
    model.print_trainable_parameters()


    ## data
    train_data = InstrutionDataset(     # 文本生成任务的训练数据集,包含输入序列和目标序列
        data_path = data_args.train_path)
    
    data_collator = InstrutionCollator( # 利用tokenizer处理batch中的数据,包括padding等操作
        tokenizer=tokenizer,
        max_len = training_args.max_len,
        max_input_len=training_args.max_input_len
    )


    # start train
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    

    trainer.train()
    writer.close()

    # save model
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
