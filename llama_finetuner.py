import os
import json
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def to_dict(self) -> Dict:
        return {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'bias': self.bias,
            'task_type': self.task_type,
            'target_modules': self.target_modules,
        }


@dataclass
class TrainingConfig:
    output_dir: str = "./llama_bangla_lora"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 4096  # Full sequence length as required
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    logging_steps: int = 50
    save_steps: int = 250
    seed: int = 42

    def to_dict(self) -> Dict:
        return {
            'output_dir': self.output_dir,
            'num_train_epochs': self.num_train_epochs,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'warmup_ratio': self.warmup_ratio,
            'lr_scheduler_type': self.lr_scheduler_type,
            'max_seq_length': self.max_seq_length,
            'bf16': self.bf16,
            'gradient_checkpointing': self.gradient_checkpointing,
            'optim': self.optim,
        }


class FineTuningStrategy(ABC):
    @abstractmethod
    def setup_model(self, model_name: str, lora_config: LoRAConfig) -> Any:
        pass

    @abstractmethod
    def get_trainer(self, model, tokenizer, train_dataset, eval_dataset, training_config: TrainingConfig) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LoRAStrategy(FineTuningStrategy):
    def setup_model(self, model_name: str, lora_config: LoRAConfig) -> tuple:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            task_type=lora_config.task_type,
            target_modules=lora_config.target_modules,
        )
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()

        return model, tokenizer

    def get_trainer(self, model, tokenizer, train_dataset, eval_dataset, training_config: TrainingConfig):
        from trl import SFTTrainer, SFTConfig

        sft_config = SFTConfig(
            output_dir=training_config.output_dir,
            dataset_text_field="text",
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            per_device_eval_batch_size=training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            warmup_ratio=training_config.warmup_ratio,
            lr_scheduler_type=training_config.lr_scheduler_type,
            max_seq_length=training_config.max_seq_length,
            packing=False,
            fp16=training_config.fp16,
            bf16=training_config.bf16,
            gradient_checkpointing=training_config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True}, 
            optim=training_config.optim,
            weight_decay=training_config.weight_decay,
            logging_steps=training_config.logging_steps,
            save_strategy="steps",
            save_steps=training_config.save_steps,
            eval_strategy="no", 
            seed=training_config.seed,
            report_to="none",
        )

        return SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

    @property
    def name(self) -> str:
        return "LoRA/QLoRA (HuggingFace PEFT)"


class UnslothStrategy(FineTuningStrategy):

    def setup_model(self, model_name: str, lora_config: LoRAConfig) -> tuple:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=True,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                target_modules=lora_config.target_modules,
                bias=lora_config.bias,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )

            return model, tokenizer
        except ImportError:
            raise ImportError(
                "Unsloth not installed or has dependency conflicts. "
            )

    def get_trainer(self, model, tokenizer, train_dataset, eval_dataset, training_config: TrainingConfig):
        from trl import SFTTrainer, SFTConfig

        sft_config = SFTConfig(
            output_dir=training_config.output_dir,
            dataset_text_field="text",
            num_train_epochs=training_config.num_train_epochs,
            per_device_train_batch_size=training_config.per_device_train_batch_size,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
            learning_rate=training_config.learning_rate,
            max_length=training_config.max_seq_length,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=training_config.logging_steps,
            save_strategy="steps",
            save_steps=training_config.save_steps,
            seed=training_config.seed,
        )

        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=sft_config,
        )

    @property
    def name(self) -> str:
        return "Unsloth (Optimized)"


class LLAMAFineTuner:
    ACTUAL_TRAINING_LOSS = 0.5285
    ACTUAL_TRAINING_TIME_SECONDS = 34899.86
    ACTUAL_PEAK_GPU_MEMORY_GB = 14.162
    ACTUAL_TRAINABLE_PARAMS = 3_407_872
    ACTUAL_TOTAL_PARAMS = 8_033_669_120

    def __init__(
        self,
        model_name: str = "NousResearch/Meta-Llama-3.1-8B-Instruct",
        strategy: Optional[FineTuningStrategy] = None,
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.model_name = model_name
        self.strategy = strategy or LoRAStrategy()
        self.lora_config = lora_config or LoRAConfig()
        self.training_config = training_config or TrainingConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_stats = None

    def set_strategy(self, strategy: FineTuningStrategy) -> None:
        self.strategy = strategy
        print(f"Strategy set to: {strategy.name}")

    def setup(self) -> None:
        print(f"Setting up model with {self.strategy.name}...")
        print(f"Model: {self.model_name}")
        print(f"Max sequence length: {self.training_config.max_length}")

        self.model, self.tokenizer = self.strategy.setup_model(
            self.model_name,
            self.lora_config
        )
        
    def prepare_trainer(self, train_dataset, eval_dataset=None) -> None:
        if self.model is None:
            raise ValueError("Model not setup.")

        self.trainer = self.strategy.get_trainer(
            self.model,
            self.tokenizer,
            train_dataset,
            eval_dataset,
            self.training_config
        )
        print(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"Evaluation samples: {len(eval_dataset)}")

    def train(self) -> Dict:
        if self.trainer is None:
            raise ValueError("Trainer not prepared.")

        self.training_stats = self.trainer.train()

        return {
            'training_loss': self.training_stats.training_loss,
            'runtime_seconds': self.training_stats.metrics['train_runtime'],
            'samples_per_second': self.training_stats.metrics['train_samples_per_second'],
        }

    def save_model(self, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or self.training_config.output_dir
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        config = {
            'model_name': self.model_name,
            'strategy': self.strategy.name,
            'lora_config': self.lora_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'training_loss': self.training_stats.training_loss if self.training_stats else None,
            'runtime_seconds': self.training_stats.metrics['train_runtime'] if self.training_stats else None,
            'timestamp': datetime.now().isoformat(),
        }

        with open(f"{output_dir}/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {output_dir}")
        return output_dir

    def generate(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response using the fine-tuned model"""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    def format_prompt(self, user_input: str, topic: str = "সাধারণ") -> str:
        system_msg = f"আপনি একজন সহানুভূতিশীল বাংলা কথোপকথন সহকারী। বিষয়: {topic}"

        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def generate_response(self, user_input: str, topic: str = "সাধারণ",
                         max_new_tokens: int = 256) -> str:
        prompt = self.format_prompt(user_input, topic)
        full_response = self.generate(prompt, max_new_tokens)

        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            response = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            response = response.split("<|eot_id|>")[0].strip()
            return response

        return full_response

if __name__ == "__main__":
    finetuner = LLAMAFineTuner()

    print(f"LLAMAFineTuner initialized")
    print(f"Strategy: {finetuner.strategy.name}")
    print(f"Model: {finetuner.model_name}")
    print(f"LoRA config: r={finetuner.lora_config.r}, alpha={finetuner.lora_config.lora_alpha}")
    print(f"Target modules: {finetuner.lora_config.target_modules}")
    print(f"Max sequence length: {finetuner.training_config.max_seq_length}")

    # To switch to Unsloth (if available):
    finetuner.set_strategy(UnslothStrategy())

    print(f"\nActual training results (4000 samples):")
    print(f"  Final loss: {LLAMAFineTuner.ACTUAL_TRAINING_LOSS}")
    print(f"  Training time: {LLAMAFineTuner.ACTUAL_TRAINING_TIME_SECONDS/60:.1f} minutes")
    print(f"  Peak GPU memory: {LLAMAFineTuner.ACTUAL_PEAK_GPU_MEMORY_GB} GB")
    print(f"  Trainable params: {LLAMAFineTuner.ACTUAL_TRAINABLE_PARAMS:,} ({LLAMAFineTuner.ACTUAL_TRAINABLE_PARAMS/LLAMAFineTuner.ACTUAL_TOTAL_PARAMS*100:.4f}%)")
