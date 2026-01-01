import re
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessingConfig:
    min_question_length: int = 10
    min_answer_length: int = 20
    max_char_length: int = 5000  # Prevents OOM on long sequences
    max_train_samples: int = 4000
    max_val_samples: int = 300
    max_test_samples: int = 300
    random_seed: int = 42

    def to_dict(self) -> Dict:
        return {
            'min_question_length': self.min_question_length,
            'min_answer_length': self.min_answer_length,
            'max_char_length': self.max_char_length,
            'max_train_samples': self.max_train_samples,
            'max_val_samples': self.max_val_samples,
            'max_test_samples': self.max_test_samples,
            'random_seed': self.random_seed,
        }


class ConversationFormatter(ABC):

    @abstractmethod
    def format(self, topic: str, question_title: str, question: str, answer: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class LLaMAInstructFormatter(ConversationFormatter):
    def format(self, topic: str, question_title: str, question: str, answer: str) -> str:
        system_msg = f"আপনি একজন সহানুভূতিশীল বাংলা কথোপকথন সহকারী। বিষয়: {topic}"

        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question_title}

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer}<|eot_id|><|end_of_text|>"""

        return formatted

    @property
    def name(self) -> str:
        return "LLaMA-3.1-Instruct"


class ChatMLFormatter(ConversationFormatter):
    def format(self, topic: str, question_title: str, question: str, answer: str) -> str:
        system_msg = f"আপনি একজন সহানুভূতিশীল বাংলা কথোপকথন সহকারী। বিষয়: {topic}"

        formatted = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{question_title}

{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>"""

        return formatted

    @property
    def name(self) -> str:
        return "ChatML"


class AlpacaFormatter(ConversationFormatter):
    def format(self, topic: str, question_title: str, question: str, answer: str) -> str:
        return f"""### Instruction:
আপনি একজন সহানুভূতিশীল বাংলা কথোপকথন সহকারী। বিষয়: {topic}

### Input:
{question_title}

{question}

### Response:
{answer}"""

    @property
    def name(self) -> str:
        return "Alpaca"


class DatasetProcessor:
    ACTUAL_RAW_SAMPLES = 38233
    ACTUAL_VALID_SAMPLES = 33731
    ACTUAL_TRAIN_SAMPLES = 4000
    ACTUAL_VAL_SAMPLES = 300
    ACTUAL_TEST_SAMPLES = 300

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        formatter: Optional[ConversationFormatter] = None
    ):
        self.config = config or ProcessingConfig()
        self.formatter = formatter or LLaMAInstructFormatter()
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.df_filtered: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def set_formatter(self, formatter: ConversationFormatter) -> None:
        self.formatter = formatter
        print(f"Formatter set to: {formatter.name}")

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def load_data(self, file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
        self.df_raw = pd.read_csv(file_path, encoding=encoding)
        print(f"Dataset shape: {self.df_raw.shape}")
        print(f"Column Names: {self.df_raw.columns.tolist()}")
        return self.df_raw

    def preprocess(self) -> pd.DataFrame:
        if self.df_raw is None:
            raise ValueError("No data loaded. Call load_data()")
        df = self.df_raw.copy()

        df = df.dropna(subset=['Questions', 'Answers'])

        df = df[df['Questions'].str.len() > self.config.min_question_length]
        df = df[df['Answers'].str.len() > self.config.min_answer_length]
        df['text'] = df.apply(
            lambda row: self.formatter.format(
                topic=self.clean_text(row['Topics']),
                question_title=self.clean_text(row['Question-Title']),
                question=self.clean_text(row['Questions']),
                answer=self.clean_text(row['Answers'])
            ),
            axis=1
        )

        self.df_processed = df
        print(f"Number of valid samples: {len(df)}")
        return df

    def filter_by_length(self) -> pd.DataFrame:
        if self.df_processed is None:
            raise ValueError("No processed data. Call preprocess()")

        self.df_filtered = self.df_processed[
            self.df_processed['text'].str.len() < self.config.max_char_length
        ].copy()

        print(f"After length filter: {len(self.df_filtered)} samples")
        return self.df_filtered

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df_filtered is None:
            if self.df_processed is None:
                raise ValueError("No processed data. Call preprocess()")
            self.df_filtered = self.df_processed

        df = self.df_filtered.sample(
            frac=1,
            random_state=self.config.random_seed
        ).reset_index(drop=True)

        self.train_df = df[:self.config.max_train_samples]
        self.val_df = df[
            self.config.max_train_samples:
            self.config.max_train_samples + self.config.max_val_samples
        ]
        self.test_df = df[
            self.config.max_train_samples + self.config.max_val_samples:
            self.config.max_train_samples + self.config.max_val_samples + self.config.max_test_samples
        ]

        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")
        return self.train_df, self.val_df, self.test_df

    def save_jsonl(self, df: pd.DataFrame, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                f.write(json.dumps({'text': row['text']}, ensure_ascii=False) + '\n')
        print(f"Saved {len(df)} samples to {path}")

    def save_splits(self, output_dir: str = ".") -> Dict[str, str]:
        paths = {}

        if self.train_df is not None:
            train_path = f"{output_dir}/train.jsonl"
            self.save_jsonl(self.train_df, train_path)
            paths['train'] = train_path

        if self.val_df is not None:
            val_path = f"{output_dir}/val.jsonl"
            self.save_jsonl(self.val_df, val_path)
            paths['val'] = val_path

        if self.test_df is not None:
            test_path = f"{output_dir}/test.jsonl"
            self.save_jsonl(self.test_df, test_path)
            paths['test'] = test_path

        return paths

    def get_statistics(self) -> Dict:
        stats = {
            'raw_samples': len(self.df_raw) if self.df_raw is not None else 0,
            'processed_samples': len(self.df_processed) if self.df_processed is not None else 0,
            'filtered_samples': len(self.df_filtered) if self.df_filtered is not None else 0,
            'train_samples': len(self.train_df) if self.train_df is not None else 0,
            'val_samples': len(self.val_df) if self.val_df is not None else 0,
            'test_samples': len(self.test_df) if self.test_df is not None else 0,
            'formatter': self.formatter.name,
            'config': self.config.to_dict(),
        }
        return stats

    def process_pipeline(self, file_path: str, output_dir: str = ".") -> Dict:
        print("=" * 60)
        print("DATASET PROCESSING PIPELINE")
        print("=" * 60)

        print("\n1. Loading data...")
        self.load_data(file_path)

        print("\n2. Preprocessing...")
        self.preprocess()

        print("\n3. Filtering by length...")
        self.filter_by_length()

        print("\n4. Splitting data...")
        self.split_data()

        print("\n5. Saving splits...")
        paths = self.save_splits(output_dir)

        stats = self.get_statistics()
        stats['output_paths'] = paths
        stats['timestamp'] = datetime.now().isoformat()

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)

        return stats


if __name__ == "__main__":
    processor = DatasetProcessor()

    print("DatasetProcessor initialized")
    print(f"Formatter: {processor.formatter.name}")
    print(f"Config: {processor.config.to_dict()}")

    # Switch formatter:
    processor.set_formatter(ChatMLFormatter())
    processor.set_formatter(AlpacaFormatter())

    print(f"\nActual training run statistics:")
    print(f"Raw samples: {DatasetProcessor.ACTUAL_RAW_SAMPLES:,}")
    print(f"Valid samples: {DatasetProcessor.ACTUAL_VALID_SAMPLES:,}")
    print(f"Train samples: {DatasetProcessor.ACTUAL_TRAIN_SAMPLES:,}")
    print(f"Validation samples: {DatasetProcessor.ACTUAL_VAL_SAMPLES}")
    print(f"Test samples: {DatasetProcessor.ACTUAL_TEST_SAMPLES}")

