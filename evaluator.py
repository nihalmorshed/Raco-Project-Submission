import json
import sqlite3
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime



class ExperimentLogger:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS LLAMAExperiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                lora_config TEXT,
                train_loss REAL,
                val_loss REAL,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS GeneratedResponses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                input_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES LLAMAExperiments(id)
            )
        """)

        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")

    def log_experiment(
        self,
        model_name: str,
        lora_config: Dict,
        train_loss: float,
        val_loss: Optional[float],
        metrics: Dict
    ) -> int:
        """Log an experiment and return experiment ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO LLAMAExperiments (model_name, lora_config, train_loss, val_loss, metrics)
            VALUES (?, ?, ?, ?, ?)
        """, (
            model_name,
            json.dumps(lora_config),
            train_loss,
            val_loss,
            json.dumps(metrics)
        ))

        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"Logged experiment ID: {experiment_id}")
        return experiment_id

    def log_response(
        self,
        experiment_id: int,
        input_text: str,
        response_text: str
    ) -> int:
        """Log a generated response"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO GeneratedResponses (experiment_id, input_text, response_text)
            VALUES (?, ?, ?)
        """, (experiment_id, input_text, response_text))

        response_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return response_id

    def log_batch_responses(
        self,
        experiment_id: int,
        responses: List[Dict[str, str]]
    ) -> List[int]:
        """Log multiple responses in a single transaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        response_ids = []
        for resp in responses:
            cursor.execute("""
                INSERT INTO GeneratedResponses (experiment_id, input_text, response_text)
                VALUES (?, ?, ?)
            """, (experiment_id, resp.get('input', ''), resp.get('response', '')))
            response_ids.append(cursor.lastrowid)

        conn.commit()
        conn.close()

        return response_ids

    def get_experiment(self, experiment_id: int) -> Optional[Dict]:
        """Get experiment by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM LLAMAExperiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'id': row[0],
                'model_name': row[1],
                'lora_config': json.loads(row[2]) if row[2] else None,
                'train_loss': row[3],
                'val_loss': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6]
            }
        return None

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM LLAMAExperiments ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'model_name': row[1],
                'lora_config': json.loads(row[2]) if row[2] else None,
                'train_loss': row[3],
                'val_loss': row[4],
                'metrics': json.loads(row[5]) if row[5] else None,
                'timestamp': row[6]
            }
            for row in rows
        ]

    def get_responses(self, experiment_id: int) -> List[Dict]:
        """Get all responses for an experiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM GeneratedResponses WHERE experiment_id = ?",
            (experiment_id,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'id': row[0],
                'experiment_id': row[1],
                'input_text': row[2],
                'response_text': row[3],
                'timestamp': row[4]
            }
            for row in rows
        ]


class MetricCalculator(ABC):

    @abstractmethod
    def calculate(self, predictions: List[str], references: List[str]) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BLEUCalculator(MetricCalculator):
    ACTUAL_SCORE = 0.81

    def calculate(self, predictions: List[str], references: List[str]) -> float:
        try:
            import evaluate
            bleu = evaluate.load("sacrebleu")
            refs_formatted = [[ref] for ref in references]
            result = bleu.compute(predictions=predictions, references=refs_formatted)
            return result['score']
        except Exception as e:
            print(f"BLEU calculation error: {e}")
            return 0.0

    @property
    def name(self) -> str:
        return "BLEU"


class ROUGECalculator(MetricCalculator):

    ACTUAL_SCORES = {
        'rouge1': 0.0000,
        'rouge2': 0.0000,
        'rougeL': 0.0000
    }

    def calculate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        try:
            import evaluate
            rouge = evaluate.load("rouge")
            result = rouge.compute(predictions=predictions, references=references)
            return {
                'rouge1': result['rouge1'],
                'rouge2': result['rouge2'],
                'rougeL': result['rougeL']
            }
        except Exception as e:
            print(f"ROUGE calculation error: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    @property
    def name(self) -> str:
        return "ROUGE"


class PerplexityCalculator(MetricCalculator):
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def calculate(self, predictions: List[str], references: List[str] = None) -> float:
        """Calculate perplexity on given texts"""
        if self.model is None or self.tokenizer is None:
            print("Model/tokenizer not set for perplexity calculation")
            return float('inf')

        try:
            self.model.eval()
            total_loss = 0
            total_tokens = 0

            with torch.no_grad():
                for text in predictions[:10]:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)

            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            return float(perplexity)

        except Exception as e:
            print(f"Perplexity calculation error: {e}")
            return float('inf')

    @property
    def name(self) -> str:
        return "Perplexity"


class EmpathyScoreCalculator(MetricCalculator):
    
    ACTUAL_SCORE = 3.67

    EMPATHETIC_PATTERNS = [
        "বুঝতে পারছি",  # I understand
        "বুঝি",         # I understand
        "দুঃখিত",       # Sorry
        "কষ্ট",         # Pain/suffering
        "মন খারাপ",     # Sad
        "পাশে আছি",    # I'm here for you
        "সাহায্য",      # Help
        "চেষ্টা",       # Try
        "সাহস",        # Courage
        "আশা",         # Hope
        "স্বাভাবিক",    # Normal
        "ঠিক আছে",     # It's okay
        "চিন্তা করবেন না",  # Don't worry
        "একা নন"       # You're not alone
    ]

    def calculate(self, predictions: List[str], references: List[str] = None) -> float:
        scores = []
        for pred in predictions:
            matches = sum(1 for pattern in self.EMPATHETIC_PATTERNS if pattern in pred)
            score = min(100, (matches / len(self.EMPATHETIC_PATTERNS)) * 200)
            scores.append(score)
        return float(np.mean(scores))

    @property
    def name(self) -> str:
        return "Empathy Score (Keyword)"


class SemanticEmpathyCalculator(MetricCalculator):
    ACTUAL_SCORE = 19.33

    EMPATHETIC_PATTERNS = [
        "বুঝতে পারছি", "বুঝি", "দুঃখিত", "কষ্ট", "মন খারাপ",
        "পাশে আছি", "সাহায্য", "চেষ্টা", "সাহস", "আশা",
        "স্বাভাবিক", "ঠিক আছে"
    ]

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.model = None
        self.pattern_embeddings = None

    def _load_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
                self.pattern_embeddings = self.model.encode(
                    self.EMPATHETIC_PATTERNS,
                    convert_to_tensor=True
                )
            except ImportError:
                print("sentence-transformers not installed.")
                return False
        return True

    def calculate(self, predictions: List[str], references: List[str] = None) -> float:
        if not self._load_model():
            return 0.0

        try:
            from sentence_transformers import util

            scores = []
            for pred in predictions:
                pred_embedding = self.model.encode(pred, convert_to_tensor=True)
                similarities = util.cos_sim(pred_embedding, self.pattern_embeddings)[0]
                matches = (similarities > self.threshold).sum().item()
                score = min(100, (matches / len(self.EMPATHETIC_PATTERNS)) * 200)
                scores.append(score)

            return float(np.mean(scores))
        except Exception as e:
            print(f"Semantic empathy calculation error: {e}")
            return 0.0

    @property
    def name(self) -> str:
        return "Empathy Score (Semantic)"


class CombinedScoreCalculator(MetricCalculator):
    
    ACTUAL_SCORE = 0.320

    def __init__(self, w_bleu: float = 0.25, w_rouge: float = 0.15, w_empathy: float = 0.60):
        self.w_bleu = w_bleu
        self.w_rouge = w_rouge
        self.w_empathy = w_empathy

    def calculate(
        self,
        bleu: float,
        rouge_l: float,
        empathy: float
    ) -> float:
        empathy_norm = empathy / 100.0  
        return (self.w_bleu * bleu) + (self.w_rouge * rouge_l) + (self.w_empathy * empathy_norm)

    @property
    def name(self) -> str:
        return "Combined Quality Score"

@dataclass
class HumanEvaluation:
    response_id: int
    evaluator_id: str
    empathy_score: int  # 1-5
    relevance_score: int  # 1-5
    fluency_score: int  # 1-5
    overall_score: int  # 1-5
    comments: str = ""


class HumanEvaluationFramework:

    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS HumanEvaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER,
                evaluator_id TEXT,
                empathy_score INTEGER,
                relevance_score INTEGER,
                fluency_score INTEGER,
                overall_score INTEGER,
                comments TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (response_id) REFERENCES GeneratedResponses(id)
            )
        """)

        conn.commit()
        conn.close()

    def add_evaluation(self, evaluation: HumanEvaluation) -> int:
        """Add a human evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO HumanEvaluations
            (response_id, evaluator_id, empathy_score, relevance_score, fluency_score, overall_score, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation.response_id,
            evaluation.evaluator_id,
            evaluation.empathy_score,
            evaluation.relevance_score,
            evaluation.fluency_score,
            evaluation.overall_score,
            evaluation.comments
        ))

        eval_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return eval_id

    def get_evaluation_summary(self, experiment_id: int) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                AVG(he.empathy_score) as avg_empathy,
                AVG(he.relevance_score) as avg_relevance,
                AVG(he.fluency_score) as avg_fluency,
                AVG(he.overall_score) as avg_overall,
                COUNT(*) as num_evaluations
            FROM HumanEvaluations he
            JOIN GeneratedResponses gr ON he.response_id = gr.id
            WHERE gr.experiment_id = ?
        """, (experiment_id,))

        row = cursor.fetchone()
        conn.close()

        return {
            'avg_empathy': row[0] or 0,
            'avg_relevance': row[1] or 0,
            'avg_fluency': row[2] or 0,
            'avg_overall': row[3] or 0,
            'num_evaluations': row[4] or 0
        }



class Evaluator:
    ACTUAL_RESULTS = {
        'bleu': 0.81,
        'rouge1': 0.0000,
        'rouge2': 0.0000,
        'rougeL': 0.0000,
        'empathy_keyword': 3.67,
        'empathy_semantic': 19.33,
        'combined_score': 0.320,
        'training_loss': 0.5285,
    }

    def __init__(self, model=None, tokenizer=None, db_path: str = "experiments.db"):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = ExperimentLogger(db_path)
        self.human_eval = HumanEvaluationFramework(db_path)

        self.bleu = BLEUCalculator()
        self.rouge = ROUGECalculator()
        self.perplexity = PerplexityCalculator(model, tokenizer)
        self.empathy_keyword = EmpathyScoreCalculator()
        self.empathy_semantic = SemanticEmpathyCalculator()
        self.combined = CombinedScoreCalculator()

    def evaluate_all(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        results = {}

        print("Calculating BLEU...")
        results['bleu'] = self.bleu.calculate(predictions, references)

        print("Calculating ROUGE...")
        rouge_scores = self.rouge.calculate(predictions, references)
        results['rouge1'] = rouge_scores['rouge1']
        results['rouge2'] = rouge_scores['rouge2']
        results['rougeL'] = rouge_scores['rougeL']

        results['empathy_keyword'] = self.empathy_keyword.calculate(predictions)

        results['empathy_semantic'] = self.empathy_semantic.calculate(predictions)

        if self.model is not None:
            print("Calculating Perplexity...")
            results['perplexity'] = self.perplexity.calculate(predictions)

        results['combined_score'] = self.combined.calculate(
            results['bleu'],
            results['rougeL'],
            results['empathy_semantic']
        )

        return results

    def log_experiment_results(
        self,
        model_name: str,
        lora_config: Dict,
        train_loss: float,
        metrics: Dict,
        val_loss: Optional[float] = None,
        responses: List[Dict] = None
    ) -> int:
        experiment_id = self.logger.log_experiment(
            model_name=model_name,
            lora_config=lora_config,
            train_loss=train_loss,
            val_loss=val_loss,
            metrics=metrics
        )

        if responses:
            self.logger.log_batch_responses(experiment_id, responses)

        return experiment_id

    def generate_report(self, experiment_id: int) -> str:
        exp = self.logger.get_experiment(experiment_id)
        human_summary = self.human_eval.get_evaluation_summary(experiment_id)

        if exp is None:
            return f"Experiment {experiment_id} not found"

        report = f"""
{'='*60}
EVALUATION REPORT - Experiment #{experiment_id}
{'='*60}

Model: {exp['model_name']}
Timestamp: {exp['timestamp']}
Training Loss: {exp['train_loss']:.4f}

AUTOMATIC METRICS:
-----------------
"""
        if exp['metrics']:
            for key, value in exp['metrics'].items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        report += f"  {k}: {v:.4f}\n"
                elif isinstance(value, (int, float)):
                    report += f"  {key}: {value:.4f}\n"
                else:
                    report += f"  {key}: {value}\n"

        report += f"""
HUMAN EVALUATION:
-----------------
  Average Empathy: {human_summary['avg_empathy']:.2f}/5
  Average Relevance: {human_summary['avg_relevance']:.2f}/5
  Average Fluency: {human_summary['avg_fluency']:.2f}/5
  Average Overall: {human_summary['avg_overall']:.2f}/5
  Total Evaluations: {human_summary['num_evaluations']}

{'='*60}
"""
        return report

    def save_results_json(self, results: Dict, output_path: str = "evaluation_results.json"):
        """Save evaluation results to JSON file"""
        output = {
            'metrics': results,
            'timestamp': datetime.now().isoformat(),
            'reference_actual_results': self.ACTUAL_RESULTS,
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    evaluator = Evaluator()

    for key, value in Evaluator.ACTUAL_RESULTS.items():
        print(f"  {key}: {value}")

    print("\n" + "="*60)
    print("EXAMPLE EVALUATION")
    print("="*60)

    predictions = ["আমি বুঝতে পারছি আপনার কষ্ট। চিন্তা করবেন না, সব ঠিক হয়ে যাবে।"]
    references = ["আপনার কষ্ট বুঝি। আশা রাখুন, ভালো সময় আসবে।"]

    results = evaluator.evaluate_all(predictions, references)

    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
