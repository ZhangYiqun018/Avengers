import sqlite3
import hashlib
import json
import os
import time
import threading
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path
from loguru import logger
from scipy.spatial.distance import cosine, cdist

from openai import OpenAI, RateLimitError, APIError, NOT_GIVEN
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput
from transformers import AutoTokenizer, AutoModel


class KeywordsGenerator:
    def __init__(self, client, model_name, temperature=0.2, top_p=1.0, top_k=NOT_GIVEN):
        self.client = client
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
    # 定义重试日志记录函数
    def _log_retry(retry_state):
        exception = retry_state.outcome.exception()
        if exception:
            logger.warning(f"Retrying KeywordsGenerator.generate due to error: {str(exception)}. Attempt {retry_state.attempt_number}/{retry_state.retry_object.stop.max_attempt_number}")
        return None
    
    @retry(
        stop=stop_after_attempt(5),  # 最多重试5次
        wait=wait_exponential(multiplier=1, min=2, max=60),  # 指数退避策略：1*2^x 秒，最少2秒，最多60秒
        retry=retry_if_exception_type((Exception)),  # 捕获所有异常进行重试
        before_sleep=_log_retry  # 重试前记录日志
    )
    def generate_with_retry(self, question: str):
        if "Distill" in self.model or "EXAOME" in self.model:
            question += "Don't make your reasoning and thinking too long.\n"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": question}],
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=200,
            )
            usage = response.usage
            choices = response.choices
            assert choices[0].message.content is not None, f"choices[0].message.content is None"
            return {
                "first_output": choices[0].message.content,
                "raw_output": [choice.message.content for choice in choices],
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens
            }
        except Exception as e:
            logger.error(f"Error in KeywordsGenerator.generate: {str(e)}, model_name: {self.model}")
            raise  # 重新抛出异常，让重试装饰器捕获
    
    def generate(self, question: str):
        try:
            return self.generate_with_retry(question=question)
        except Exception as e:
            logger.error(
                f"Error in KeywordsGenerator.generate after all retries: "
                f"{str(e)}, model_name: {self.model}"
            )
            return {
                "first_output": "failed to generate",
                "raw_output": ["failed to generate"],
                "prompt_tokens": 0,
                "completion_tokens": 0
            }

    def extract_keywords(self, question: str):
        prompt = (
            f"Question: {question}\n"
            f"What are the core knowledge, subjects or skills needed to solve this problem? "
            f"List 2-5 keywords separated in comma. "
            f"Example keywords: psychology, virology, behavioral theory, microbiology, "
            f"diplomacy, political science, property law, finance, business. "
            f"Give ONLY the keywords, no other words or explanation. "
            f"Follow this format: Keywords: <keyword1>, <keyword2>..."
        )
        result = self.generate(prompt)
        output = result["first_output"]
        keywords = output.split("Keywords:")[-1].split(",")
        keywords = [i.strip().lower().replace(".", "") for i in keywords]
        return keywords


class EmbeddingCache:
    """Thin wrapper around the OpenAI embedding endpoint with SQLite caching."""
    def __init__(
        self,
        base_url: str = "http://172.30.28.129:8000/v1",
        api_key: str = "sk-1234567890",
        model_name: str = "bge-m3",
        cache_dir: str | os.PathLike = "/fs-computility/mabasic/zhangyiqun/Revengers/.cache",
        max_retries: int = 5,
        initial_delay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "embeddings.db"
        # —— 单一持久连接 ——
        self._conn = self._open_conn()
        self._init_db()

        # 写锁，保证一次只写一条
        self._w_lock = threading.Lock()
        
        self._init_db()

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------

    def get(self, text: str) -> List[float]:
        """Return the embedding for *text*, fetching from cache or remote."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 1. try cache ------------------------------------------------------
        row = self._select(text_hash)
        if row is not None:
            return row

        # 2. call OpenAI ----------------------------------------------------
        delay = self.initial_delay
        for attempt in range(self.max_retries):
            try:
                rsp = self._client.embeddings.create(input=text, model=self.model_name)
                emb: List[float] = rsp.data[0].embedding  # type: ignore[index]
                self._insert(text_hash, text, emb)
                return emb

            except RateLimitError as e:
                logger.warning(
                    f"Rate limited (attempt {attempt+1}/{self.max_retries}). Retry in {delay:.1f}s"
                )
            except APIError as e:
                logger.warning(
                    f"OpenAI API error (attempt {attempt+1}/{self.max_retries}): {e}. Retry in {delay:.1f}s"
                )
            except Exception as e:
                logger.error(f"Unexpected error — abort: {e}")
                raise

            time.sleep(delay)
            delay *= 2

        raise RuntimeError("Failed to get embedding after multiple retries.")

    def batch(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for a list of texts (keeps order)."""
        # split into cache hits + misses to minimise remote calls
        hits: List[List[float]] = []
        misses: List[str] = []
        mapping: dict[str, int] = {}

        for idx, t in enumerate(texts):
            h = hashlib.md5(t.encode()).hexdigest()
            row = self._select(h)
            if row is not None:
                hits.append(row)
            else:
                mapping[t] = idx
                misses.append(t)

        # fetch misses in one API call if any
        if misses:
            rsp = self._client.embeddings.create(input=misses, model=self.model_name)
            for text, record in zip(misses, rsp.data):  # type: ignore[attr-defined]
                emb: List[float] = record.embedding
                self._insert(hashlib.md5(text.encode()).hexdigest(), text, emb)
                hits.insert(mapping[text], emb)

        return hits

    # ------------------------------------------------------------------
    # private db helpers
    # ------------------------------------------------------------------
    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=30,               # 等锁 30s
            check_same_thread=False,  # 允许跨线程
            isolation_level=None      # autocommit
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn
    
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT,
                    model     TEXT,
                    embedding TEXT NOT NULL,
                    text      TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, model)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model);")

    def _select(self, text_hash: str) -> List[float] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=?",
                (text_hash, self.model_name),
            ).fetchone()
            if row:
                return json.loads(row[0])  # stored as JSON string for readability
            return None

    def _insert(self, text_hash: str, text: str, embedding: List[float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, text) VALUES (?,?,?,?)",
                (text_hash, self.model_name, json.dumps(embedding), text),
            )
     
class SymbolicMoERouter(BaseRouter):
    def __init__(self, normal_experts: List[Expert], thinking_experts: List[Expert], router_config: dict):
        super().__init__(normal_experts, thinking_experts)
        self.config = router_config["symbolic_moe_router"]
        self.keywords_generator = KeywordsGenerator(
            client=OpenAI(base_url=self.config['base_url'], api_key=self.config['api_key']),
            model_name=self.config["keywords_model"]
        )
        self.embedding_cache = EmbeddingCache()
        self.model_profiles, self.all_profile_keywords, self.profile_embeddings, self.model_specific_embeddings = self._load_model_profiles()
        self.global_competency = self._calculate_global_competency()

    def _load_model_profiles(self) -> Tuple[Dict[str, Dict[str, float]], List[str], List[List[float]]]:
        """Load model profiles from JSON files and compute profile keywords embeddings."""
        profiles = {}
        all_profile_keywords = set()
        
        # Load profiles and collect all keywords
        for model_name in self.config['model_aliases'].keys():
            profile_path = Path(self.config['profiles_dir']) / f"{model_name}_profile.json"
            if not profile_path.exists():
                raise FileNotFoundError(f"Profile not found for model {model_name}")
            
            with open(profile_path, 'r') as f:
                profiles[model_name] = json.load(f)
                all_profile_keywords.update(profiles[model_name].keys())
        
        # Convert to list and compute embeddings
        all_profile_keywords = list(all_profile_keywords)
        profile_embeddings = self.embedding_cache.batch(all_profile_keywords)
        
        # Pre-calculate model-specific profile keywords and embeddings
        model_specific_embeddings = {}
        for model_name, profile in profiles.items():
            model_profile_keywords = list(profile.keys())
            model_profile_indices = [all_profile_keywords.index(k) for k in model_profile_keywords]
            model_profile_embeddings = [profile_embeddings[i] for i in model_profile_indices]
            model_specific_embeddings[model_name] = {
                'keywords': model_profile_keywords,
                'embeddings': np.array(model_profile_embeddings)
            }
        
        logger.info(f"Loaded {len(profiles)} model profiles with {len(all_profile_keywords)} keywords")
        
        return profiles, all_profile_keywords, profile_embeddings, model_specific_embeddings

    def _normalize_to_probability(self, numbers: List[float], temperature: float = 1.0) -> List[float]:
        """Normalize numbers to probabilities using temperature scaling."""
        min_val = min(numbers)
        if min_val < 0:
            shifted = [x - min_val + 1 for x in numbers]
        else:
            shifted = numbers
        
        scaled = [x ** (1/temperature) for x in shifted]
        
        total = sum(scaled)
        probabilities = [x/total for x in scaled]
        
        return probabilities

    def _calculate_global_competency(self) -> Dict[str, float]:
        """Calculate normalized global competency scores for each model."""
        total_scores = {}
        for model_name, profile in self.model_profiles.items():
            total_scores[model_name] = sum(profile.values())
        
        # Convert scores to list and normalize
        scores = list(total_scores.values())
        probabilities = self._normalize_to_probability(scores)
        
        # Convert back to dictionary
        return {model: float(prob) for model, prob in zip(total_scores.keys(), probabilities)}

    def _calculate_local_suitability(self, question_keywords: List[str]) -> Dict[str, float]:
        """Calculate local suitability scores for each model.
        For each model, map question keywords to its profile keywords and calculate scores."""
        model_scores = {}
        
        # Get embeddings for question keywords
        question_embeddings = self.embedding_cache.batch(question_keywords)
        
        # For each model, calculate its score
        for model_name, profile in self.model_profiles.items():
            # Map question keywords to this model's profile keywords
            mapped_keywords = []
            
            # Get pre-calculated model-specific keywords and embeddings
            model_profile_keywords = self.model_specific_embeddings[model_name]['keywords']
            model_profile_embeddings = self.model_specific_embeddings[model_name]['embeddings']
            
            for q_emb in question_embeddings:
                # Convert to numpy array for vectorized operation
                q_emb_array = np.array(q_emb).reshape(1, -1)
                
                # Vectorized cosine distance calculation
                # 1 - cosine distance = cosine similarity
                similarities = 1 - cdist(q_emb_array, model_profile_embeddings, metric='cosine')[0]
                
                closest_idx = np.argmax(similarities)
                mapped_keywords.append(model_profile_keywords[closest_idx])
            
            # Calculate score using mapped keywords
            score = sum(profile[k] for k in mapped_keywords)
            model_scores[model_name] = score
        
        # Get positive models
        positive_models = {model: score for model, score in model_scores.items() if score > 0}
        
        if positive_models:
            # Use positive models directly
            return positive_models
        else:
            # Shift scores to make them positive
            min_score = min(model_scores.values())
            return {model: score - min_score + 1 for model, score in model_scores.items()}

    def _select_experts(self, weights: Dict[str, float], k: int) -> List[Expert]:
        """Select k experts based on weighted probabilities."""
        # Calculate weights by combining local suitability with global competency
        model_dict = dict(weights)
        positive_models = {model: score for model, score in model_dict.items() if score > 0}
        
        if positive_models:
            models = list(positive_models.keys())
            weights_list = [score * self.global_competency[model] for model, score in positive_models.items()]
        else:
            min_score = min(model_dict.values())
            shifted_dict = {model: score - min_score + 1 for model, score in model_dict.items()}
            models = list(shifted_dict.keys())
            weights_list = [score * self.global_competency[model] for model, score in shifted_dict.items()]
        
        # Calculate probabilities
        total_weight = sum(weights_list)
        probabilities = [w/total_weight for w in weights_list]
        
        # Handle the edge case where no models are available
        if not models:
            logger.warning(f"No models available to select from. Using default experts.")
            # Return first k experts as fallback
            return self.normal_experts[:k] if len(self.normal_experts) >= k else self.normal_experts
        
        # Sample models
        if len(models) == 1:
            # If we only have one model, use it k times
            selected_models = [models[0]] * k
        elif len(models) >= k:
            # If we have enough models, select k without replacement
            selected_indices = np.random.choice(
                len(models),
                size=k,
                replace=True,
                p=probabilities
            )
            selected_models = [models[i] for i in selected_indices]
        else:
            # If we don't have enough models, select all available models
            # and then sample with replacement to reach k
            selected_indices_without_replacement = np.arange(len(models))
            additional_indices = np.random.choice(
                len(models),
                size=k-len(models),
                replace=True,
                p=probabilities
            )
            selected_indices = np.concatenate([selected_indices_without_replacement, additional_indices])
            selected_models = [models[i] for i in selected_indices]
        
        assert len(selected_models) == k, f"Selected models number is not equal to k, selected_models: {selected_models}, k: {k}"
        
        # Filter experts based on selected models
        # Allow duplicates based on selected_models (which may contain duplicates)
        selected_experts = []
        for model_name in selected_models:
            for expert in self.normal_experts:
                if expert.model_name == self.config["model_aliases"][model_name]:
                    selected_experts.append(expert)
                    break
        assert len(selected_experts) == k, f"Selected experts number is not equal to k, selected_experts: {len(selected_experts)}, k: {k}"
        return selected_experts
    

    def route(self, question: str) -> RouterOutput:
        try:
            # 1. Extract keywords
            question_keywords = self.keywords_generator.extract_keywords(question)
            # logger.debug(f"Question keywords: {question_keywords}")
            
            # 2. Calculate local suitability (now includes keyword mapping)
            local_suitability = self._calculate_local_suitability(question_keywords)
            # logger.debugs(f"Local suitability: {local_suitability}")
            
            # 3. Combine with global competency
            weights = {
                model: self.global_competency[model] * score
                for model, score in local_suitability.items()
            }
            
            # 4. Select experts
            selected_experts = self._select_experts(
                weights,
                self.config['max_router']
            )
            # logger.debug(f"Selected experts: {selected_experts}")
            return RouterOutput(
                normal_experts=selected_experts,
                thinking_experts=None
            )
            
        except Exception as e:
            logger.error(f"Error in SymbolicMoERouter.route: {str(e)}")
            raise e
        
