import asyncio
import json
import re
from typing import Awaitable, Sequence

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from omegaconf import DictConfig
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion
from tqdm import tqdm

from src.paths import jinja_templates_dir, llm_cluster_label_json, llm_synth_ds_json
from src.settings import LLMSettings
from src.utils import get_config


class LabelingHandler:
    def __init__(self, llm_settings: LLMSettings, config_settings: DictConfig, seed: int = 42) -> None:
        self.config_settings = config_settings
        self.openai_client = AsyncOpenAI(
            api_key=llm_settings.api_key,
            base_url=llm_settings.base_url,
        )
        env = Environment(loader=FileSystemLoader(jinja_templates_dir))
        self.main_template = env.get_template(self.config_settings.main_prompt_name)
        self.synthetic_ds_template = env.get_template(self.config_settings.synthetic_ds_prompt_name)

        self.features = self._load_features()
        self.seed = seed

    async def analyze_clusters(
        self,
        df: pd.DataFrame,
        n_samples: int = 3,
        max_length: int = 400,
        save_every: int = 5,
        llm_params_configname: str = "llm_labeling_config.yaml",
    ) -> tuple[dict, list[dict]]:
        df["cluster"] = df["cluster"].astype(str)
        df = df[df["cluster"] != "-1"]
        clusters = df["cluster"].unique()

        llm_params, semaphore = self._prepare_llm_env(llm_params_configname)

        tasks = [
            self._process_cluster(cid, df, n_samples, max_length, self.features, semaphore, llm_params)
            for cid in clusters
        ]

        results = await self._run_tasks_with_progress(
            tasks, desc="Cluster analysis", save_path=str(llm_cluster_label_json), save_every=save_every
        )

        return results, self.features

    async def generate_synthetic_ds(
        self,
        n_per_language: int = 5,
        target_features: list[str] | None = None,
        llm_params_configname: str = "llm_synth_ds_config.yaml",
    ) -> tuple[dict, list[dict]]:
        llm_params, semaphore = self._prepare_llm_env(llm_params_configname)

        features_to_process = (
            [f for f in self.features if f["name"] in target_features] if target_features else self.features
        )
        if target_features and not features_to_process:
            logger.warning("No matching features found for the given target_features list. Skipping generation.")
            return {}, self.features

        if not features_to_process:
            logger.warning("No features available for synthetic dataset generation.")
            return {}, self.features

        tasks = [
            self._process_feature(f, self.features, n_per_language, semaphore, llm_params) for f in features_to_process
        ]
        results = await self._run_tasks_with_progress(
            tasks, desc="Synthetic DS generation", save_path=str(llm_synth_ds_json)
        )

        return results, self.features

    def _prepare_llm_env(self, llm_params_configname: str) -> tuple[DictConfig, asyncio.Semaphore]:
        llm_params = get_config(llm_params_configname)
        semaphore = asyncio.Semaphore(self.config_settings.max_parallel)
        return llm_params, semaphore

    async def _process_cluster(
        self,
        cluster_id: str,
        df: pd.DataFrame,
        n_samples: int,
        max_length: int,
        features: list,
        semaphore: asyncio.Semaphore,
        llm_params: DictConfig,
    ) -> tuple[str, dict | str | None]:
        async with semaphore:
            cluster_sentences = df.loc[df["cluster"] == cluster_id, "sentence"].tolist()
            sentences = self._get_processed_sentences(cluster_sentences, n_samples, max_length)
            prompt = self.main_template.render(cluster_id=cluster_id, sentences=sentences, features=features)
            return cluster_id, await self._analyze_with_llm(
                cluster_id, prompt, entity_type="cluster", llm_params=llm_params
            )

    def _get_processed_sentences(self, cluster_sentences: list[str], n_samples: int, max_length: int) -> list[str]:
        np.random.seed(self.seed)

        short_sentences = [s for s in cluster_sentences if len(s) <= max_length]
        long_sentences = [s for s in cluster_sentences if len(s) > max_length]

        n_available_short = len(short_sentences)
        n_needed = min(n_samples, len(cluster_sentences))

        if n_available_short >= n_needed:
            processed_sentences = np.random.choice(short_sentences, n_needed, replace=False).tolist()
        elif n_available_short > 0:
            processed_sentences = short_sentences.copy()
            n_additional = min(len(long_sentences), n_needed - n_available_short)
            additional_samples = np.random.choice(long_sentences, n_additional, replace=False).tolist()
            truncated_samples = [s[:max_length] + "..." for s in additional_samples]
            processed_sentences.extend(truncated_samples)
        else:
            selected = np.random.choice(long_sentences, min(n_samples, len(long_sentences)), replace=False).tolist()
            processed_sentences = [s[:max_length] + "..." for s in selected]

        return processed_sentences

    async def _process_feature(
        self,
        feature: dict,
        all_features: list[dict],
        n_per_language: int,
        semaphore: asyncio.Semaphore,
        llm_params: DictConfig,
    ) -> tuple[str, dict | str | None]:
        async with semaphore:
            feature_name = feature["name"]
            prompt = self.synthetic_ds_template.render(
                target_feature=feature, all_features=all_features, n_per_language=n_per_language
            )
            return feature_name, await self._analyze_with_llm(
                feature_name, prompt, entity_type="feature", llm_params=llm_params
            )

    async def _analyze_with_llm(
        self, name: str, prompt: str, entity_type: str, llm_params: DictConfig
    ) -> dict | str | None:
        try:
            response = await self._get_response_with_retry(prompt, llm_params)
            if not response:
                msg = f"LLM returned None for {entity_type} '{name}'"
                logger.error(msg)
                return msg
        except Exception as e:
            msg = f"LLM request failed for {entity_type} '{name}': {e}"
            logger.error(msg)
            return msg

        return self._parse_llm_response(response, name, entity_type)

    async def _get_response_with_retry(self, prompt: str, llm_params: DictConfig) -> ChatCompletion | None:
        max_retries = self.config_settings.max_retries
        delay = self.config_settings.initial_delay
        backoff_factor = self.config_settings.backoff_factor

        for attempt in range(max_retries):
            try:
                return await self.openai_client.chat.completions.create(
                    model=llm_params.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=llm_params.max_tokens,
                    temperature=llm_params.temperature,
                    top_p=llm_params.top_p,
                    presence_penalty=llm_params.presence_penalty,
                    extra_body={
                        "top_k": llm_params.top_k,
                        "chat_template_kwargs": {"enable_thinking": llm_params.enable_thinking},
                    },
                    response_format=llm_params.response_format,
                    reasoning_effort=llm_params.reasoning_effort,
                )
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{max_retries}): "
                    f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                )
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= backoff_factor
            except OpenAIError as e:
                logger.exception(f"Unrecoverable OpenAI error: {str(e)}")
                raise
        return None

    def _parse_llm_response(self, response: ChatCompletion, name: str, entity_type: str) -> dict | str | None:
        content = response.choices[0].message.content
        if not content:
            logger.error(f"Empty content in LLM response for {entity_type} '{name}'")
            return None

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            logger.error(f"No JSON found in LLM response for {entity_type} '{name}': {content}")
            return content

        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON for {entity_type} '{name}': {content}")
            return content

    async def _run_tasks_with_progress(
        self, tasks: Sequence[Awaitable[tuple[str, dict | str | None]]], desc: str, save_path: str, save_every: int = 10
    ) -> dict:
        results, completed = {}, 0
        progress = tqdm(total=len(tasks), desc=desc)

        for coro in asyncio.as_completed(tasks):
            name, output = await coro
            if output:
                results[name] = output
            completed += 1
            progress.update(1)
            if completed % save_every == 0:
                self._save_results(results, save_path)

        progress.close()
        self._save_results(results, save_path)
        return results

    def _load_features(self, config_name: str = "features.yaml") -> list[dict]:
        return get_config(config_name).features

    def _save_results(self, results: dict, path: str) -> None:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")
