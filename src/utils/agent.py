import asyncio
import importlib.util
from copy import deepcopy
from typing import Optional

import aiohttp

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SingleTurnRolloutExecutor:
    def __init__(self, reward_source: Optional[str] = None):
        reward_endpoints = [reward_source] if isinstance(reward_source, str) else reward_source
        self.reward_endpoints = reward_endpoints or []
        self.reward_func = None

        if self.reward_endpoints and self.reward_endpoints[0].endswith(".py"):
            logger.info(f"Loading reward_func from {self.reward_endpoints[0]}")
            spec = importlib.util.spec_from_file_location("reward_func", self.reward_endpoints[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.reward_func = reward_module.reward_func

    async def execute(self, prompt, label, sampling_params, max_length: int, hf_tokenizer, llm_engine):
        prompt_token_ids = hf_tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        max_prompt_length = max_length - sampling_params.max_tokens
        if len(prompt_token_ids) > max_prompt_length:
            logger.warning(
                f"Prompt length ({len(prompt_token_ids)}) exceeds max_prompt_length ({max_prompt_length}); truncating."
            )
            prompt_token_ids = prompt_token_ids[-max_prompt_length:]

        request_output = await llm_engine.generate(prompt_token_ids, deepcopy(sampling_params))
        output = request_output.outputs[0]
        action_token_ids = output.token_ids

        rollout_log_probs = None
        if sampling_params.logprobs is not None and output.logprobs is not None:
            rollout_log_probs = [0.0] * len(prompt_token_ids)
            for token_id, logprob_dict in zip(action_token_ids, output.logprobs):
                token_logprob = logprob_dict.get(token_id)
                rollout_log_probs.append(token_logprob.logprob if token_logprob is not None else 0.0)

        response = {
            "prompt": prompt,
            "label": label,
            "observation_tokens": prompt_token_ids + action_token_ids,
            "action_ranges": [(len(prompt_token_ids), len(prompt_token_ids) + len(action_token_ids))],
            "rollout_log_probs": rollout_log_probs,
            "truncated": output.finish_reason == "length",
            "reward": None,
            "scores": None,
            "extra_logs": {},
        }

        if self.reward_endpoints:
            query = hf_tokenizer.decode(response["observation_tokens"], skip_special_tokens=False)
            reward_payload = await self._fetch_rewards([query], [prompt], [label])
            reward_info = reward_payload[0] if reward_payload else None
            if reward_info:
                response["reward"] = reward_info.get("rewards")
                response["scores"] = reward_info.get("scores", response["reward"])
                response["extra_logs"] = reward_info.get("extra_logs", {})

        return response

    async def _fetch_rewards(self, queries, prompts, labels):
        if self.reward_func is not None:
            return await asyncio.gather(asyncio.to_thread(self.reward_func, queries, prompts, labels))
        return await self._fetch_rewards_via_http(queries, prompts, labels)

    async def _fetch_rewards_via_http(self, queries, prompts, labels):
        num_servers = len(self.reward_endpoints)
        batch_size = (len(queries) + num_servers - 1) // num_servers
        timeout = aiohttp.ClientTimeout(total=180)
        tasks = []

        for i, endpoint in enumerate(self.reward_endpoints):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(queries))
            payload = {
                "query": queries[start_idx:end_idx],
                "prompts": prompts[start_idx:end_idx],
                "labels": labels[start_idx:end_idx],
            }

            async def _post_request(url, data, try_max_times=5):
                for _ in range(try_max_times):
                    try:
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            async with session.post(url, json=data) as response:
                                response.raise_for_status()
                                return await response.json()
                    except aiohttp.ClientError as exc:
                        logger.info(f"Reward request error: {exc}")
                    await asyncio.sleep(1)
                raise RuntimeError(f"Reward request failed for {url}")

            tasks.append(asyncio.create_task(_post_request(endpoint, payload)))

        return await asyncio.gather(*tasks)
