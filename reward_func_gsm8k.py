import re
import torch

def _extract_answer(text: str) -> str:
    """Extract the final numerical answer after '####'."""
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    # Fallback: grab the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else ""

def reward_func(queries, prompts, labels, **kwargs):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        response = query[len(prompt):] if query.startswith(prompt) else query
        predicted = _extract_answer(response)
        expected = _extract_answer(label)
        reward = 1.0 if predicted == expected else 0.0
        rewards.append(reward)

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    return {
        "rewards": reward_tensor,
        "scores": reward_tensor,
        "extra_logs": {"accuracy": reward_tensor},
    }
