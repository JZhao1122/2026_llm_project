# Student Assignment Guide: SFT First, Then GRPO

This document is written for teaching use.

## Part 1: SFT

### Goal

Students implement a minimal `Supervised Fine-Tuning (SFT)` pipeline:

- build tokenized training samples
- compute per-token `log probabilities`
- compute masked negative log-prob loss on assistant tokens only

### Files To Implement

| File | Function(s) | Purpose |
|---|---|---|
| `datasets/sft_dataset.py` | `__len__`, `__getitem__` | turn one raw example into one tokenized training example |
| `models/sft_model.py` | `__init__`, `forward` | wrap a causal LM and output per-token log-probs |
| `models/loss.py` | `SFTLoss.forward` | compute masked SFT loss |

### Training Data Flow

The SFT pipeline in this repo follows this order:

1. raw dataset sample
2. `SFTDataset.__getitem__(idx)`
3. return `(input_ids, attention_mask, loss_mask)` with shape `[1, seq_len]`
4. `collate_fn` pads samples into batch tensors
5. trainer squeezes dimension `1`
6. model returns `per_token_log_probs`
7. loss uses `loss_mask[:, :-1]`

You can see the trainer-side contract in [`_student_v0/trainer/sft_trainer.py`](./trainer/sft_trainer.py): the training loop does

```python
per_token_log_probs = self.model(inputs, attention_mask=attention_mask)
loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
```

So the shapes must line up exactly.

### Main Symbols

| Symbol | Type / Shape | Meaning |
|---|---|---|
| `input_ids` | `LongTensor[1, T]` in `__getitem__`, later `LongTensor[B, T]` | token ids of `prompt + response + eos` |
| `attention_mask` | `LongTensor[1, T]` or `LongTensor[B, T]` | `1` for valid tokens, `0` for padding |
| `loss_mask` | `FloatTensor[1, T]` or `FloatTensor[B, T]` | `1` only where loss should be counted |
| `per_token_log_probs` | `FloatTensor[B, T-1]` | log-prob of the next token at each prediction position |
| `labels` | `LongTensor[B, T]` before slicing | shifted target tokens, usually from `torch.roll(input_ids, -1, dims=1)` |
| `position_ids` | `LongTensor[B, T]` | position ids built from `attention_mask` |

### What To Tell Students

You can give students the following task statement.

> Implement the minimal SFT pipeline for a causal language model.
> Your code should:
> 1. turn each `(prompt, response)` pair into `input_ids`, `attention_mask`, and `loss_mask`
> 2. compute per-token next-token `log probabilities`
> 3. compute the masked negative log-prob loss on response tokens only
>
> Important:
> - The model is trained with `next-token prediction`
> - Loss should only be applied to the assistant response, not the prompt
> - Padding tokens should not contribute to the loss

### Function Contracts

#### `SFTDataset.__len__`

Expected behavior:

- return the number of samples in the dataset
- return type is `int`

Minimal expectation:

- `return len(self.dataset)`

#### `SFTDataset.__getitem__(idx)`

Expected behavior:

1. read one raw sample from `self.dataset[idx]`
2. extract prompt text from `input_key`
3. extract response text from `output_key`
4. if `prompt_template` exists in `kwargs`, format the prompt first
5. concatenate prompt and response into one training sequence
6. append `eos_token` if needed
7. tokenize into `input_ids` and `attention_mask`
8. build `loss_mask` so that only response tokens contribute to loss
9. return `(input_ids, attention_mask, loss_mask)`

Recommended shape contract:

| Output | Shape |
|---|---|
| `input_ids` | `[1, seq_len]` |
| `attention_mask` | `[1, seq_len]` |
| `loss_mask` | `[1, seq_len]` |

Important masking rule:

- If the prompt length is `P`, then the first predicted response token is aligned with position `P - 1`.
- So for single-turn SFT, a standard choice is:
  - set `loss_mask[0, P - 1 : -1] = 1`
  - keep prompt region and final padded area as `0`

Why `P - 1` instead of `P`?

- because causal LM predicts the token at position `t + 1` from hidden state at position `t`
- this is the classic off-by-one point students often miss

#### `SFTModel.__init__`

Expected behavior:

- load a causal LM with `AutoModelForCausalLM.from_pretrained(...)`
- if `ZeRO-3` is used, initialize `HfDeepSpeedConfig`
- convert `param_dtype` string to a torch dtype
- store the Hugging Face model in `self.model`
- set `self.model.config.use_cache = False`

The official reference version does exactly this in [`openrlhf/models/sft_model.py`](../models/sft_model.py).

#### `SFTModel.forward(input_ids, attention_mask, **kwargs)`

Expected behavior:

1. compute `position_ids` from `attention_mask`
2. run the causal LM forward pass
3. get logits and convert them to `float32`
4. create shifted labels from `input_ids`
5. compute token-level `log probabilities`
6. drop the last position, because there is no valid next token after the final token
7. return `log_probs` of shape `[batch_size, seq_len - 1]`

Important shape relation:

| Tensor | Shape |
|---|---|
| `input_ids` | `[B, T]` |
| `logits` | `[B, T, V]` |
| `log_probs` before slice | `[B, T]` |
| final output | `[B, T-1]` |

#### `SFTLoss.forward(per_token_logps, loss_mask, **kwargs)`

Expected behavior:

- compute masked negative log-prob loss
- average only over tokens where `loss_mask == 1`
- return a scalar tensor

Mathematically:

```text
loss = mean( - per_token_logps ) over valid response tokens only
```

This is the minimal SFT objective in this repo.

---

## Part 2: GRPO

### Goal

Students implement the post-rollout logic in `GRPO`:

- actor `log_probs`
- reference `log_probs`
- reward scoring
- group-relative reward normalization
- `KL` penalty and discounted returns

### File To Implement

| File | Function(s) | Purpose |
|---|---|---|
| `trainer/grpo_trainer.py` | `_compute_action_log_probs` | actor-side log probs |
| `trainer/grpo_trainer.py` | `_compute_reference_log_probs` | reference log probs |
| `trainer/grpo_trainer.py` | `_score_with_reward_model` | sequence-level rewards |
| `trainer/grpo_trainer.py` | `_compute_grpo_advantages` | group normalization |
| `trainer/grpo_trainer.py` | `_compute_kl_and_returns` | KL penalty, token rewards, returns |

### Data Flow

This is the correct training flow in this repo:

1. dataloader yields `(datasources, prompts, labels)`
2. `_generate_rollouts(prompts, labels)` returns `samples`
3. each sample is one `GRPOExperience`
4. `_split_into_micro_batches(samples)` turns them into `batches`
5. the five target functions operate on `batches`

This point is important:

- `batches` is **not** the raw dataloader batch
- `batches` is `List[GRPOExperience]`
- each element of that list is already a padded `micro-batch`

### Main Symbols

| Symbol | Type / Shape | Meaning |
|---|---|---|
| `batches` | `List[GRPOExperience]` | a list of micro-batches |
| `batch` | `GRPOExperience` | one padded experience batch |
| `batch.sequences` | `LongTensor[B, T]` | full token sequence, usually `prompt + response` |
| `batch.attention_mask` | `LongTensor[B, T]` | valid-token mask |
| `batch.action_mask` | `BoolTensor[B, T-1]` | which token positions belong to response actions |
| `batch.rewards` | `FloatTensor[B]` or `None` | sequence-level reward for each sampled response |
| `old_log_probs` | `FloatTensor[B, A]` | actor log probs on action positions |
| `base_log_probs` | `FloatTensor[B, A]` or `None` | reference log probs |
| `normalized_rewards` | `FloatTensor[num_samples]` | reward normalized within each prompt group |
| `group_reward_std` | `FloatTensor[num_samples]` | group std for logging |
| `batch.returns` | `FloatTensor[B, A]` | discounted token-level returns |
| `batch.advantages` | `FloatTensor[B, A]` | in this simplified trainer, copied from returns |

### What To Tell Students

You can give students the following task statement.

> Implement the GRPO post-rollout pipeline.
> Your code should:
> 1. compute action-token `log probabilities` from the actor model
> 2. optionally compute reference-model `log probabilities`
> 3. score each generated response with a reward model
> 4. normalize rewards within each prompt group
> 5. convert sequence-level rewards into token-level returns with an optional `KL` penalty
>
> Important:
> - `batches` is a `List[GRPOExperience]`, not the dataloader output
> - reward is sequence-level first, then converted to token-level
> - the code mutates each `batch` in-place for later policy update

### `GRPOExperience`

Students should understand the `GRPOExperience` contract before coding.
The structure is defined in [`_student_v0/trainer/grpo_types.py`](./trainer/grpo_types.py).

Important fields:

| Field | Meaning |
|---|---|
| `sequences` | padded token ids |
| `attention_mask` | valid-token mask |
| `action_mask` | response-action mask |
| `old_action_log_probs` | actor log probs stored for PPO-style update |
| `base_action_log_probs` | reference log probs stored for KL loss mode |
| `advantages` | token-level advantages |
| `returns` | discounted token-level returns |
| `rewards` | sequence-level scalar reward per sampled response |
| `info` | logging information such as reward, return, response length, KL |

### Function Contracts

#### `_compute_action_log_probs(self, batches)`

Expected behavior:

- call the actor model group with:
  - `sequences=[b.sequences for b in batches]`
  - `action_mask=[b.action_mask for b in batches]`
  - `attention_mask=[b.attention_mask for b in batches]`
- flatten Ray refs
- return `List[torch.Tensor]`

Returned shape:

- one tensor per micro-batch
- each tensor is typically `[B, A]`, where `A` matches the action dimension

#### `_compute_reference_log_probs(self, batches)`

Expected behavior:

- if `self.reference_model_group is None`, return `[None] * len(batches)`
- otherwise do the same logic as actor forward
- return `List[Optional[torch.Tensor]]`

This is mainly a control-flow exercise plus correct batching.

#### `_score_with_reward_model(self, batches)`

Expected behavior:

- if `self.reward_model_group is None`, do nothing
- otherwise run reward model forward on each micro-batch
- reward model returns one scalar reward per sequence
- for each `batch`, write:
  - `batch.rewards = reward.float()`
  - `batch.info["reward"] = reward.float()`
  - `batch.info["score"] = reward.float()`

Important point:

- this function mutates `batches` in-place
- `batch.rewards` is shape `[B]`, not `[B, T]`

#### `_compute_grpo_advantages(self, batches)`

Expected behavior:

1. concatenate all rewards into one flat tensor
2. reshape to `[num_prompts, n_samples_per_prompt]`
3. compute mean and std inside each prompt group
4. normalize rewards inside the group
5. return:
  - `normalized_rewards` as flat tensor
  - `group_reward_std` as flat tensor

Core formula:

```text
A_i = (r_i - mean(group)) / (std(group) + eps)
```

Important hidden assumption:

- rollout order must still preserve prompt grouping
- otherwise `.view(-1, n_samples_per_prompt)` is wrong

#### `_compute_kl_and_returns(...)`

Expected behavior:

For each micro-batch:

1. slice the correct reward segment from the flat normalized rewards
2. compute approximate `KL` if reference log probs exist and `use_kl_loss` is `False`
3. otherwise use zero `KL`
4. convert scalar normalized reward into token-level reward with `compute_reward(...)`
5. compute discounted returns
6. write the necessary fields back into the batch
7. update logging info

This function should mutate each batch in-place:

- `batch.old_action_log_probs`
- `batch.base_action_log_probs`
- `batch.returns`
- `batch.advantages`
- `batch.info["return"]`
- `batch.info["group_reward_std"]`
- optionally `batch.info["kl"]`

Important semantic detail:

- when `use_kl_loss == False`:
  - `KL` is used immediately as a token-level reward penalty
  - `batch.base_action_log_probs` is set to `None`
- when `use_kl_loss == True`:
  - do not inject KL penalty here
  - keep `base_action_log_probs` for the later actor loss computation

### Why This GRPO Split Works Well

This assignment split is reasonable because it separates:

- simple system-plumbing tasks:
  - `_compute_action_log_probs`
  - `_compute_reference_log_probs`
  - `_score_with_reward_model`
- core algorithmic tasks:
  - `_compute_grpo_advantages`
  - `_compute_kl_and_returns`

So students can first connect the pipeline, then understand the real `GRPO` math.

### Common Mistakes In GRPO

- confusing dataloader output with `batches`
- forgetting that `batch.rewards` is sequence-level, not token-level
- mismatching shapes `[B, T]` vs `[B, T-1]`
- breaking prompt-group order before reward normalization
- forgetting in-place mutation of batch fields
- misunderstanding the `use_kl_loss` branch
- forgetting to move sliced tensors onto the same device as `old_log_probs`

### Suggested Minimal Self-Checks

You can ask students to verify:

1. `len(action_log_probs) == len(batches)`
2. each reward tensor has shape `[B]`
3. `normalized_rewards.numel() == total_num_samples`
4. each `batch.returns.shape == batch.action_mask.shape`
5. after `_compute_kl_and_returns`, every batch has non-`None` `old_action_log_probs`, `returns`, and `advantages`

---

## Recommended Teaching Order

If you want the assignment difficulty to feel smooth, I recommend this order:

1. `SFTDataset.__len__`
2. `SFTDataset.__getitem__`
3. `SFTModel.forward`
4. `SFTLoss.forward`
5. `GRPO._compute_action_log_probs`
6. `GRPO._compute_reference_log_probs`
7. `GRPO._score_with_reward_model`
8. `GRPO._compute_grpo_advantages`
9. `GRPO._compute_kl_and_returns`

This order is good because students first learn:

- tokenization
- masks
- next-token prediction
- masked loss

and only then move into:

- rollout processing
- reward normalization
- `KL` penalty
- return computation

## Bottom-Line Recommendation

Yes, this teaching plan is workable.

My only strong recommendation is:

- for `SFT`, clearly specify the shape contract and the `loss_mask` alignment
- for `GRPO`, clearly specify that `batches` is `List[GRPOExperience]`, not the dataloader batch
- for both parts, give students at least one tiny shape-based self-check

If you do that, the assignment difficulty is reasonable and the conceptual progression is clean.
