from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_prompt(
    data,
    input_key="input",
    label_key=None,
    prompt_template=None,
    apply_chat_template=None,
):
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if prompt_template:
            prompt = prompt_template.format(prompt)

    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """Prompt dataset supporting both the simplified GRPO path and the original PPO/Ray path."""

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy=None,
        input_template=None,
        input_key="input",
        label_key=None,
        prompt_template=None,
        apply_chat_template=False,
        tokenizer_chat_template=None,
        show_progress=True,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        if prompt_template is None:
            prompt_template = input_template
        if strategy is not None:
            input_key = getattr(strategy.args, "input_key", input_key)
            label_key = getattr(strategy.args, "label_key", label_key)
            apply_chat_template = getattr(strategy.args, "apply_chat_template", apply_chat_template)
            show_progress = strategy.is_rank_0()

        chat_template = None
        if apply_chat_template:
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
            chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        self.labels = []
        self.datasources = []
        iterator = tqdm(dataset, desc="Preprocessing prompts", disable=not show_progress)
        for data in iterator:
            prompt, label = preprocess_prompt(
                data,
                input_key=input_key,
                label_key=label_key,
                prompt_template=prompt_template,
                apply_chat_template=chat_template,
            )
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        for datasource, prompt, label in item_list:
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
        return datasources, prompts, labels
