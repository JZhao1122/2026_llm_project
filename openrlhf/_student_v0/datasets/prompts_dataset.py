from torch.utils.data import Dataset
from tqdm import tqdm


class PromptDataset(Dataset):
    """Prompt dataset supporting both the simplified GRPO path and the original PPO/Ray path."""

    def __init__(
        self,
        dataset,
        input_key,
        label_key,
        **kwargs
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.input_key = input_key
        self.label_key = label_key

        self.datasources = []
        iterator = tqdm(dataset, desc="Preprocessing prompts")
        for data in iterator:
            self.datasources.append(data.get("datasource", "default"))
        
        '''NOTE: You can add more code here.'''

    def __len__(self):
        '''
        TODO: Return the number of samples in the dataset.
        return an integer
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======

    def __getitem__(self, idx):
        '''
        TODO: Get a single prompt sample for GRPO rollout.
        Returns:
            tuple: (datasource, prompt, label)
                - datasource: str, data source name of this sample
                - prompt: str, input text sent to the policy / vLLM
                - label: str, reference answer used by reward/evaluation; set to "" if unused
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======

    def collate_fn(self, item_list):
        datasources = []
        prompts = []
        labels = []
        for datasource, prompt, label in item_list:
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
        return datasources, prompts, labels
