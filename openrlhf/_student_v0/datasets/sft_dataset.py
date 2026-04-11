from typing import Callable
from torch.utils.data import Dataset

class SFTDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        input_key: str,
        output_key: str,
        **kwargs
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_key = input_key
        self.output_key = output_key

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
        TODO: Get a single tokenized training sample.
        Returns:
            tuple: (input_ids, attention_mask, loss_mask)
                - input_ids: tensor of shape [1, seq_len]
                - attention_mask: tensor of shape [1, seq_len]  
                - loss_mask: tensor of shape [1, seq_len]
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======
