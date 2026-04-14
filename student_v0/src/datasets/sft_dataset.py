from typing import Callable
from torch.utils.data import Dataset
from openrlhf.utils.utils import zero_pad_sequences

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

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []
        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks
