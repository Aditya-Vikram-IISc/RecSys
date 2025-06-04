import torch
from torch.utils.data import Dataset

class MatrixFactorizationExplicitDataset(Dataset):
    def __init__(self, users_list, items_list, score_list):
        super().__init__()
        

        if not (len(users_list) == len(items_list) == len(score_list)):
            raise ValueError(f"All input lists i.e users_list, items_list, score_list should be of same size")

        self.users_list = torch.tensor(users_list, dtype = torch.long)
        self.items_list = torch.tensor(items_list, dtype = torch.long)
        self.score_list = torch.tensor(score_list, dtype = torch.float32)

    def __len__(self):
        return len(self.users_list)
    
    def __getitem__(self, idx):
        return self.users_list[idx], self.items_list[idx], self.score_list[idx]