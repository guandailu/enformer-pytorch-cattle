import os
import torch


class BasenjiDataset(torch.utils.data.Dataset):

    def __init__(self, cattle_file):
        self._cattle_file = cattle_file
        self._cattle_data = torch.load(self._cattle_file)

    @property
    def cattle_data(self):
        return self._cattle_data
    
    def __len__(self):
        return len(self.cattle_data)

    def __getitem__(self, idx):
        return {"cattle": {"sequence": self.cattle_data["sequence"][idx],
                          "target": self.cattle_data["target"][idx]}}
