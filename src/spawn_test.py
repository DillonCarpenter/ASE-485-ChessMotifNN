from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx

if __name__ == "__main__":
    ds = TestDataset()

    dl = DataLoader(
        ds,
        batch_size=10,
        num_workers=4
    )

    for x in dl:
        print(x)
        break