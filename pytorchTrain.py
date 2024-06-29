import torch
from pytorchModel import Model
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split

class myDataset(Dataset):
    def __init__(self, ds) -> None:
        self.ds = ds
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index) -> dict[str,torch.Tensor]:
        return {
            "x": torch.tensor(self.ds[index]["sequence"], dtype=torch.float),
            "y": torch.tensor(self.ds[index]["label"], dtype=torch.float)
        }

sequences = np.load('sequences.npy')
labels = np.load('labels.npy')

assert len(labels) == len(sequences), f"Length of labels and sequences must match. Found {len(labels)} and {len(sequences)}"

dataset = [{"sequence": sequences[i], "label": labels[i]} for i in range(len(labels))]

train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_ds = myDataset(train_ds)
val_ds = myDataset(val_ds)

train_dataloader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=8, shuffle=True)

num_classes = 5
learning_rate = 1e-5
epochs = 100
_, *input_shape = sequences.shape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(input_shape, 128, num_classes).to(device)
for par in model.parameters():
    if par.dim() > 1:
        torch.nn.init.xavier_uniform_(par)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch: {epoch:02d}")
    for batch in batch_iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        output = model(x)
        loss = loss_function(y, output)
        loss.backward()
        optimizer.step()
        batch_iterator.set_postfix({"loss":loss.item()})
    model.eval()

print()
print("*************  Starting Validation  *****************")
model.eval()
batch_iterator = tqdm(val_dataloader, desc=f"Epoch: {epoch:02d}")
for i, batch in enumerate(batch_iterator):
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    output = model(x)
    loss = loss_function(y, output)
    print(f"***** {i} ******", f"output: {torch.argmax(output, dim=1)}", f"expected: {torch.argmax(y, dim=1)}", sep="\n")
    batch_iterator.set_postfix({"loss":loss.item()})

torch.save(model, "torch_model.pt")