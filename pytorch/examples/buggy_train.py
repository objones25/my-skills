"""A training script with eight planted bugs, for examples/05-reviewing-a-training-script.md.

Every line marked BUG is a mistake the skill's references describe. It runs without
crashing on CPU, which is the point: none of these raise.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(self, d=32, n_classes=10):
        super().__init__()
        self.blocks = [nn.Linear(d, d) for _ in range(2)]        # BUG: plain list
        self.head = nn.Linear(d, n_classes)
        self.temperature = torch.ones(1)                          # BUG: unregistered tensor

    def forward(self, x):
        for blk in self.blocks:
            x = F.dropout(F.relu(blk(x)), p=0.1)                  # BUG: dropout ignores eval()
        return F.softmax(self.head(x) / self.temperature, dim=-1)


def main():
    device = torch.device("cpu")
    x, y = torch.randn(512, 32), torch.randint(0, 10, (512,))
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    model = Net().to(device)
    opt = torch.optim.AdamW(set(model.parameters()), lr=1e-3)    # BUG: set of params
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(3):
        for xb, yb in loader:
            xb.to(device)                                         # BUG: result discarded
            probs = model(xb)
            loss = loss_fn(probs, yb)                             # BUG: softmax output into CE
            loss.backward()
            opt.step()                                            # BUG: no zero_grad anywhere
            history.append(loss)                                  # BUG: keeps every graph alive
    model.eval()
    print("val acc:", (model(x).argmax(1) == y).float().mean().item())


if __name__ == "__main__":
    main()
