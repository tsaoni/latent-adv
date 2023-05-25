from lib import *


# Define a custom hook function
def hook_fn(module, input, output):
    print(f"Module: {module}")
    print(f"Input: {input}")
    print(f"Output: {output}")

# Create a sample model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

import pdb 
pdb.set_trace()

def myforward(self, x, forward_fn):
    out = forward_fn(self, x)
    print("out1: ", out)
    out += 1.
    return out

input_tensor = torch.randn(1, 10)
model = MyModel()
forward_fn = copy.deepcopy(model.forward)
model.forward = myforward
model(input_tensor, forward_fn)

# Register the hook to a specific layer/module
hook = model.fc.register_forward_hook(hook_fn)

# Forward pass through the model

output = model(input_tensor)

# Remove the hook after using it
hook.remove()


class myDataset(Dataset):
    def __init__(self):
        self.data = [i for i in range(101)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
    def collate_fn(self, batch):
        print('batch: ', batch)
        return batch
    
if __name__ == '__main__':

    import pdb 
    pdb.set_trace()
    model = nn.Linear(3, 2)
    model.eval()
    a =nn.Parameter(torch.randn(10, 3), requires_grad=True)
    output = model(a)
    target = torch.ones(10, dtype=torch.int64)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(output, target)
    loss.backward()

    dataset = myDataset()
    batch_size = 3
    num_workers = 1

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        num_workers=num_workers,
        sampler=None,
        drop_last=True, 
    )
    data_iter = iter(loader)
    try:
        while True:
            next(data_iter)
    except StopIteration:
        print('eval epoch end. ')
