import torch
import torch.nn as nn
import torch.optim as optim
print(torch.__version__)


distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

model=nn.Sequential(nn.Linear(1,1))

loss_function=nn.MSELoss()
optimizer=optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    output=model(distances)
    loss=loss_function(output,times)
    loss.backward()
    optimizer.step()
    if (epoch+1)%50==0:
        print(f"The loss on epoch {epoch} is {loss}")
