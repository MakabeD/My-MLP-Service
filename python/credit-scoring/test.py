import torch

# creacion de tensores
x = torch.tensor(3.0)
y = torch.tensor(4.0, requires_grad=True)  # probando gradiente activado
w = x * y
print(w)  # ok
# calculo de las derivadas
w.backward()

print(y.grad)
