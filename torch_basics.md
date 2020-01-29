###Diffence in layers
```python
# torch
a = self.layer1(x)	# [b, 1, 28, 28]  -> [b, 16, 14, 14] (b, c, h, w)
 # tf2
 a = self.layer1(x)  # [b, 28, 28, 1] -> (b, h, w, c)
```
* **Conclusion**: 
TF2 is ch **Last**, torch is Ch **First** 

---
###Torch padding
- Torch does not have padding='same' like tf. 
To add padding='same' in torch, paddiing val is :
```
O = (I +2P - F + 1)/S
P = (O * S - I  + F - 1)/2
```
* **P = (F-1) / 2 ** (same)
ex: F = 5, I = 28, O = 14, S = 2, 
P=(14*2 - 28 + 5 -1)/2 = 2

---
###torch.no_grad():
* https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
* **Conclusion**:
 no_grad() is the faster way, eval is the slower but safer way.

---
###Torch: tensor vs Variable
```python
import torch
from torch.autograd import Variable   # import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])  # data 2 tensor
variable = Variable(tensor, requires_grad=True)  # tensor 2 Variable

print(tensor)  # out tensor...
print(variable)  # out Variable_containing

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)  

# --- do back propagation ---
v_out.backward()
print(variable.grad)  # check 
```
> OUT:
tensor([[1., 2.],
        [3., 4.]])
variable ([[1., 2.],
        [3., 4.]], requires_grad=True)
variable grad([[0.5000, 1.0000],
        [1.5000, 2.0000]])  
        # grad = dv / d(var) = 1/4*d(var * var) = 1/2 * var
        
