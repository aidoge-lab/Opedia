# Identity


Identity¶


class torch.nn.Identity(*args, **kwargs)[source]¶
A placeholder identity operator that is argument-insensitive.

Parameters

args (Any) – any argument (unused)
kwargs (Any) – any keyword argument (unused)




Shape:
Input: (∗)(*)(∗), where ∗*∗ means any number of dimensions.
Output: (∗)(*)(∗), same shape as the input.



Examples:
>>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 20])



