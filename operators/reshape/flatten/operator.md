# Flatten


Flatten¶


class torch.nn.Flatten(start_dim=1, end_dim=-1)[source]¶
Flattens a contiguous range of dims into a tensor.
For use with Sequential, see torch.flatten() for details.

Shape:
Input: (∗,Sstart,...,Si,...,Send,∗)(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)(∗,Sstart​,...,Si​,...,Send​,∗),’
where SiS_{i}Si​ is the size at dimension iii and ∗*∗ means any
number of dimensions including none.
Output: (∗,∏i=startendSi,∗)(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)(∗,∏i=startend​Si​,∗).




Parameters

start_dim (int) – first dim to flatten (default = 1).
end_dim (int) – last dim to flatten (default = -1).




Examples::>>> input = torch.randn(32, 1, 5, 5)
>>> # With default parameters
>>> m = nn.Flatten()
>>> output = m(input)
>>> output.size()
torch.Size([32, 25])
>>> # With non-default parameters
>>> m = nn.Flatten(0, 2)
>>> output = m(input)
>>> output.size()
torch.Size([160, 5])





