from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

# output model structure (resnet50)
# print("ResNetForImageClassification")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# with open('./models/cv/resnet50/model.structure', 'w') as f:
#     print(model, file=f)

# output model structure (gpt2)
print("GPT2Model")
from transformers import GPT2Tokenizer, GPT2Model
model = GPT2Model.from_pretrained('gpt2')
with open('./models/nlp/gpt2/model.structure', 'w') as f:
    print(model, file=f)


