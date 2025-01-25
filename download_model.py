from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

# output model structure (resnet50)
print("ResNetForImageClassification")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
with open('./models/cv/resnet50/model.structure', 'w') as f:
    print(model, file=f)


