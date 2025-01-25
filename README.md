# Opedia
A resource for deep learning operators, offering detailed explanations, use cases, and tips for various neural network operations.

# Models
Save model structure and description in yaml format, each directory represents a model, which contains two files
- model.yaml: model configuration
- model.structure: model structure

- [GPT-2](models/nlp/gpt2/model.yaml)
- [ResNet50](models/cv/resnet50/model.yaml)

# Operators
Each operator has a directory, which contains two files:
- operator.md: introduction to the operator, including input/output definition, usage, and examples
- history.yaml: introduction to the operator's usage in different models, including parameters

## Activations

# Automatic analysis program
Help me write a python program, input is the models directory, it will automatically traverse the entire directory, read the structure of each model, and analyze the operators used in it.
Then, it will establish an index for each operator, and maintain a record of which models use the operator and how they are used (parameters).
Finally, it will output to the operators directory, and for each operator, it will automatically create a directory, and download the operator's description from the pytorch website to the operator.md file, and write the operator's usage record to the history.yaml file.