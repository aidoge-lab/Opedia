#!/usr/bin/env python3
import os
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Set
import logging
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OperatorAnalyzer:
    def __init__(self, models_dir: str, operators_dir: str):
        self.models_dir = Path(models_dir)
        self.operators_dir = Path(operators_dir)
        self.operator_usage = {}  # {operator_name: {model_name: [usage_info]}}
        
    def analyze_models(self):
        """Traverse through all model.structure files and analyze operators."""
        for model_file in self.models_dir.rglob('model.structure'):
            try:
                with open(model_file, 'r') as f:
                    model_data = f.read()
                model_name = str(model_file.parent.relative_to(self.models_dir))
                self._analyze_model_structure(model_data, model_name)
            except Exception as e:
                logger.error(f"Error processing {model_file}: {str(e)}")

    def _analyze_model_structure(self, model_data: str, model_name: str):
        """Analyze a single model's structure and extract operator information."""
        def extract_operators(text: str):
            """ model structure example:
            GPT2Model(
                (wte): Embedding(50257, 768)
                (wpe): Embedding(1024, 768)
                (drop): Dropout(p=0.1, inplace=False)
                (h): ModuleList(
                    (0-11): 12 x GPT2Block(
                    (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (attn): GPT2Attention(
                        (c_attn): Conv1D()
                        (c_proj): Conv1D()
                        (attn_dropout): Dropout(p=0.1, inplace=False)
                        (resid_dropout): Dropout(p=0.1, inplace=False)
                    )
                    (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    (mlp): GPT2MLP(
                        (c_fc): Conv1D()
                        (c_proj): Conv1D()
                        (act): NewGELUActivation()
                        (dropout): Dropout(p=0.1, inplace=False)
                    )
                    )
                )
                (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                )
            """

            # 使用正则表达式提取算子及其参数
            import re
            for line in text.split('\n'):
                match = re.search(r"(\w+)\((.*?)\)", line)
                if match:
                    op_name, params = match.groups()
                    if op_name not in self.operator_usage:
                        self.operator_usage[op_name] = {}
                    if model_name not in self.operator_usage[op_name]:
                        self.operator_usage[op_name][model_name] = []
                    # 解析参数
                    param_dict = {}
                    for param in params.split(', '):
                        if '=' in param:
                            key, value = param.split('=')
                            param_dict[key.strip()] = value.strip()
                    self.operator_usage[op_name][model_name].append(param_dict)

        extract_operators(model_data)

    def _fetch_pytorch_docs(self, operator_name: str) -> str:
        """Fetch operator documentation from PyTorch website."""
        base_url = "https://pytorch.org/docs/stable/generated/"
        try:
            # Convert operator name to PyTorch documentation format
            doc_name = f"torch.nn.{operator_name}"
            response = requests.get(f"{base_url}{doc_name}.html")
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract the main content
                content = soup.find('div', {'class': 'section'})
                if content:
                    return content.get_text()
            return f"Documentation not found for {operator_name}"
        except Exception as e:
            logger.error(f"Error fetching documentation for {operator_name}: {str(e)}")
            return f"Error fetching documentation: {str(e)}"

    def generate_documentation(self):
        """Generate documentation for each operator."""
        for operator_name, usage_info in self.operator_usage.items():
            # 创建算子类型目录
            operator_type = self._get_operator_type(operator_name)
            operator_dir = self.operators_dir / operator_type / operator_name.lower()
            operator_dir.mkdir(parents=True, exist_ok=True)

            # 生成operator.md
            docs = self._fetch_pytorch_docs(operator_name)
            with open(operator_dir / 'operator.md', 'w') as f:
                f.write(f"# {operator_name}\n\n")
                f.write(docs)

            # 生成history.yaml
            with open(operator_dir / 'history.yaml', 'w') as f:
                yaml.dump({operator_name: usage_info}, f, default_flow_style=False)

    def _get_operator_type(self, operator_name: str) -> str:
        """Get the type category of an operator."""
        type_mapping = {
            'Conv': 'convolution',
            'BatchNorm': 'normalization',
            'ReLU': 'activation',
            'MaxPool': 'pooling',
            'AdaptiveAvgPool': 'pooling',
            'Linear': 'linear',
            'Flatten': 'reshape',
            'Identity': 'activation'
        }
        
        for prefix, op_type in type_mapping.items():
            if operator_name.startswith(prefix):
                return op_type
        return 'other'

def main():
    models_dir = "models"
    operators_dir = "operators"
    
    analyzer = OperatorAnalyzer(models_dir, operators_dir)
    logger.info("Analyzing models...")
    analyzer.analyze_models()
    logger.info("Generating documentation...")
    analyzer.generate_documentation()
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
