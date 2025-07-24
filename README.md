# Unsloth LoRA Inference 

This notebook demonstrates how to run inference on a fine-tuned MedGemma model with a LoRA adapter using the Unsloth library.

- **Base Model**: [unsloth/medgemma-4b-pt](https://huggingface.co/unsloth/medgemma-4b-pt)

- **LoRA Adapter**: [huseyincavus/medgemma-4b-guidelines-lora](https://huggingface.co/huseyincavus/medgemma-4b-guidelines-lora)

## How to Run

1. **Install Dependencies:** Open the notebook and run the first code cell to install unsloth and other required packages.

2. **Run All Cells:** Execute the rest of the notebook.

## Core Logic

The key is to use Unsloth's FastLanguageModel for loading the model and adapter to ensure compatibility and performance.

**Generated python**

```python
from unsloth import FastLanguageModel

# Load the 4-bit base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/medgemma-4b-pt",
    load_in_4bit = True,
)

# Apply the LoRA adapter
model.load_adapter("huseyincavus/medgemma-4b-guidelines-lora")

# The model is now ready for inference.
```

# License
This project is licensed under the MIT License. See the LICENSE file for more details.
