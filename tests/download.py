import sys
sys.path.append('/root/autodl-tmp/models/deit_highway')
sys.path.insert(0, '/root/autodl-tmp/')

import torch
# Load model directly
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",
                                                      resume_download=True)
config = MBartConfig.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

print(model)
print(config)