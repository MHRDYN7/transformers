import os

import torch


model_name = 'dinov2_vitb14'
# load original model from torch hub

path = "transformers/src/transformers/models/dnv2/model.pt"

if not os.path.exists(path):
    original_model = torch.hub.load("facebookresearch/dinov2", model_name)
    original_model.eval()
    torch.save(original_model, path)
    print("done!")

# load state_dict of original model, remove and rename some keys
#state_dict = torch.load('model.pt')
#state_dict = original_model.state_dict()

#torch.save(original_model, 'src/transformers/models/dnv2/model_architecture.pt')
print("ok")
model = torch.load(path)

print("ok2")
for key in model.state_dict():
    print(key)
