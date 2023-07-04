import torch
import torch_shape_inspector as tsi

from models.baseline_classification import Conv_Baseline, MLP_Baseline, ResNet18_Baseline

# Instantiate the model
cfgs = {
    "print_locals_at_forward": True,
    "print_model_modules": True,
    "print_model": False, 
    "print_model_params": True,
    "model_cfgs_default": True
}


inspector = tsi.TorchShapeInspector(model_class=ResNet18_Baseline, cfgs = cfgs)
model_cfgs = inspector.parse_config_file(filename="./cfgs/baseline_classification_cnn.yaml")
inspector.instantiate_model(cfg = model_cfgs)
inspector.inspect_model(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256))

