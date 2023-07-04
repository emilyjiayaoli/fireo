import torch
import torch_shape_inspector as tsi

from sample_model.models.baseline_classification import Conv_Baseline, MLP_Baseline, ResNet18_Baseline

cfgs = {
    # Instantiate the model
    "print_model_modules": True,
    "print_model": False, 
    "print_model_params": True,

    # Inspect the model
    "print_locals_at_forward": True,
    "print_local_vars_at_error": True,
}

inspector = tsi.TorchShapeInspector(model_class=MLP_Baseline, cfgs = cfgs)
model_cfgs = inspector.parse_config_file(filename="./sample_model/cfgs/baseline_classification_mlp.yaml")
inspector.instantiate_model(cfg = model_cfgs)
inspector.inspect_model(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256))

