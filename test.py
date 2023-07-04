import torch
import fireo.torch_shape_inspector as tsi

from example_model import MLP_Baseline

cfgs = {
    # Instantiate the model
    "print_model_modules": True,
    "print_model": False, 
    "print_model_params": True,
    "save_updated_forward_fn_path": 'outputs/',

    # Inspect the model
    "print_locals_at_forward": True,
    "print_local_vars_at_error": True,
    "print_fn_call_stack": True,
}

inspector = tsi.TorchShapeInspector(model_class=MLP_Baseline, cfgs = cfgs)
model_cfgs = inspector.parse_config_file(filename="./example_model_cfg.yaml")
inspector.instantiate_model(cfg = model_cfgs)

# should bug out, since 2nd input is supposed to have 1 channel, not 3
inspector.inspect_model(torch.randn(model_cfgs.batch_size, 3, 256, 256), torch.randn(model_cfgs.batch_size, 1, 256, 256)) 

# should run successfully
inspector.inspect_model(torch.randn(model_cfgs.batch_size, 3, 256, 256), torch.randn(model_cfgs.batch_size, 3, 256, 256))

