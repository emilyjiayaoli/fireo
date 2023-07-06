# fireo 🔥
Model shape debugger for torch. Think torch.summary but better. Streamlining your PyTorch debugging experience.

### Why fireo

**Basic:**
- calculates model's trainable parameter #
- model shape debugging via model initialization + one forward pass given arbitrary inputs
- currently supports pytorch model and yaml config
  
**Bonus:**
- only useful print statements, excluding PyTorch internals.
- auto-tracks and saves local variable shapes w/o manual `print()` statements or debugger
- handles unmultipliable shapes with ease, identifying and printing problematic shapes.
- no modifications to source code needed

## Get Started
### Code Snippet
```python
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
```

### Run existing tests
In root, run:
```python
python test.py
```

## Setup
```
pip install fireo/requirements.txt

```
