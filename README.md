# fireo
Model shape debugger for torch. Think torch.summary but better. Streamlining your PyTorch debugging experience.

### Why fireo

Basic:
- calculates model's trainable parameter #
- model shape debugging via model initialization + one forward pass given arbitrary inputs
  
Bonus:
- only useful print statements, excluding PyTorch internals.
- auto-tracks and saves local variable shapes w/o manual `print()` statements or debugger
- handles unmultipliable shapes with ease, identifying and printing problematic shapes.
- no modifications to source code needed

## Get Started
```python
import fireo.torch_shape_inspector as tsi
from sample_model.models.baseline_classification import MLP_Baseline # import model

inspector_cfgs = {
    # Instantiate the model
    "print_model_modules": True,
    "print_model": False, 
    "print_model_params": True,

    # Inspect the model
    "print_locals_at_forward": True,
    "print_local_vars_at_error": True,
}

inspector = tsi.TorchShapeInspector(model_class=MLP_Baseline, cfgs = inspector_cfgs)
model_cfgs = inspector.parse_config_file(filename="./sample_model/cfgs/baseline_classification_mlp.yaml")
inspector.instantiate_model(cfg = model_cfgs) # instantiate model as normal
inspector.inspect_model(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256))

```

## Setup
```
pip install requirements.txt

```

