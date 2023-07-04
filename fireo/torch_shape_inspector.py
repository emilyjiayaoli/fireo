import inspect
import torch
import argparse
import traceback
import yaml
import os

class TorchShapeInspector():
    def __init__(self, model_class, cfgs=None):
        self.cfgs = cfgs
        self.model_class = model_class

        self.old_forward = self.model_class.forward
        self.updated_forward_source = None

        if self.cfgs["print_locals_at_forward"]:
            self.model_class.forward = self.__get_updated_forward(self.old_forward) #store_locals(self, self.updated_forward)
        
        self.model = None
        self.model_modules = None

    def _remove_initial_indentation(self, text):
        '''Remove the initial indentation from a block of text'''
        lines = text.split('\n')
        # Remove leading/trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
            
        # Determine the number of leading whitespace characters in the first line
        initial_indentation = len(lines[0]) - len(lines[0].lstrip())
        # Remove this many leading whitespace characters from every line
        lines = [line[initial_indentation:] if len(line) > initial_indentation else '' for line in lines]
        return '\n'.join(lines)

    def __get_updated_forward(self, old_forward):
        source = inspect.getsource(old_forward)
        return_indices = [i for i in range(len(source)) if source.startswith("return", i)]

        new_command = " locals(), "

        updated_source = source
        for i in range(len(return_indices)):
            cur_return_idx = return_indices[i] + (i * len(new_command)) + len("return")
            updated_source = updated_source[:cur_return_idx] + new_command + updated_source[cur_return_idx:]

        updated_source = self._remove_initial_indentation(updated_source)

        namespace = self.old_forward.__globals__

        self.updated_forward_source = updated_source
        
        if self.cfgs["save_updated_forward_fn_path"] is not None:
            with open(os.path.join(self.cfgs["save_updated_forward_fn_path"], 'updated_source.py'), 'w') as f:
                f.write(updated_source)

        compiled = compile(updated_source, filename="<self.model.forward>", mode="exec")
        exec(compiled, namespace)
        
        return namespace['forward']

    def _print_fn_call_stack(self, stack_trace):
        '''Prints the function call stack'''
        print("fn_call_stack")
        for i in range(len(stack_trace)):
            frame = stack_trace[i]
            print("\t", i, ":", frame.function, "in", frame.filename, "at line", frame.lineno)

    def _get_fn_line_from_path(self, file_path, line_number:int):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return lines[line_number - 1]
    
    def _get_fn_line(self, source:str, line_number:int):
        lines = source.splitlines()
        return lines[line_number - 1]
    
    def _print_local_vars(self, local_vars:dict, fn_name=None, line_number=None, file_path=None):
        '''Prints the local variables. If fn_name, line_number, and file_path are provided, it prints the local variables at the time of error. 
            Else, it prints the local variables at the time of the successful forward pass.'''
        if fn_name is None:
            print("Printing local variables...")
        else:
            print("\tPrinting local variables...", "in", fn_name, "at line", line_number, "in", file_path)

        print("\t* Local variables keys:", local_vars.keys())
        for var_name, var_value in local_vars.items():
            if var_name == "self":
                continue
            var_name_type = type(var_name)
            if isinstance(var_value, torch.Tensor):
                print("\t---", var_name, ":", var_value.shape)
            elif isinstance(var_value, list) or isinstance(var_value, tuple):
                print("\t---", var_name, ":", len(var_value), "elements", "Type:", type(var_value))
            elif isinstance(var_value, dict):
                print("\t--", var_name, ":", "keys", var_value.keys(), "Type:", type(var_value))
            else:
                print("\t---", var_name, ":", var_value, "Type:", type(var_value))

    def _dive_deeper(self, stack_trace):
        '''Given the stack_trace, this method attempts to dive deeper into the error and print out the local variables in user defined functions at the time of error'''
        
        print("\nLet's dive_deeper.... tracing back to where the error occured in user defined code...")
        
        for i in range(len(stack_trace))[::-1]: # iterate backwards to see where the error occured before stepping into torch
            frame = stack_trace[i]
            local_vars = frame[0].f_locals
            if "/torch/" not in frame.filename and "/torch_shape_inspector.py" not in frame.filename:
                if frame.filename == "<self.model.forward>":
                    # print("here:", frame.function, frame.filename, frame.lineno)
                    line = self._get_fn_line(self.updated_forward_source, frame.lineno)
                else:
                    line = self._get_fn_line_from_path(frame.filename, frame.lineno)

                print("\n\nError that occured (before stepping into torch) was:")
                print("\t\t", line, "\tat line", frame.lineno, "in", frame.filename)

                self._print_local_vars(local_vars, frame[3], frame[2], frame[1])


    def _print_model_params(self):
        """
        Prints the total number of parameters in the model. If verbosity is turned on in the configurations,
        it also prints the architecture of the model.

        This method also checks if there's more than one GPU being used. If so, it prints the total parameters
        divided by two, else it prints the total parameters.
        """
        print('---------- Network Parameters -------------')
        total_params = 0
        for name, module in self.model.named_children():
            num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"[Module {name}] # parameters: {num_params / 1e6:.3f} M")
            total_params += num_params
        
        print(f'Total parameter count: {total_params / 1e6:.3f} M')


    def parse_config_file(self, filename):
        '''Parses a YAML configuration file and returns a dictionary.'''
        with open(filename, 'r') as f:
            config = yaml.safe_load(f)
        
        parser = argparse.ArgumentParser()
        for key, value in config.items():
            if type(value) is bool:
                parser.add_argument(f'--{key}', default=value, action=argparse.BooleanOptionalAction)
            else:
                parser.add_argument(f'--{key}', type=type(value), default=value)

        # Parse the arguments
        args = parser.parse_args()
        return args


    def instantiate_model(self, *args, **kwargs):
        '''Instantiates the model'''
        try:
            print('---------- Instantiating Model... -------------')
            self.model = self.model_class(*args, **kwargs)
            print('Success!')
            if self.cfgs["print_model"]:
                print("Model:", self.model)

            if self.cfgs["print_model_params"]:
                self._print_model_params()
            
            self.model_modules = vars(self.model)["_modules"]
            if self.cfgs["print_model_modules"]:
                print("Model attributes:", self.model_modules.keys())

        except Exception as e:
            print(f"Instantiated model unsuccessfully. Caught an error: {e}, at line {inspect.trace()[-1][2]} in {inspect.trace()[-1][1]}")
            traceback.print_exc()


    def inspect_model(self, *inputs):
        '''Attempts forward pass through model to inspect shapes of tensors in model
        '''
        print('---------- Inspecting Model... -------------')
        if self.model is None:
            print("Unsuccessful. Model has not been instantiated. Please call instantiate_model() first.")
        else:
            try:
                outputs = self.model(*inputs)
                print('Successful forward pass.')
                if len(outputs) == 1:
                    raise ValueError("Model only has one output in forward. make sure to 'return locals(), output' ")
                else:
                    local_vars, output, = outputs

                if self.cfgs["print_locals_at_forward"]:
                    print("Successfully inspected model. Printing local variables...")
                    self._print_local_vars(local_vars) #not sure abt the last 2/3 args
                    print("\nOUTPUT SHAPE:", output.shape)

            except Exception as e:
                stack_trace = inspect.trace() # list of FrameInfo objects
                frame = stack_trace[-1] # get frame at which error occured, is a list [FrameInfo, 26, fn3, [error], ...]
                print(f"Inspection Failed. Caught an error ", e, "in the method:", frame[3], "at line", frame[2], "in", frame[1])
                traceback.print_exc()

                print("Local variables at the moment of error:")
                local_vars = frame[0].f_locals
                self._print_local_vars(local_vars=frame[0].f_locals, fn_name=frame[3], line_number=frame[2], file_path=frame[1])


                if self.cfgs["print_fn_call_stack"]:
                    self._print_fn_call_stack(stack_trace)
                
                if self.cfgs["print_local_vars_at_error"]:
                    self._dive_deeper(stack_trace)
            






