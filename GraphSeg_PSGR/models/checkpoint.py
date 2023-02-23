import torch
import warnings


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, input_tensor, param_tensor):
        ctx.run_function = run_function
        ctx.input_tensor = input_tensor
        ctx.input_param = param_tensor
        with torch.no_grad():
            output_tensors = ctx.run_function(ctx.input_tensor)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        temp = ctx.input_tensor
        ctx.input_tensor = temp.detach()
        ctx.input_tensor.requires_grad = temp.requires_grad
        with torch.enable_grad():
            output_tensor = ctx.run_function(ctx.input_tensor)
        input_grads = torch.autograd.grad(output_tensor, ctx.input_tensor + ctx.input_param, output_grads, allow_unused=True)
        return (None, None) + input_grads