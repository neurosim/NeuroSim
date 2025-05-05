import torch
from torch.nn.functional import unfold

import torch
from torch.nn.functional import unfold

import torch

def conv2d_as_matmul(quant_input, scale_input, quant_weight, scale_weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
    """
    Performs 2D convolution by converting it into a matrix multiplication where each row of the weight matrix represents a filter.

    Arguments:
        quant_input (torch.Tensor): Quantized input tensor of shape (N, C_in, H_in, W_in).
        scale_input (torch.Tensor): Scaling factor of the input tensor.
        quant_weight (torch.Tensor): Quantized weight tensor of shape (C_out, C_in/groups, kH, kW).
        scale_weight (torch.Tensor): Scaling factor of the weight tensor.
        bias (torch.Tensor): Bias tensor of shape (C_out).
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        dilation (int or tuple): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.

    Returns:
        torch.Tensor: Output tensor of shape (N, C_out, H_out, W_out).
    """
    # Convert inputs and weights to integers
    quant_input = quant_input * scale_input
    quant_weight = quant_weight * scale_weight

    scale = (scale_input * scale_weight).view(1, -1, 1, 1)

    # Shift inputs and weights so they are positive
    shift_input = -torch.min(quant_input)
    shift_weight = -torch.min(quant_weight)
    # quant_input = quant_input + shift_input
    # quant_weight = quant_weight + shift_weight
    
    # Unfold the input tensor into a matrix
    input_unfolded = torch.nn.functional.unfold(quant_input, kernel_size=quant_weight.shape[-2:], dilation=dilation,
                                                 padding=padding, stride=stride).transpose(1, 2)

    # Transpose the weight tensor and reshape it into a matrix
    weight_matrix = quant_weight.view(quant_weight.shape[0], -1).t()

    # Perform matrix multiplication with input_matrix on the left and weight_matrix on the right
    output = input_unfolded @ weight_matrix

    # Reshape the output tensor
    N = quant_input.shape[0]
    C_out = quant_weight.shape[0]
    H_out = (quant_input.shape[-2] + 2 * padding[0] - dilation[0] * (quant_weight.shape[-2] - 1) - 1) // stride[0] + 1
    W_out = (quant_input.shape[-1] + 2 * padding[1] - dilation[1] * (quant_weight.shape[-1] - 1) - 1) // stride[1] + 1
    output = output.view(N, H_out, W_out, C_out).permute(0, 3, 1, 2)

    # Add bias if provided
    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    # Rescale
    output = output / scale

    return output



# Set up test inputs
x = torch.randn(1, 3, 8, 8, device='cuda')
w = torch.randn(4, 3, 3, 3, device='cuda')
b = torch.randn(4, device='cuda')

x_scale = torch.tensor(0.1, device='cuda')
w_scale = torch.randn(4,1,1,1, device='cuda')

# Compute convolution using PyTorch's conv2d function
y1 = torch.nn.functional.conv2d(x, w, bias=b, stride=2, padding=1, dilation=2)

# Compute convolution using our conv2d_as_matmul function
y2 = conv2d_as_matmul(x, x_scale, w, w_scale, bias=b, stride=(2,2), padding=(1,1), dilation=(2,2))

# Check that the output shapes match
assert torch.allclose(y1, y2)
