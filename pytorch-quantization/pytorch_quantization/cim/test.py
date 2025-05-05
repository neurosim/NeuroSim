import torch
import math

def convert_to_n_ary(dec_matrix, base, bits=8):
    # expand each column in the decimal matrix to an n-ary number
    rows, cols = dec_matrix.shape
    dec_matrix = dec_matrix.flatten().reshape(-1,1).int()

    max_val = 2**bits
    num_digits = math.ceil(math.log(max_val, base))

    n_ary = base**torch.arange(num_digits).flip(0)

    out = dec_matrix // n_ary % base

    return out.reshape(rows, num_digits*cols)

def test_convert_to_nary():
    dec_matrix = torch.tensor([[1, 2], [3, 4]])

    # Test conversion to base 2
    base2_result = convert_to_n_ary(dec_matrix, 2)
    print(base2_result)
    # assert torch.all(base2_result == torch.tensor([[0, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0]]))

    # Test conversion to base 4
    base4_result = convert_to_n_ary(dec_matrix, 4)
    print(base4_result)
    # assert torch.all(base4_result == torch.tensor([[0, 1, 0, 2], [0, 3, 1, 0]]))

    # Test conversion to base 8
    base8_result = convert_to_n_ary(dec_matrix, 8)
    print(base8_result)
    # assert torch.all(base8_result == torch.tensor([[0, 1, 0, 2], [0, 3, 0, 4]]))
# test_convert_to_nary()

def test_reshape_matmul(input_tensor, weight_tensor):
    # Direct matrix multiplication
    direct_output = torch.matmul(input_tensor, weight_tensor)

    # Reshaped matrix multiplication
    batch_size, input_size = input_tensor.shape
    output_size = weight_tensor.shape[1]
    group_size = 4
    num_groups = input_size // group_size

    reshaped_input = input_tensor.view(batch_size, num_groups, group_size)
    reshaped_weight = weight_tensor.view(num_groups, group_size, output_size)

    reshaped_output = torch.einsum('ijk,jkl->ijl', reshaped_input, reshaped_weight)

    # Check if outputs are equal
    assert torch.allclose(direct_output, reshaped_output), "Outputs are not equal"

# Example usage
input_tensor = torch.randn(100, 144)
weight_tensor = torch.randn(144, 520)

test_reshape_matmul(input_tensor, weight_tensor)