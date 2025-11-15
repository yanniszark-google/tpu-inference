import jax
import jax.numpy as jnp
import numpy as np
# The Layout class is part of the experimental maps module for defining memory ordering.
from jax.experimental.layout import Layout, Format
# Import SingleDeviceSharding for explicit device placement and layout specification
from jax.sharding import SingleDeviceSharding

# Set a configuration flag to ensure array layout information is present for inspection
# jax.config.update('jax_array', True)

# 1. Define the function (it simply acts as an identity function)
def change_layout(x):
    """
    An identity function used to demonstrate forcing a layout change
    during JIT compilation using in_shardings and out_shardings.
    """
    return x

# 2. Define the Layout specifications
# major_to_minor=(0, 1): Standard C-style layout (row-major for a 2D array)
layout_row_major = Layout(major_to_minor=(0, 1))

# major_to_minor=(1, 0): Reversed/Fortran-style layout (column-major for a 2D array)
layout_col_major = Layout(major_to_minor=(1, 0))

dev = jax.devices()[0]
assert dev.platform == 'tpu', "This script is designed to run on a TPU device."

sharding = SingleDeviceSharding(dev)

print("--- Layout Specifications ---")
print(f"Input Layout (Row-Major): {layout_row_major}")
print(f"Output Layout (Col-Major): {layout_col_major}\n")

# 3. Compile the function using jax.jit with explicit layout constraints.
# The JIT compiler is instructed to expect the input in (0, 1) layout
# and to produce the output in (1, 0) layout.
try:
    change_layout_compiled = jax.jit(
        change_layout,
        in_shardings=Format(layout_row_major, sharding=sharding),
        out_shardings=Format(layout_col_major, sharding=sharding),
    )
except Exception as e:
    print(f"Could not compile: {e}")
    exit(1)

shape = (2, 3)

input_tensor = jax.device_put(
    np.ones(shape),
    Format(layout_row_major, sharding=sharding)
)

print("\n--- Direct Array Placement Demonstration ---")
print("input_tensor (jax.device_put with explicit sharding/layout): Forced Row-Major Layout")
print(f"input_tensor Array:\n{input_tensor}")
print(f"input_tensor Stored Layout: {input_tensor.format.layout}")

# 4. Run the compiled function
output_tensor = change_layout_compiled(input_tensor)

# 6. Output and Verification
print("--- JIT Output Tensor Details ---")
print(f"Output Array (The data itself remains logically the same):\n{output_tensor}")
print(f"Output Array Shape: {output_tensor.shape}")

# Check the layout attribute to see if the forced layout is present.
print(f"Output Array Stored Layout (Forced by JIT): {output_tensor.format.layout}")
print("\nVerification: If the stored layout shows major_to_minor=(1, 0), the layout change inside JIT was successful.")

