import numpy as np
import torch
import cuda.tile as ct
from nbcc.compiler import compile_to_mlir
from nbcc.cutile_backend.loader import compiler_context
from nbcc.cutile_backend.backend import CuTileBackend

def main():

    mlir_mod =compile_to_mlir("tile_loop_wrapper.spy", be_type=CuTileBackend)
    mlir_text = mlir_mod.operation.get_asm()
    print(mlir_text)

    kernel_name = "spy_tile_loop_wrapper$exported$export_ifelse"
    with compiler_context() as cc:
        kernel = cc.compile_kernel(
            mlir_text, kernel_name, (False, False, False)
        )
        nelem = 128
        # Test with bypass compute ON
        x_tensor = torch.arange(nelem, dtype=torch.float64, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x_tensor, True))

        np.testing.assert_array_equal(
            np.asarray(x_tensor.cpu()), np.arange(nelem, dtype=np.float64) * 2
        )

        # Repeat test without the bypass compute
        x_tensor = torch.arange(nelem, dtype=torch.float64, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x_tensor, False))

        np.testing.assert_array_equal(
            np.asarray(x_tensor.cpu()), 4 * np.arange(nelem, dtype=np.float64)
        )

        print(x_tensor)


if __name__ == "__main__":
    main()
