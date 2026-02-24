import os
import os.path
import tempfile
from contextlib import contextmanager
from pathlib import Path
from subprocess import check_output
from typing import Any, Generator

import numpy as np

import pytest

import nbcc
from nbcc.compiler import compile_to_mlir
from nbcc.cutile_backend.backend import CuTileBackend


example_dir = (
    Path(os.path.dirname(nbcc.__file__)) / ".." / "examples" / "cuda_tile"
)


@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


@contextmanager
def compile_mlir(filename: str) -> Generator[Any, None, None]:
    path = example_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        mlir_mod = compile_to_mlir(str(path), be_type=CuTileBackend)
        yield mlir_mod


def test_has_examples():
    assert example_dir.exists()


def test_cuda_tile_to_mlir():
    with compile_mlir("tile_example.spy") as mlir_mod:
        mlir_text = mlir_mod.operation.get_asm()
        assert "entry @spy_tile_example$exported$export_foo" in mlir_text
        assert "entry @spy_tile_example$exported$export_vecadd" in mlir_text
        assert "entry @spy_tile_example$exported$export_ifelse" in mlir_text
        bcbytes = check_output(
            [
                "cuda-tile-translate",
                "--bytecode-version=13.1",
                "--mlir-to-cudatilebc",
                "--no-implicit-module",
            ],
            input=mlir_text.encode(),
        )


def test_cuda_tile_vecadd():
    import torch
    import cuda.tile as ct
    from nbcc.cutile_backend.loader import compiler_context

    with compile_mlir("tile_example.spy") as mlir_mod:
        mlir_text = mlir_mod.operation.get_asm()

    kernel_name = "spy_tile_example$exported$export_vecadd"
    with compiler_context() as cc:
        kernel = cc.compile_kernel(
            mlir_text, kernel_name, (False, False, False)
        )
        x_tensor = torch.arange(16, dtype=torch.float64, device="cuda")
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x_tensor,))

    got = np.asarray(x_tensor.cpu())
    expect = np.arange(got.size, dtype=np.float64) * 2
    np.testing.assert_array_equal(actual=got, desired=expect)
