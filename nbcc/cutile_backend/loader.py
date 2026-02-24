from typing import Generator
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from subprocess import check_call
import tempfile

from cuda.tile._cext import TileDispatcher
from cuda.tile import _compile
from cuda.tile._compiler_options import CompilerOptions


CUDA_TILE_TRANSLATE_PATH = "cuda-tile-translate"


@dataclass
class CompilerContext:
    """

    Note:
        Keep this alive until all kernel launches are invoked. The TileBC files
        are compiled lazily. The cubin is stored in the same temporary
        directory.
    """

    dirpath: Path
    bytecode_version: str = "13.1"
    cuda_tile_translate_path: str = CUDA_TILE_TRANSLATE_PATH

    def compile_to_cubin(self, mlir_text: str) -> str:
        tilebc_file = str(self.dirpath / "kernel.tilebc")
        mlir_file = str(self.dirpath / "kernel.mlir")
        with open(mlir_file, "w") as fout:
            print(mlir_text, file=fout)

        sm_arch = _compile.get_sm_arch()
        check_call(
            [
                self.cuda_tile_translate_path,
                mlir_file,
                f"--bytecode-version={self.bytecode_version}",
                "--mlir-to-cudatilebc",
                "--no-implicit-module",
                "-o",
                tilebc_file,
            ]
        )
        cubin_path = _compile.compile_cubin(
            tilebc_file, CompilerOptions(), sm_arch, None
        )
        return cubin_path

    def compile_kernel(
        self,
        mlir_text: str,
        kernel_name: str,
        arg_constant_flags: tuple[bool, ...],
    ):
        cubin_path = self.compile_to_cubin(mlir_text)
        compile_callback = HackCompileCallback(str(cubin_path), kernel_name)
        kernel = TileDispatcher(arg_constant_flags, compile_callback)
        return kernel


@contextmanager
def compiler_context() -> Generator[CompilerContext, None, None]:
    with tempfile.TemporaryDirectory() as dirpathstr:
        yield CompilerContext(dirpath=Path(dirpathstr))


@dataclass
class HackCompileCallback:
    # From: https://github.com/NVIDIA/cutile-python/blob/361048e152636435ec2a660e650481acbd001745/test/test_bytecode.py#L103-L113
    cubin_path: str
    func_name: str

    def __call__(self, args, ctx):
        return self.cubin_path, self.func_name
