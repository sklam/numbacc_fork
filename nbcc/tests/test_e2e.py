import os.path
import subprocess as subp
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import nbcc
from nbcc.compiler import compile

e2e_dir = Path(os.path.dirname(nbcc.__file__)) / ".." / "examples" / "e2e"


@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


def run(path: Path | str) -> str:
    return subp.check_output(str(path), encoding="utf-8")


def run_e2e_test(filename: str, expected_output: str) -> None:
    """Helper function to run an end-to-end test for a given file."""
    path = e2e_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        outpath = dir / "a.out"
        compile(str(path), str(outpath))
        output = run(outpath)

    assert output == expected_output.lstrip()


def test_has_e2e_examples():
    assert e2e_dir.exists()


def test_e2e_ifelse():
    expected = """
b is bigger
endif
23
a is bigger
endif
389
43
"""
    run_e2e_test("e2e_ifelse.spy", expected)


def test_e2e_loops():
    expected = """
loopbody i c
0
0
loopbody i c
1
0
loopbody i c
2
1
loopbody i c
3
3
loopbody i c
4
6
loopbody i c
5
10
loopbody i c
6
15
loopbody i c
7
21
loopbody i c
8
28
loopbody i c
9
36
endloop
10
45
"""
    run_e2e_test("e2e_loops.spy", expected)


def test_e2e_class():
    expected = "42\n"
    run_e2e_test("e2e_class.spy", expected)
