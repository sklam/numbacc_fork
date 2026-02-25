import tempfile
from nbcc.frontend import frontend


def _compile(src: str):
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".spy", prefix="test", delete=False
    ) as tmpfile:
        tmpfile.write(src)
        tmpfile.flush()
        path = tmpfile.name
        return frontend(path)


def test_basic_if():
    source = """
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    if a < b:
        c = b
    else:
        c = a

    print(c)
"""
    _compile(source)


def test_basic_if_nested_1():
    source = """
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    if a < b:
        c = b
        if a < b * 2:
            c = c + 1
        else:
            return
    else:
        c = a

    print(c)
"""
    _compile(source)


def test_basic_while():
    source = """
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    while a < b:
        a = a + 1

    print(a)
"""
    _compile(source)


def test_basic_for_break():
    source = """
from _range import range
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    for i in range(a, b):
        if i == b - a:
            print("endloop")
            break
        print(i)

    print(a)
"""
    _compile(source)


def test_basic_for_continue():
    source = """
from _range import range
def main() -> None:
    a: i32
    b: i32
    c: i32

    a = 1
    b = 2
    for i in range(a, b):
        if i == b - a:
            print("endloop")
            continue
        print(i)

    print(a)
"""
    _compile(source)
