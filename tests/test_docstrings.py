"""Every public module, class, function and method of the package has a docstring.

The package is a library: its docstrings are the reference a user reads in an editor, and the
Google-style sections in them (``Args:``, ``Returns:``, ``Raises:``) are where the contract of each
call is written down. A name that is public but undocumented is a gap in that contract.

What counts as public: a name that does not start with an underscore (``__init__`` aside), in a
module of the package, or of a subpackage at any depth, that does not either, defined in that
module rather than imported into it. The generated protobuf code is excluded: it is not written
here.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest

import pipecat_spatialreal

# Subpackages that hold no hand-written code.
GENERATED = {"proto"}


def _doc_of(obj: object) -> str:
    """The docstring ``obj`` itself carries: an inherited one does not count.

    A dataclass without a docstring is given a generated one that only repeats its signature, which
    is not documentation either.
    """
    doc = obj.__doc__ or ""
    if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
        signature = str(inspect.signature(obj)).replace(" -> None", "")
        if doc == f"{obj.__name__}{signature}":
            return ""
    return doc.strip()


def _members(cls: type) -> Iterator[tuple[str, object]]:
    """The public functions and classes ``cls`` defines itself, as ``(name, object)``.

    ``__init__`` is included, since the constructor's arguments are documented there; a
    dataclass's generated one is not, because its class docstring lists the fields. A property is
    its getter.
    """
    for name, member in vars(cls).items():
        if name.startswith("_") and name != "__init__":
            continue
        if name == "__init__" and dataclasses.is_dataclass(cls):
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        elif isinstance(member, property):
            member = member.fget
        if inspect.isfunction(member) or inspect.isclass(member):
            yield name, member


def undocumented(name: str, obj: object) -> list[str]:
    """``name`` if ``obj`` has no docstring of its own, then the same for what a class defines.

    Args:
        name: The dotted name to report ``obj`` under.
        obj: A class or function.

    Returns:
        The names with no docstring, ``obj`` first, then its members in the order defined.
    """
    found = [] if _doc_of(obj) else [name]
    if inspect.isclass(obj):
        for member_name, member in _members(obj):
            found += undocumented(f"{name}.{member_name}", member)
    return found


def missing_docstrings(module: types.ModuleType) -> list[str]:
    """The dotted names in ``module`` that are public and have no docstring of their own.

    Args:
        module: The module to check.

    Returns:
        The module itself if it has none, then each class and function defined in it, with the
        members of the classes, in the order they appear.
    """
    found = [] if _doc_of(module) else [module.__name__]
    for name, obj in vars(module).items():
        if name.startswith("_") or not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if obj.__module__ == module.__name__:  # defined here, not imported
            found += undocumented(f"{module.__name__}.{name}", obj)
    return found


def public_modules(package: types.ModuleType = pipecat_spatialreal) -> list[types.ModuleType]:
    """``package`` and every hand-written module below it, subpackages at any depth, imported.

    A name that starts with an underscore, and the generated subpackages, are left out together
    with everything below them.

    Args:
        package: The package to walk.

    Returns:
        The package first, then its modules and subpackages.
    """
    modules = [package]
    prefix = f"{package.__name__}."
    for info in pkgutil.walk_packages(package.__path__, prefix):
        parts = info.name.removeprefix(prefix).split(".")
        if not any(part.startswith("_") or part in GENERATED for part in parts):
            modules.append(importlib.import_module(info.name))
    return modules


@pytest.mark.parametrize("module", public_modules(), ids=lambda module: module.__name__)
def test_every_public_name_has_a_docstring(module):
    assert missing_docstrings(module) == []


def test_the_walk_reaches_every_hand_written_module():
    root = Path(pipecat_spatialreal.__file__).parent
    on_disk = {pipecat_spatialreal.__name__}
    for path in root.rglob("*.py"):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        if not any(part.startswith("_") or part in GENERATED for part in parts):
            on_disk.add(".".join([pipecat_spatialreal.__name__, *parts]))
    walked = {module.__name__ for module in public_modules()}
    assert walked == on_disk  # a module the walk misses, or one it invents, is named here
    assert "pipecat_spatialreal.processor" in walked  # and the walk is not empty


def test_the_walk_goes_down_subpackages_but_not_into_generated_or_private_ones(
    tmp_path, monkeypatch
):
    layout = {
        "__init__.py": '"""Doc."""',
        "top.py": '"""Doc."""',
        "sub/__init__.py": '"""Doc."""',
        "sub/deep.py": "def undocumented(): ...",
        "sub/inner/__init__.py": '"""Doc."""',
        "sub/inner/leaf.py": '"""Doc."""',
        "sub/_hidden/__init__.py": "",
        "proto/__init__.py": "",
        "proto/generated.py": "def undocumented(): ...",
        "_private.py": "",
    }
    for relative, text in layout.items():
        path = tmp_path / "walkcontrol" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        walked = public_modules(importlib.import_module("walkcontrol"))
        assert {module.__name__ for module in walked} == {
            "walkcontrol",
            "walkcontrol.top",
            "walkcontrol.sub",
            "walkcontrol.sub.deep",
            "walkcontrol.sub.inner",
            "walkcontrol.sub.inner.leaf",
        }
        assert walked[0].__name__ == "walkcontrol"
        # The two gaps are two levels down, and only the hand-written one is reported.
        assert sorted(name for module in walked for name in missing_docstrings(module)) == [
            "walkcontrol.sub.deep",
            "walkcontrol.sub.deep.undocumented",
        ]
    finally:
        for name in [n for n in sys.modules if n == "walkcontrol" or n.startswith("walkcontrol.")]:
            del sys.modules[name]


def test_the_check_looks_at_the_processors_constructor_and_overrides():
    processor = importlib.import_module("pipecat_spatialreal.processor")
    members = dict(_members(processor.SpatialRealAvatarProcessor))
    assert {"__init__", "setup", "cleanup", "process_frame"} <= members.keys()
    assert "_open_turn" not in members


def test_the_check_finds_what_is_missing():
    class Plain:
        def method(self): ...

        def _private(self): ...

        @property
        def prop(self): ...

        @staticmethod
        def make(): ...

        class Nested: ...

    @dataclasses.dataclass
    class Record:
        field: int

    @dataclasses.dataclass
    class Documented:
        """Has one."""

        field: int

    class Constructed:
        """Has one."""

        def __init__(self): ...

    def function(): ...

    assert undocumented("Plain", Plain) == [
        "Plain",
        "Plain.method",
        "Plain.prop",
        "Plain.make",
        "Plain.Nested",
    ]
    assert undocumented("Record", Record) == ["Record"]  # its generated docstring does not count
    assert undocumented("Documented", Documented) == []
    assert undocumented("Constructed", Constructed) == ["Constructed.__init__"]
    assert undocumented("function", function) == ["function"]


def test_the_check_finds_what_is_missing_in_a_module():
    scratch = types.ModuleType("scratch")
    scratch.__doc__ = "A module."

    class Bare: ...

    def bare(): ...

    class Documented:
        """Has one."""

    def documented():
        """Has one."""

    def _private(): ...

    def elsewhere(): ...  # defined in this module, so it is imported as far as scratch is concerned

    for obj in (Bare, bare, Documented, documented, _private):
        obj.__module__ = scratch.__name__
        setattr(scratch, obj.__name__, obj)
    scratch.elsewhere = elsewhere

    assert missing_docstrings(scratch) == ["scratch.Bare", "scratch.bare"]
    assert missing_docstrings(types.ModuleType("bare_module")) == ["bare_module"]
