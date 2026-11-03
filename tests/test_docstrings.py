"""Every public module, class, function and method of the package has a docstring.

The package is a library: its docstrings are the reference a user reads in an editor, and the
Google-style sections in them (``Args:``, ``Returns:``, ``Raises:``) are where the contract of each
call is written down. A name that is public but undocumented is a gap in that contract.

What counts as public: a name that does not start with an underscore (``__init__`` aside), in a
module of the package that does not, defined in that module rather than imported into it. The
generated protobuf module is excluded: it is not written here.
"""

import dataclasses
import importlib
import inspect
import pkgutil
import types
from collections.abc import Iterator

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


def public_modules() -> list[types.ModuleType]:
    """The package and its hand-written modules, imported."""
    modules = [pipecat_spatialreal]
    for info in pkgutil.iter_modules(pipecat_spatialreal.__path__):
        if not info.name.startswith("_") and info.name not in GENERATED:
            modules.append(importlib.import_module(f"{pipecat_spatialreal.__name__}.{info.name}"))
    return modules


@pytest.mark.parametrize("module", public_modules(), ids=lambda module: module.__name__)
def test_every_public_name_has_a_docstring(module):
    assert missing_docstrings(module) == []


def test_the_check_covers_the_package_and_leaves_out_the_generated_code():
    names = {module.__name__ for module in public_modules()}
    assert names >= {
        "pipecat_spatialreal",
        "pipecat_spatialreal.driven",
        "pipecat_spatialreal.envelope",
        "pipecat_spatialreal.onset",
        "pipecat_spatialreal.processor",
        "pipecat_spatialreal.version",
    }
    assert not [name for name in names if ".proto" in name]
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
    assert missing_docstrings(types.ModuleType("scratch")) == ["scratch"]
