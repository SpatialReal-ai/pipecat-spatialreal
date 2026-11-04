"""The README keeps in step with the contract it describes."""

import re
from pathlib import Path

from pipecat_spatialreal.proto.generated import avatar_pb2

README = Path(__file__).resolve().parent.parent / "README.md"


def test_the_readme_lists_exactly_the_messages_of_the_contract():
    # "The messages between the processor and the browser (`a`, `b` and `c`) are defined in ..."
    text = " ".join(README.read_text().split())
    match = re.search(
        r"The messages between the processor and the browser \(([^)]*)\) are defined", text
    )
    assert match, "the README sentence listing the messages is gone or was reworded"
    listed = re.findall(r"`([a-z_]+)`", match.group(1))
    arms = [f.name for f in avatar_pb2.AvatarMessage.DESCRIPTOR.oneofs_by_name["payload"].fields]

    assert len(listed) == len(set(listed)), f"a message is listed twice: {listed}"
    assert set(listed) == set(arms), f"README lists {sorted(listed)}, the proto has {sorted(arms)}"
