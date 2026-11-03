"""The example web page, ``examples/client/index.html``, driven the way a browser drives it.

``client_page_check.mjs`` runs the page's module script under Node with every import replaced by a
recording fake (pipecat-client-js, its transports and the web SDK), and with fake ``fetch``,
timers and elements. It presses Start and Stop and fires the bot's callbacks, and checks what the
page asks of the bot's server (``GET /spatialreal-token``, then ``POST /start`` with the body
Pipecat's runner needs), what it hands the web SDK (the session's options, the deployment from the
token route, only the bot's audio track, every server message), and that Stop works and a bot that
never gets ready is given up on. Without ``node`` the test is skipped locally and fails in CI,
where the workflow installs Node.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "examples" / "client" / "index.html"
CHECK = Path(__file__).with_name("client_page_check.mjs")


@pytest.mark.skipif(shutil.which("node") is None and not os.environ.get("CI"), reason="needs node")
def test_the_page_wires_the_bot_and_the_avatar_session(tmp_path):
    assert shutil.which("node"), "Node is not installed; the workflow must install it"
    done = subprocess.run(
        ["node", str(CHECK), str(PAGE), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "FAIL" not in done.stdout, done.stdout
