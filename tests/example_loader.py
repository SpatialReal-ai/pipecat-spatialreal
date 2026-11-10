"""Load ``examples/bot.py`` for the tests, once, the way ``python examples/bot.py`` loads it.

Importing the example registers its token route and its pages on the Pipecat runner's
module-level app, so a second import would register them twice. Every test module that needs the
example takes it from :func:`load_bot`, which imports it on first use and hands out the same module
afterwards. The callers skip themselves first when the example extra is not installed.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType

BOT_PATH = Path(__file__).resolve().parent.parent / "examples" / "bot.py"
_MODULE_NAME = "example_bot"

# The example imports Pipecat's Silero VAD, and with it onnxruntime, whose telemetry uploads its
# events from a thread and waits for the upload when the process exits: where the network does not
# let the upload through, the test process never ends. Tests send no telemetry. onnxruntime is
# first imported with the example, so this is set before that, and kept; the subprocesses that run
# the example get it too.
TELEMETRY_OFF = {"ORT_DISABLE_TELEMETRY": "1"}


def load_bot() -> ModuleType:
    """Return the example module, importing it on first use.

    Importing the Pipecat runner reads a ``.env`` into ``os.environ`` and sets ``ENV``; the
    environment is put back afterwards so that no other test sees a developer's keys, except for
    :data:`TELEMETRY_OFF`. The deprecation warnings the example's third-party imports raise (the
    legacy websockets server in Pipecat's eval transport, a deprecated TTS base class) are not ours
    to fix and are dropped here, so the package's own warning filter stays as strict as it was.

    Returns:
        The module ``examples/bot.py`` defines.
    """
    loaded = sys.modules.get(_MODULE_NAME)
    if loaded is not None:
        return loaded
    os.environ.update(TELEMETRY_OFF)
    saved = dict(os.environ)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, BOT_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    sys.modules[_MODULE_NAME] = module
    return module
