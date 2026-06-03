import sys

from . import legacy_nn

# Preserve the historical nets.nn module path for pickled checkpoints while
# making legacy_nn the single implementation in the package.
nn = legacy_nn
sys.modules[__name__ + '.nn'] = legacy_nn
