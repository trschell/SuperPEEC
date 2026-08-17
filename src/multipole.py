# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""multipole.py - backward-compatible facade.

The original ~4000-line module was split into multipole_common, special,
greens, levels, leaf_induct, leaf_poten, and tree.  `import multipole as mp`
still exposes everything (mp.Tree, etc.).
"""
from multipole_common import *  # noqa: F401,F403
from special import *  # noqa: F401,F403
from greens import *  # noqa: F401,F403
from levels import *  # noqa: F401,F403
from leaf_induct import *  # noqa: F401,F403
from leaf_poten import *  # noqa: F401,F403
from tree import *  # noqa: F401,F403
