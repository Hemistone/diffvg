# pydiffvg.optimize package reference

The `pydiffvg.optimize` package provides modular building blocks that were
previously embedded inside the monolithic `pydiffvg.optimize_svg` module.  The
package can be consumed à la carte or via the high-level driver introduced in
Stage 5.

## Overview

Import the entire package through `import pydiffvg.optimize as opt` or via the
top-level aliases re-exported by `pydiffvg.__init__`.

| Module | Purpose | Key symbols |
| ------ | ------- | ----------- |
| `opt.settings` | Load/store optimization parameters, JSON helpers | `SvgOptimizationSettings` |
| `opt.transforms` | Matrix decomposition/recomposition utilities shared by parser and gradients | `TransformTools` |
| `opt.scene_graph` | Scene node definitions backed by torch tensors | `SvgNode`, `PathNode`, `GroupNode`, … |
| `opt.parser` | `SvgParserMixin` attaching parsing helpers to `OptimizableSvg` | `SvgParserMixin` |
| `opt.writer` | XML serialization helpers | `SvgWriterMixin` |
| `opt.core` | Implementation of `OptimizableSvg` using the mixins/modules above | `OptimizableSvg` |
| `opt.driver` | High-level orchestration loop | `SvgOptimizationDriver` |

All modules are available from `pydiffvg.optimize` and from the legacy
`pydiffvg.optimize_svg` shim (which now emits a deprecation warning).

## Quick start

```python
import torch
import pydiffvg
from pydiffvg.optimize import (
    SvgOptimizationSettings,
    SvgOptimizationDriver,
)

pydiffvg.set_use_gpu(torch.cuda.is_available())

settings = SvgOptimizationSettings()
driver = SvgOptimizationDriver(
    "apps/imgs/note_small.svg",
    settings=settings,
    optimize_background=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

target = torch.ones((driver.document.canvas[1], driver.document.canvas[0], 4), dtype=torch.float32)

def mse_loss(image, iteration, drv):
    return torch.nn.functional.mse_loss(image, target)

driver.optimize(mse_loss, iterations=10)
driver.save_svg("results/note_small_driver.svg")
```

### Accessing lower-level pieces

The driver simply delegates to `OptimizableSvg`.  If you need finer control:

```python
from pydiffvg.optimize import OptimizableSvg, SvgParserMixin, SvgWriterMixin

settings = SvgOptimizationSettings()
doc = OptimizableSvg("scene.svg", settings=settings, verbose=True)
img = doc.render(seed=42)
doc.zero_grad()
# ... compute custom loss/gradients ...
doc.step()
```

`SvgParserMixin` and `SvgWriterMixin` are exposed so downstream applications can
embed the parsing / serialization logic into their own subclasses if desired.

### Legacy imports

Existing code that still does `from pydiffvg.optimize_svg import OptimizableSvg`
continues to work.  The shim forwards to the package described above and will
print a `DeprecationWarning`.  Prefer migrating to `pydiffvg.optimize` imports
going forward.
