# Plotter Workflow Context

This document records how the current repo relates to the longer-term pen
plotter goal.

## What The Plotter Goal Actually Means Here

The repo is **not** currently maintained as a full plotter application.
It is maintained as a stroke-first vector engine that should remain compatible
with a future plotter-oriented workflow.

That future workflow is expected to look like:

1. raster image
2. stroke-first vector optimization
3. canonical SVG output
4. optional cleanup / path ordering
5. optional G-code / pen plotter export

## Current Role Of Plotter Constraints

Plotter constraints matter today as **hard constraints**, not as the primary
optimization objective.

That means the maintained engine already cares about:

- fixed palette support
- fixed physical pen width support
- stroke-only outputs
- SVG artifacts that can be inspected and postprocessed

But it does **not** yet fully optimize for:

- pen-up travel
- stroke ordering
- stylized pen rhythm
- NN-guided artistic priors

## Why Plotter Tooling Is Still In-Repo

The repo is still worth keeping as one end-to-end codebase because the core
runtime is comparatively small and the long-term product is likely to want:

- engine
- painterly/vectorization apps
- SVG inspection and cleanup
- plotter-oriented evaluation

Keeping these layers in one repo is reasonable as long as they stay cleanly
separated:

- `pydiffvg/openstroke/*`: core runtime
- `apps/*`: workflows and benches
- `pydiffvg/plotter/*`: downstream tooling

## What Is Deferred

The following are intentionally not first-priority right now:

- aggressive plotter-aware loss terms
- G-code generation inside the core engine work
- generic SVG geometry cleanup as a replacement for semantic optimization

Those can come later, after engine quality and SVG fidelity are more stable.

## Current Practical Rule

When evaluating the engine for plotter-facing use, check all three:

1. wall-clock time / time-to-quality
2. raster result quality
3. SVG structure quality

SVG structure quality should include at least:

- stroke count
- fragmentation
- travel ratio
- cleanup gain under fixed palette / fixed pen width presets

## Current Honest Baseline

For plotter-facing experiments, the most honest present-day baseline is:

- single fixed black pen
- fixed width
- open-stroke only
- optional TEED / line-art / flowline preconditioning

This exposes geometry quality and fragmentation directly without hiding behind
multi-pen color complexity.

## Relationship To Upstream NN Tools

The likely future role of this repo is to serve as the internal runtime for
upstream sketch models such as:

- `ControlSketch`
- `Clipasso`
- similar NN-guided raster-to-stroke systems

That is why the current priority is still engine quality and canonical SVG
output, not final plotter polish.
