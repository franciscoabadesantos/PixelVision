# Contributing

## Scope

PixelVision is meant to be a compact inference and experimentation toolkit:

- keep detection wrappers small and easy to inspect
- keep segmentation utilities reusable and dataset-agnostic
- avoid coupling the repo to heavyweight training pipelines

## Development Notes

- avoid committing model weights unless there is a strong reason
- document any new assumptions about image format, color ranges, or model inputs
- keep examples runnable from the repository root

## Pull Requests

Before opening a pull request:

1. Update documentation for behavior changes.
2. Run a syntax check for edited Python files.
3. Keep new dependencies justified and minimal.
