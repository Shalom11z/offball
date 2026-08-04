# offball (Python package)

Automated off-the-ball positioning analysis from soccer match footage.

This is the Python component of the [offball](https://github.com/Shalom11z/offball)
platform. Full documentation, architecture notes and the metric definitions live
in the repository root, under [`docs/`](https://github.com/Shalom11z/offball/tree/main/docs).

## Install

```bash
pip install offball            # tactics layer only, no heavy dependencies
pip install 'offball[vision]'  # + OpenCV, ultralytics, numpy
pip install 'offball[api]'     # + FastAPI, uvicorn
```

## Use

```python
from offball.tactics import score_frame
from offball.tactics.report import build_report

scores = [s for s in (score_frame(f) for f in frames) if s is not None]
report = build_report(scores, frames_total=len(frames), fps=25.0)
print(report.to_dict())
```

The tactics layer runs on the standard library alone and works on tracking data
from any provider — you do not need this project's vision stage to use it.

Try it without footage:

```bash
offball demo
offball info
```

## Compiled kernels

The pitch-control grid has a Rust implementation in `rust/offball-core`. It is
optional; without it the pure-Python reference runs instead, with identical
results and much lower throughput. Check with `offball info`, build with:

```bash
cd rust/offball-core && maturin develop --release --features python
```
