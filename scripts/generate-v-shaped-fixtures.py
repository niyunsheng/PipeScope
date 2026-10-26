"""Regenerate upstream order fixtures (optional; not part of npm test).

python3 scripts/generate-v-shaped-fixtures.py /path/to/pinned/v1f1b.py
Source: sail-sg/zero-bubble-pipeline-parallelism commit
39f2c186a27580b7051eb51e0f651e1efd0a8170. Only run with that trusted source.
The import shim avoids requiring Megatron / Torch to run the reference order.
"""
import json
import pathlib
import sys

class Node:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

source = pathlib.Path(sys.argv[1]).read_text()
source = "\n".join(line for line in source.splitlines() if not line.startswith("from megatron."))
namespace = {"GraphConfig": object, "ScheduledNode": Node, "F": "F", "BW": "B"}
exec(compile(source, sys.argv[1], "exec"), namespace)
cases = []
for pp, m in [(1, 1), (2, 3), (3, 8), (4, 8), (5, 8), (6, 13), (8, 2)]:
    ranks = namespace["schedule"](pp, m, [1, 1, 1, 0])
    cases.append({"pp": pp, "microBatches": m, "order": [
        [f"{op.type}{op.chunk}:{op.microbatch}" for op in rank] for rank in ranks
    ]})
output = pathlib.Path(__file__).resolve().parents[1] / "tests/fixtures/v-shaped-reference.json"
output.write_text(json.dumps({
    "source": "sail-sg/zero-bubble-pipeline-parallelism@39f2c186a27580b7051eb51e0f651e1efd0a8170:megatron/core/pipeline_parallel/zerobubble/scheduler/v1f1b.py",
    "cases": cases,
}, indent=2) + "\n")
