# Notes for AI agents

- **The code** is described in `README.md` (pipeline in `src/sgi/`, simulator in `sim/`, models in
  `configs/default.yaml`). Tests: `pytest`.
- **The results** start at `results/README.md`: the question, the main findings with their files, the layout, how to
  unpack the run logs (`results/logs/`, Git LFS), and a glossary of the run names.
- Every number in the results traces to a scoring script and its output under `results/`; prefer re-computing from
  the logs to quoting a note.
- Model calls need API keys in a local `.env` (see `.env.example`); none are in this repository, and nothing in
  `results/` needs one.
