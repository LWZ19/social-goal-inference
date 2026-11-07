# Shared helpers

- `paths.py`: every location the scripts need (results, the still-scene run folders, the simulator run folders, the
  moving-line design folder). Scripts import it instead of counting parent folders, so folders can move without
  breaking them; how to import it is at the top of the file.
- `mcompare.py`: loads simulator episodes and pairs two versions by scene and seed (wins, losses, ties, sign test).
