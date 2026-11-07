"""Scripts that run inside Blender's own Python, started by sgi.synth.realistic.blender.run_script.

Nothing here is imported by sgi, and nothing here imports sgi, pydantic or Pillow: Blender's Python has bpy
and numpy and none of those. Blender does not put a script's folder on sys.path, so a script that imports
one beside it adds its own folder first.
"""
