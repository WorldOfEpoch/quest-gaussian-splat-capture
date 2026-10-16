"""Quest Splat Bridge Blender add-on.

Install this directory as a Blender add-on. Network and catalog logic lives in
client.py so it can be exercised without importing Blender.
"""

bl_info = {
    "name": "Quest Splat Bridge",
    "author": "Quest Splat Bridge contributors",
    "version": (0, 1, 0),
    "blender": (5, 3, 0),
    "location": "3D View > Sidebar > Quest Scan",
    "description": "Manage Quest scan projects and capture sessions on the PC bridge",
    "category": "3D View",
}


def register():
    from .blender_ui import register as register_ui

    register_ui()


def unregister():
    from .blender_ui import unregister as unregister_ui

    unregister_ui()
