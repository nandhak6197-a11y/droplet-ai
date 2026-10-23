import os
import sys
import tkinter as tk
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_droplet.ui.chat_widget import HoverChatWidget

def main():
    root = tk.Tk()
    root.withdraw()

    submitted = []
    def on_submit(cmd):
        print(f"Submitted: {cmd}")
        submitted.append(cmd)

    def on_toggle():
        print("Toggle dots clicked")

    def on_refresh():
        print("Refresh clicked")

    droplet_vis = [True]
    def on_toggle_droplets():
        droplet_vis[0] = not droplet_vis[0]
        print(f"Toggle droplets callback invoked: visible={droplet_vis[0]}")
        return droplet_vis[0]

    widget = HoverChatWidget(
        root,
        on_submit_callback=on_submit,
        on_toggle_callback=on_toggle,
        on_refresh_callback=on_refresh,
        on_toggle_droplets_callback=on_toggle_droplets
    )

    print("Initial rect:", widget.get_rect())
    assert widget.droplets_visible is True, "Default droplets_visible must be True"
    widget.set_status("Test Status: Active", color="#00e5ff")
    
    # Test toggle droplets (Hide)
    widget.toggle_droplets()
    assert widget.droplets_visible is False, "Droplets must be hidden after toggle"
    print("Droplets visibility after hide:", widget.droplets_visible)

    # Test toggle droplets (Show / See)
    widget.toggle_droplets()
    assert widget.droplets_visible is True, "Droplets must be visible after second toggle"
    print("Droplets visibility after see:", widget.droplets_visible)

    # Test collapse to sphere orb
    widget.collapse_to_orb()
    print("Collapsed rect:", widget.get_rect())

    # Test toggle droplets in orb mode
    widget.toggle_droplets()
    assert widget.droplets_visible is False, "Droplets must toggle in orb mode"
    print("Orb mode droplets visibility:", widget.droplets_visible)

    # Test expand to capsule
    widget.expand_to_capsule()
    print("Expanded rect:", widget.get_rect())

    widget.win.destroy()
    root.destroy()
    print("HoverChatWidget test with Hide/See Droplets passed cleanly!")

if __name__ == '__main__':
    main()
