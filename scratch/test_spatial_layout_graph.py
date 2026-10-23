"""
Test suite for 2D Visual Spatial Layout & Connectivity Graph.
Tests screen zone classification across [MODAL/POPUP], [HEADER/NAV], [SIDEBAR],
[WORKSPACE], and [FOOTER], as well as layout graph construction and spatial connectivity hints.
"""

import sys
import os
import pytest

# Ensure utf-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("."))

from spatial_droplet.reasoning.reasoner import DiagonalDropletReasoner
from spatial_droplet.core.models import DiagonalDroplet


def test_classify_screen_zone_all_zones():
    """Verify classify_screen_zone accurately places coordinates into the 5 functional zones."""
    sw = 1920
    sh = 1080
    modal_rect = (500, 300, 900, 700)

    # 1. Modal / Popup takes precedence
    assert DiagonalDropletReasoner.classify_screen_zone(600, 400, sw, sh, modal_rect) == "[MODAL/POPUP]"
    assert DiagonalDropletReasoner.classify_screen_zone(505, 305, sw, sh, modal_rect) == "[MODAL/POPUP]"

    # 2. Header / Nav Bar (y < 0.12 * 1080 = 129.6)
    assert DiagonalDropletReasoner.classify_screen_zone(100, 50, sw, sh) == "[HEADER/NAV]"
    assert DiagonalDropletReasoner.classify_screen_zone(1000, 120, sw, sh) == "[HEADER/NAV]"

    # 3. Footer / Bottom Bar (y >= 0.88 * 1080 = 950.4)
    assert DiagonalDropletReasoner.classify_screen_zone(200, 1000, sw, sh) == "[FOOTER]"
    assert DiagonalDropletReasoner.classify_screen_zone(1500, 960, sw, sh) == "[FOOTER]"

    # 4. Sidebar / Nav Drawer (x < 0.22 * 1920 = 422.4, outside header & footer)
    assert DiagonalDropletReasoner.classify_screen_zone(150, 400, sw, sh) == "[SIDEBAR]"
    assert DiagonalDropletReasoner.classify_screen_zone(350, 600, sw, sh) == "[SIDEBAR]"

    # 5. Workspace / Central Viewport (interior)
    assert DiagonalDropletReasoner.classify_screen_zone(960, 540, sw, sh) == "[WORKSPACE]"
    assert DiagonalDropletReasoner.classify_screen_zone(500, 200, sw, sh) == "[WORKSPACE]"


def test_build_spatial_layout_graph_empty():
    """Verify _build_spatial_layout_graph handles empty node lists gracefully."""
    graph_str = DiagonalDropletReasoner._build_spatial_layout_graph([], 1920, 1080)
    assert "No droplets present on screen" in graph_str
    assert "2D VISUAL SPATIAL LAYOUT & CONNECTIVITY GRAPH" in graph_str


def test_build_spatial_layout_graph_multi_zone_clustering():
    """Verify droplets are correctly clustered into zone buckets with coordinates, archetypes, and hints."""
    sw = 1920
    sh = 1080
    modal_rect = (700, 400, 1100, 600)

    d_header1 = DiagonalDroplet(
        id=1,
        center_x=45,
        center_y=45,
        category="button",
        label="Back Button",
        diag_wave=90,
        archetype="button",
        container_title="Edge"
    )
    d_header2 = DiagonalDroplet(
        id=2,
        center_x=85,
        center_y=45,
        category="button",
        label="Forward Button",
        diag_wave=130,
        archetype="button",
        container_title="Edge"
    )
    d_sidebar = DiagonalDroplet(
        id=10,
        center_x=120,
        center_y=350,
        category="interactive_widget",
        label="Favorites",
        diag_wave=470,
        archetype="link"
    )
    d_workspace = DiagonalDroplet(
        id=25,
        center_x=1350,
        center_y=500,
        category="interactive_widget",
        label="Video Player",
        diag_wave=1850,
        archetype="canvas"
    )
    d_modal = DiagonalDroplet(
        id=30,
        center_x=850,
        center_y=480,
        category="interactive_widget",
        label="Confirm Save",
        diag_wave=1330,
        archetype="button"
    )
    d_footer = DiagonalDroplet(
        id=40,
        center_x=500,
        center_y=1040,
        category="interactive_app",
        label="Windows Start",
        diag_wave=1540,
        archetype="button"
    )

    nodes = [d_header1, d_header2, d_sidebar, d_workspace, d_modal, d_footer]
    graph_str = DiagonalDropletReasoner._build_spatial_layout_graph(nodes, sw, sh, modal_rect)

    # Verify section headers
    assert "[HEADER / NAV BAR]" in graph_str
    assert "[SIDEBAR / NAV DRAWER]" in graph_str
    assert "[WORKSPACE / VIEWPORT]" in graph_str
    assert "[MODAL / POPUP DIALOG]" in graph_str
    assert "[FOOTER / BOTTOM BAR]" in graph_str

    # Verify node clustering
    assert "• [1]" in graph_str and "Back Button" in graph_str
    assert "• [2]" in graph_str and "Forward Button" in graph_str
    assert "• [10]" in graph_str and "Favorites" in graph_str
    assert "• [25]" in graph_str and "Video Player" in graph_str
    assert "• [30]" in graph_str and "Confirm Save" in graph_str
    assert "• [40]" in graph_str and "Windows Start" in graph_str

    # Verify horizontal neighborhood connectivity hint between d_header1 and d_header2
    # d_header2 (x=85) is Right of d_header1 (x=45)
    assert "Right of Dot [1]" in graph_str or "Left of Dot [2]" in graph_str
