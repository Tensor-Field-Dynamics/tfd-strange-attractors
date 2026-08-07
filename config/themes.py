# config/themes.py
from dataclasses import dataclass
from typing import Tuple

@dataclass
class ColorTheme:
    """
    Datenstruktur zur Definition der Farbkonfiguration für das Post-Processing 
    und die Tiefenfaltung (Z-Axis Mapping). Farbwerte werden als RGB-Tupel definiert.
    """
    background: Tuple[int, int, int]
    primary_glow: Tuple[int, int, int]
    secondary_glow: Tuple[int, int, int]
    bloom_intensity: float

# Bereitstellung der standardisierten TFD-Farbprofile.
THEMES = {
    "neon_cyberpunk": ColorTheme(
        background=(5, 5, 10),        # Tiefes Dunkelblau/Schwarz
        primary_glow=(0, 255, 255),   # Cyan
        secondary_glow=(255, 0, 255), # Magenta
        bloom_intensity=1.8
    ),
    "dark_matter": ColorTheme(
        background=(0, 0, 0),
        primary_glow=(255, 100, 50),  # Orange/Kupfer
        secondary_glow=(50, 100, 255),# Tiefblau
        bloom_intensity=1.2
    )
}