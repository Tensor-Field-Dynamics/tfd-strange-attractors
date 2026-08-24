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
    ),
    "solar_flare": ColorTheme(
        background=(15, 3, 2),        # Dunkles Kastanienrot/Schwarz
        primary_glow=(255, 215, 0),   # Leuchtendes Goldgelb
        secondary_glow=(255, 45, 0),  # Intensives Feuerrot
        bloom_intensity=1.6
    ),
    "emerald_forest": ColorTheme(
        background=(2, 12, 8),        # Tiefes Smaragd-Dunkelgrün
        primary_glow=(0, 255, 128),   # Neon-Smaragd/Minze
        secondary_glow=(50, 180, 255),# Leuchtendes Türkisblau
        bloom_intensity=1.5
    ),
    "deep_ocean": ColorTheme(
        background=(2, 6, 18),        # Abyssales Ozeanblau
        primary_glow=(0, 191, 255),   # Strahlendes Aquamarin/Sky Blue
        secondary_glow=(138, 43, 226),# Blauviolett
        bloom_intensity=1.7
    ),
    "amethyst_nebula": ColorTheme(
        background=(10, 4, 16),       # Dunkles Nebel-Violett
        primary_glow=(218, 112, 214), # Amethyst-Pink/Orchidee
        secondary_glow=(75, 0, 130),  # Tiefes Indigo
        bloom_intensity=1.6
    ),
    "crimson_synth": ColorTheme(
        background=(12, 2, 4),        # Dunkler Synth-Hintergrund
        primary_glow=(255, 20, 147),  # Neongift-Pink (Deep Pink)
        secondary_glow=(255, 69, 0),  # Orangerot
        bloom_intensity=1.8
    ),
    "golden_hour": ColorTheme(
        background=(12, 8, 4),        # Warmer Bronze-Dunkelton
        primary_glow=(255, 180, 50),  # Warmes Sonnen-Gold
        secondary_glow=(255, 105, 180),# Zartes Rosa/Pink
        bloom_intensity=1.4
    ),
    "arctic_aurora": ColorTheme(
        background=(4, 12, 16),       # Arktisches Polarnacht-Blau
        primary_glow=(127, 255, 212), # Aquamarin-Nordlicht
        secondary_glow=(173, 216, 230),# Eissegel-Hellblau
        bloom_intensity=1.5
    ),
    "volcanic_ash": ColorTheme(
        background=(8, 6, 6),         # Vulkanischer Obsidian-Grauton
        primary_glow=(255, 80, 20),   # Glühende Lava (Magma-Orange)
        secondary_glow=(200, 200, 210),# Asche-Silber
        bloom_intensity=1.6
    ),
    "cyber_ghost": ColorTheme(
        background=(8, 10, 12),       # Schiefer-Dunkelgrau
        primary_glow=(240, 255, 255), # Geisterhaftes Phantomsilber/Weiß
        secondary_glow=(0, 250, 154), # Minzgrünes Leuchten
        bloom_intensity=1.3
    ),
    "cosmic_twilight": ColorTheme(
        background=(6, 4, 14),        # Mitternachts-Kosmos-Violett
        primary_glow=(255, 160, 200), # Kosmisches Neon-Dämmerungspink
        secondary_glow=(100, 149, 237),# Kornblumenblau
        bloom_intensity=1.7
    )
}