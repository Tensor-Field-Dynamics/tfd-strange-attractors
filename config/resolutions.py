# config/resolutions.py
from dataclasses import dataclass
from enum import Enum

class TargetPlatform(Enum):
    """
    In dieser Enumeration werden die definierten Zielplattformen 
    für den iterativen Renderprozess festgelegt.
    """
    INSTA_PORTRAIT = "insta_portrait"
    INSTA_STORY = "insta_story"
    WALLPAPER_4K = "wallpaper_4k"
    POSTER_A2 = "poster_a2"

@dataclass
class RenderConfig:
    """
    Konfigurationsobjekt für die spatiale Render-Auflösung.
    Um Aliasing-Effekte bei fraktalen Strukturen zu minimieren, 
    wird standardmäßig ein Supersampling-Faktor (SSAA) angewendet.
    """
    width: int
    height: int
    ssaa_factor: int = 2

    @property
    def internal_width(self) -> int:
        """
        Die interne Berechnungsbreite wird aus der Zielauflösung 
        und dem SSAA-Multiplikator abgeleitet.
        """
        return self.width * self.ssaa_factor

    @property
    def internal_height(self) -> int:
        """
        Die interne Berechnungshöhe wird aus der Zielauflösung 
        und dem SSAA-Multiplikator abgeleitet.
        """
        return self.height * self.ssaa_factor

# Es werden vordefinierte Konfigurationen für die Pipeline bereitgestellt.
# Für hochauflösende Druckformate (A2) wird der SSAA-Faktor auf 1 reduziert, 
# um Out-of-Memory-Fehler im VRAM zu vermeiden.
TARGET_CONFIGS = {
    TargetPlatform.INSTA_PORTRAIT: RenderConfig(width=1080, height=1350, ssaa_factor=2),
    TargetPlatform.INSTA_STORY: RenderConfig(width=1080, height=1920, ssaa_factor=2),
    TargetPlatform.WALLPAPER_4K: RenderConfig(width=3840, height=2160, ssaa_factor=2),
    TargetPlatform.POSTER_A2: RenderConfig(width=4960, height=7016, ssaa_factor=1) 
}