# config/resolutions.py
from dataclasses import dataclass
from enum import Enum

class TargetPlatform(Enum):
    """
    In dieser Enumeration werden die definierten Zielplattformen 
    für den iterativen Renderprozess festgelegt.
    """
    # Social Media
    INSTA_PORTRAIT = "insta_portrait"
    INSTA_STORY = "insta_story"
    
    # Desktop & Mobile Wallpapers
    WALLPAPER_FHD = "wallpaper_fhd"
    WALLPAPER_QHD = "wallpaper_qhd"
    WALLPAPER_4K = "wallpaper_4k"
    WALLPAPER_8K = "wallpaper_8k"
    WALLPAPER_ULTRAWIDE = "wallpaper_ultrawide"
    WALLPAPER_MOBILE = "wallpaper_mobile"
    
    # DIN-Druckformate (300 DPI)
    POSTER_A1 = "poster_a1"
    POSTER_A2 = "poster_a2"
    POSTER_A3 = "poster_a3"
    POSTER_A4 = "poster_a4"
    POSTER_A5 = "poster_a5"
    POSTER_A6 = "poster_a6"

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
# Für sehr hochauflösende Druckformate (A1–A3) und 8K-Auflösungen wird der SSAA-Faktor 
# auf 1 reduziert, um Out-of-Memory-Fehler im VRAM zu vermeiden.
TARGET_CONFIGS = {
    # Social Media
    TargetPlatform.INSTA_PORTRAIT: RenderConfig(width=1080, height=1350, ssaa_factor=2),
    TargetPlatform.INSTA_STORY: RenderConfig(width=1080, height=1920, ssaa_factor=2),
    
    # Wallpapers
    TargetPlatform.WALLPAPER_FHD: RenderConfig(width=1920, height=1080, ssaa_factor=2),
    TargetPlatform.WALLPAPER_QHD: RenderConfig(width=2560, height=1440, ssaa_factor=2),
    TargetPlatform.WALLPAPER_4K: RenderConfig(width=3840, height=2160, ssaa_factor=2),
    TargetPlatform.WALLPAPER_8K: RenderConfig(width=7680, height=4320, ssaa_factor=1),
    TargetPlatform.WALLPAPER_ULTRAWIDE: RenderConfig(width=3440, height=1440, ssaa_factor=2),
    TargetPlatform.WALLPAPER_MOBILE: RenderConfig(width=1440, height=3200, ssaa_factor=2),
    
    # DIN-Druckformate (300 DPI, Hochformat)
    TargetPlatform.POSTER_A1: RenderConfig(width=7016, height=9933, ssaa_factor=1),
    TargetPlatform.POSTER_A2: RenderConfig(width=4960, height=7016, ssaa_factor=1),
    TargetPlatform.POSTER_A3: RenderConfig(width=3508, height=4960, ssaa_factor=1),
    TargetPlatform.POSTER_A4: RenderConfig(width=2480, height=3508, ssaa_factor=2),
    TargetPlatform.POSTER_A5: RenderConfig(width=1748, height=2480, ssaa_factor=2),
    TargetPlatform.POSTER_A6: RenderConfig(width=1240, height=1748, ssaa_factor=2)
}