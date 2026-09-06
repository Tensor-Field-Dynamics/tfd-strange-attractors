import os
from enum import Enum

class BrandingAsset(Enum):
    """
    Verfügbare Branding-Wasserzeichen-Assets im Ordner assets/branding.
    """
    LOGO_1 = "assets/branding/tfd_logo_1.png"
    LOGO_2 = "assets/branding/tfd_logo_2.png"
    HORIZONTAL_1 = "assets/branding/tfd_logo_horizontal_1.png"
    HORIZONTAL_2 = "assets/branding/tfd_logo_horizontal_2.png"
    WHITE_LOGO = "assets/branding/tfd_logo_white.png"

# Standard-Branding-Dateipfade für schnellen Zugriff
BRANDING_ASSETS = {
    "logo_1": BrandingAsset.LOGO_1.value,
    "logo_2": BrandingAsset.LOGO_2.value,
    "horizontal_1": BrandingAsset.HORIZONTAL_1.value,
    "horizontal_2": BrandingAsset.HORIZONTAL_2.value,
    "white_logo": BrandingAsset.WHITE_LOGO.value,
    # Aliasse für intuitive Verwendung
    "tfd_logo_1": BrandingAsset.LOGO_1.value,
    "tfd_logo_2": BrandingAsset.LOGO_2.value,
    "tfd_logo_horizontal_1": BrandingAsset.HORIZONTAL_1.value,
    "tfd_logo_horizontal_2": BrandingAsset.HORIZONTAL_2.value,
    "tfd_logo_white": BrandingAsset.WHITE_LOGO.value,
    "default": BrandingAsset.WHITE_LOGO.value
}

