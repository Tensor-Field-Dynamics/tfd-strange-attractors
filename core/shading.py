import numpy as np
from PIL import Image
from config.themes import ColorTheme

def apply_aces_tonemapping(x: np.ndarray) -> np.ndarray:
    """
    Für die Farbkalibrierung wird eine Annäherung der ACES (Academy Color Encoding System)
    Tone-Mapping-Kurve angewandt. Dadurch wird ein Übersteuern der Leuchtkraft (Clipping)
    verhindert und ein filmischer Roll-off in den hellen Bereichen (Highlights) erzielt.
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    # Die ACES-Funktion wird vektorisiert auf das Array angewendet.
    x_mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(x_mapped, 0.0, 1.0)

def render_image(density_map: np.ndarray, theme: ColorTheme) -> Image.Image:
    """
    Das rohe, akkumulierte Dichtefeld wird in ein visualisierbares Farbbild transformiert.
    Die Skalierung der Dichtewerte erfolgt logarithmisch, um einen pseudo-volumetrischen
    Glow-Effekt (Bloom) zu simulieren.
    """
    # Eine logarithmische Skalierung wird durchgeführt, um den Dynamikumfang zu komprimieren.
    log_density = np.log1p(density_map * theme.bloom_intensity)
    max_density = np.max(log_density)
    
    if max_density > 0:
        log_density /= max_density

    # Für die Farbzuweisung (Color Mapping) wird ein 3D-Array vorbereitet.
    height, width = log_density.shape
    image_array = np.zeros((height, width, 3), dtype=np.float32)
    
    # RGB-Farbwerte werden aus dem gewählten Theme extrahiert und normalisiert.
    bg_color = np.array(theme.background) / 255.0
    color_1 = np.array(theme.primary_glow) / 255.0
    color_2 = np.array(theme.secondary_glow) / 255.0

    # Die Farbmischung wird anhand der normalisierten Dichte interpoliert.
    # Dichte Bereiche erhalten die primäre Farbe, mittlere Bereiche die sekundäre.
    image_array = (
        log_density[..., np.newaxis] * color_1 + 
        (1.0 - log_density[..., np.newaxis]) * log_density[..., np.newaxis] * color_2
    )
    
    # Die Hintergrundfarbe wird addiert.
    image_array += bg_color * (1.0 - log_density[..., np.newaxis])
    
    # Das finale Bild wird durch das Tone-Mapping geleitet und für den Export formatiert.
    image_array = apply_aces_tonemapping(image_array)
    image_array = (image_array * 255).astype(np.uint8)
    
    return Image.fromarray(image_array, mode="RGB")