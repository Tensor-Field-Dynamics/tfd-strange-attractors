import sys
import os
import time
from PIL import Image

# Der Modulsuchpfad wird erweitert, um relative Importe aus dem Hauptverzeichnis zu ermöglichen.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attractor_engine import CliffordAttractor, ThomasAttractor
from core.shading import render_image
from config.resolutions import TARGET_CONFIGS, TargetPlatform
from config.themes import THEMES

def main():
    """
    Der primäre Einstiegspunkt für die Generierung der Data Art.
    Hier wird die Pipeline orchestriert: Konfiguration, Berechnung und Export.
    """
    # 1. Konfiguration laden
    target = TargetPlatform.INSTA_PORTRAIT
    config = TARGET_CONFIGS[target]
    theme = THEMES["neon_cyberpunk"]
    
    print(f"Starte Rendering für {target.value}...")
    print(f"Interne Renderauflösung: {config.internal_width}x{config.internal_height} (SSAA: {config.ssaa_factor}x)")

    # 2. Attraktor initialisieren (Hier kann zwischen Clifford und Thomas gewechselt werden)
    # Ein Clifford-Attraktor mit beispielhaften, visuell ansprechenden Parametern wird instanziiert.
    attractor = CliffordAttractor(a=-1.4, b=1.6, c=1.0, d=0.7)
    
    start_time = time.time()
    
    # 3. Dichtefeld auf der GPU berechnen
    # Für hochauflösende SSAA-Bilder wird eine signifikante Partikelanzahl benötigt.
    density = attractor.generate_density_map(
        width=config.internal_width, 
        height=config.internal_height, 
        num_points=5_000_000, 
        iters_per_point=50
    )
    
    calc_time = time.time() - start_time
    print(f"Berechnung abgeschlossen in {calc_time:.2f} Sekunden.")

    # 4. Post-Processing und Shading anwenden
    img = render_image(density, theme)
    
    # Wenn SSAA (Supersampling) angewandt wurde, wird das Bild für den Export herunterskaliert.
    if config.ssaa_factor > 1:
        img = img.resize((config.width, config.height), resample=Image.Resampling.LANCZOS)

    # 5. Export der finalen Bilddatei
    output_path = os.path.join("outputs", "feed", "test_render.png")
    img.save(output_path)
    print(f"Erfolgreich exportiert nach: {output_path}")

if __name__ == "__main__":
    main()