import sys
import os
import time
from PIL import Image

# Der Modulsuchpfad wird für die lokale Modulauflösung erweitert.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attractor_engine import CliffordAttractor
from core.shading import render_image
from config.resolutions import TARGET_CONFIGS, TargetPlatform
from config.themes import THEMES

def main():
    """
    Dieses Skript orchestriert die Generierung einer temporalen Bildsequenz (Animation).
    Durch die lineare Interpolation der Attraktor-Parameter wird eine fließende
    morphologische Transformation der Geometrie erzeugt.
    """
    # 1. Konfiguration für ein Reel/Story (9:16 Format) laden
    target = TargetPlatform.INSTA_STORY
    config = TARGET_CONFIGS[target]
    theme = THEMES["neon_cyberpunk"]
    
    # 2. Animations-Parameter definieren
    num_frames = 150  # Entspricht 5 Sekunden bei 30 FPS
    
    # Start- und Endparameter für das Morphing des Clifford-Attraktors
    a_start, a_end = -1.4, -1.35
    b_start, b_end = 1.6, 1.7
    c_start, c_end = 1.0, 1.1
    d_start, d_end = 0.7, 0.75

    output_dir = os.path.join("outputs", "animations")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starte Morphing-Sequenz für {target.value} ({num_frames} Frames)...")
    total_start_time = time.time()

    # 3. Iterative Berechnung der Einzelbilder (Frames)
    for frame in range(num_frames):
        frame_start_time = time.time()
        
        # Lineare Interpolation (Lerp) der Parameter basierend auf dem aktuellen Frame
        progress = frame / float(num_frames - 1)
        current_a = a_start + (a_end - a_start) * progress
        current_b = b_start + (b_end - b_start) * progress
        current_c = c_start + (c_end - c_start) * progress
        current_d = d_start + (d_end - d_start) * progress

        # Der Attraktor wird für den spezifischen Zeitschritt neu instanziiert.
        attractor = CliffordAttractor(
            a=current_a, b=current_b, c=current_c, d=current_d
        )
        
        # Berechnung des Dichtefeldes
        density = attractor.generate_density_map(
            width=config.internal_width, 
            height=config.internal_height, 
            num_points=3_000_000,  # Leicht reduziert für flüssigere Render-Zeiten
            iters_per_point=50
        )
        
        # Shading und Supersampling
        img = render_image(density, theme)
        if config.ssaa_factor > 1:
            img = img.resize((config.width, config.height), resample=Image.Resampling.LANCZOS)

        # Das Bild wird mit einer fortlaufenden Nummerierung (Padding) gespeichert.
        filename = f"frame_{frame:04d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        frame_time = time.time() - frame_start_time
        print(f"Rendered Frame {frame+1}/{num_frames} in {frame_time:.2f}s -> {filename}")

    total_time = time.time() - total_start_time
    print(f"Sequenz erfolgreich beendet! Gesamtdauer: {total_time / 60:.2f} Minuten.")

if __name__ == "__main__":
    main()