"""
Dieses Modul generiert eine Sequenz von Einzelbildern zur Darstellung einer 
Build-Up-Animation eines Attraktors. Über die Gesamtzahl der Frames wird die 
Partikelanzahl linear skaliert, um einen schrittweisen Aufbau des Attraktors zu simulieren.
"""

import os
import sys

# Der Modulsuchpfad wird erweitert, um relative Importe aus dem Hauptverzeichnis zu ermöglichen.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from core.attractor_engine import AizawaAttractor
from core.shading import render_image
from config.resolutions import TARGET_CONFIGS, TargetPlatform
from config.themes import THEMES


def generate_buildup_animation() -> None:
    """
    Erstellt 300 Einzelbilder der Aizawa-Attraktor-Build-Up-Animation.
    Die Anzahl der Punkte steigt von Frame 0 bis Frame 299 linear von 10.000 auf 5.000.000 an.
    """
    output_dir = os.path.join("outputs", "animations", "buildup")
    os.makedirs(output_dir, exist_ok=True)

    # Parameterkonfiguration für den Aizawa-Attraktor
    attractor = AizawaAttractor(
        a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1, dt=0.01
    )

    # Auswahl der Auflösungskonfiguration und des Farb-Themes
    config = TARGET_CONFIGS[TargetPlatform.INSTA_PORTRAIT]
    theme = THEMES["neon_cyberpunk"]

    total_frames = 300
    min_points = 10_000
    max_points = 5_000_000
    iters_per_point = 50

    print(f"Die Generierung von {total_frames} Frames wird gestartet...")

    for frame_idx in range(total_frames):
        # Die Punktanzahl wird für den aktuellen Frame linear interpoliert
        progress = frame_idx / (total_frames - 1)
        num_points = int(min_points + progress * (max_points - min_points))

        # Das Dichtefeld des Aizawa-Attraktors wird berechnet
        density_map = attractor.generate_density_map(
            width=config.internal_width,
            height=config.internal_height,
            num_points=num_points,
            iters_per_point=iters_per_point
        )

        # Die Dichtekarte wird mittels des Shading-Moduls in ein RGB-Bild konvertiert
        image = render_image(density_map, theme)

        # Falls Supersampling verwendet wird, wird das Bild auf die Zielauflösung herunterskaliert
        if config.ssaa_factor > 1:
            image = image.resize((config.width, config.height), Image.Resampling.LANCZOS)

        # Das Bild wird als nummerierte PNG-Datei gespeichert
        filename = f"frame_{frame_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)

        if (frame_idx + 1) % 10 == 0 or frame_idx == total_frames - 1:
            print(f"Frame {frame_idx + 1}/{total_frames} erzeugt (Punkte: {num_points:,})...")

    print(f"Die Animation wurde erfolgreich im Verzeichnis '{output_dir}' abgelegt.")


if __name__ == "__main__":
    generate_buildup_animation()
