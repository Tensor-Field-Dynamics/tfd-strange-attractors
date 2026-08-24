import sys
import os
import time
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.attractor_engine import CliffordAttractor
from core.shading import render_image
from config.resolutions import TARGET_CONFIGS, TargetPlatform
from config.themes import THEMES

def generate_theme_showcase():
    output_dir = os.path.join("outputs", "theme_showcase")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target resolution configuration
    target = TargetPlatform.INSTA_PORTRAIT
    config = TARGET_CONFIGS[target]
    
    print(f"==================================================")
    print(f"  Clifford Attractor Theme Showcase Generation")
    print(f"==================================================")
    print(f"Zielauflösung: {config.width}x{config.height} (SSAA {config.ssaa_factor}x -> {config.internal_width}x{config.internal_height})")
    print(f"Ausgabeordner: {output_dir}\n")

    # Fixed parameters for Clifford Attractor for identical perspective/structure across all themes
    a_param, b_param, c_param, d_param = -1.4, 1.6, 1.0, 0.7
    
    # We iterate over all themes in THEMES
    for theme_name, theme in THEMES.items():
        print(f"--> Generiere Rendering für Theme: '{theme_name}'...")
        
        # Set manual seed for identical initial point sampling
        torch.manual_seed(42)
        
        attractor = CliffordAttractor(a=a_param, b=b_param, c=c_param, d=d_param)
        
        t0 = time.time()
        density = attractor.generate_density_map(
            width=config.internal_width,
            height=config.internal_height,
            num_points=5_000_000,
            iters_per_point=50
        )
        t_calc = time.time() - t0
        
        # Post-Processing & Shading
        img = render_image(density, theme)
        if config.ssaa_factor > 1:
            img = img.resize((config.width, config.height), resample=Image.Resampling.LANCZOS)
            
        file_name = f"clifford_{theme_name}.png"
        file_path = os.path.join(output_dir, file_name)
        img.save(file_path)
        print(f"    Abgeschlossen in {t_calc:.2f}s | Gespeichert: {file_path}")

    print(f"\nAlle {len(THEMES)} Themes erfolgreich im Ordner '{output_dir}' gerendert!")

if __name__ == "__main__":
    generate_theme_showcase()
