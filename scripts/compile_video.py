"""
Dieses Modul dient der Erstellung eines MP4-Videos aus einer Sequenz von PNG-Bildern.
Die Quelldateien werden aus einem spezifischen Verzeichnis gelesen und mittels OpenCV
verarbeitet und exportiert.
"""

import os
import glob
import cv2


def compile_video() -> None:
    """
    Sammelt alle PNG-Bilder aus dem Zielverzeichnis, sortiert diese alphabetisch
    und fügt sie zu einer MP4-Videodatei zusammen.
    """
    input_dir = os.path.join("outputs", "animations")
    output_dir = os.path.join("outputs", "feed")
    output_filepath = os.path.join(output_dir, "reel_01.mp4")

    # Sicherstellung, dass das Zielverzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    # Alle PNG-Bilder werden gesucht und alphabetisch sortiert
    pattern = os.path.join(input_dir, "*.png")
    image_files = sorted(glob.glob(pattern))

    if not image_files:
        print(f"Es wurden keine PNG-Bilder im Verzeichnis '{input_dir}' gefunden.")
        return

    # Das erste Bild wird geladen, um die Dimensionen des Videos zu bestimmen
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Das Bild '{image_files[0]}' konnte nicht gelesen werden.")
        return

    height, width, _ = first_image.shape
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Der VideoWriter wird initialisiert (Framerate: 30 FPS, Codec: mp4v)
    video_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

    print(f"Die Verstellung des Videos wird gestartet: {len(image_files)} Bilder verarbeiten...")

    # Jedes Bild wird eingelesen und in den Videostream geschrieben
    for filepath in image_files:
        img = cv2.imread(filepath)
        if img is not None:
            video_writer.write(img)
        else:
            print(f"Warnung: Das Bild '{filepath}' konnte nicht verarbeitet werden.")

    # Die Ressourcen des VideoWriters werden freigegeben
    video_writer.release()
    print(f"Das Video wurde erfolgreich unter '{output_filepath}' gespeichert.")


if __name__ == "__main__":
    compile_video()
