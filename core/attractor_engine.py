import torch
import numpy as np

class CliffordAttractor:
    """
    In dieser Klasse wird die Berechnung des Clifford-Attraktors gekapselt.
    Die Trajektorien werden massiv-parallel auf der GPU iteriert.
    Um den VRAM-Verbrauch zu minimieren, wird eine Batch-Verarbeitung angewandt.
    """
    def __init__(self, a: float, b: float, c: float, d: float, device: str = "cuda"):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        """
        Ein 2D-Histogramm (Dichtefeld) wird durch Iteration der Attraktor-Gleichungen akkumuliert.
        Langsame CPU-Schleifen werden hierbei durch vektorisierte Tensor-Operationen ersetzt.
        """
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = torch.rand(num_points, device=self.device) * 2 - 1
        y = torch.rand(num_points, device=self.device) * 2 - 1

        for _ in range(iters_per_point):
            x_new = torch.sin(self.a * y) + self.c * torch.cos(self.a * x)
            y_new = torch.sin(self.b * x) + self.d * torch.cos(self.b * y)
            x, y = x_new, y_new

            x_pixel = ((x + 2.5) / 5.0 * width).to(torch.long)
            y_pixel = ((y + 2.5) / 5.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class ThomasAttractor:
    """
    In dieser Klasse wird das System des zyklisch symmetrischen Thomas-Attraktors implementiert.
    Da es sich um ein kontinuierliches System von Differentialgleichungen handelt,
    wird die Trajektorie mittels des expliziten Euler-Verfahrens numerisch integriert.
    """
    def __init__(self, b: float = 0.19, dt: float = 0.05, device: str = "cuda"):
        self.b = b
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        """
        Die 3D-Koordinaten werden in Vektorform berechnet und für die 2D-Ausgabe 
        durch eine orthogonale Projektion auf eine Schnittebene reduziert.
        """
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 4) - 2
        y = (torch.rand(num_points, device=self.device) * 4) - 2
        z = (torch.rand(num_points, device=self.device) * 4) - 2

        for _ in range(iters_per_point):
            dx = (torch.sin(y) - self.b * x) * self.dt
            dy = (torch.sin(z) - self.b * y) * self.dt
            dz = (torch.sin(x) - self.b * z) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 6.0) / 12.0 * width).to(torch.long)
            y_pixel = ((y + 6.0) / 12.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()