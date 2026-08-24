import os

new_classes = """

class LorenzAttractor:
    def __init__(self, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0, dt: float = 0.01, device: str = "cuda"):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 40) - 20
        y = (torch.rand(num_points, device=self.device) * 40) - 20
        z = (torch.rand(num_points, device=self.device) * 40) - 0

        for _ in range(iters_per_point):
            dx = (self.sigma * (y - x)) * self.dt
            dy = (x * (self.rho - z) - y) * self.dt
            dz = (x * y - self.beta * z) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 30.0) / 60.0 * width).to(torch.long)
            y_pixel = ((y + 30.0) / 60.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class DadrasAttractor:
    def __init__(self, a: float = 3.0, b: float = 2.7, c: float = 1.7, d: float = 2.0, e: float = 9.0, dt: float = 0.01, device: str = "cuda"):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 20) - 10
        y = (torch.rand(num_points, device=self.device) * 20) - 10
        z = (torch.rand(num_points, device=self.device) * 20) - 10

        for _ in range(iters_per_point):
            dx = (y - self.a * x + self.b * y * z) * self.dt
            dy = (self.c * y - x * z + z) * self.dt
            dz = (self.d * x * y - self.e * z) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 20.0) / 40.0 * width).to(torch.long)
            y_pixel = ((y + 20.0) / 40.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class ChenAttractor:
    def __init__(self, alpha: float = 5.0, beta: float = -10.0, delta: float = -0.38, dt: float = 0.01, device: str = "cuda"):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 40) - 20
        y = (torch.rand(num_points, device=self.device) * 40) - 20
        z = (torch.rand(num_points, device=self.device) * 40) - 20

        for _ in range(iters_per_point):
            dx = (self.alpha * x - y * z) * self.dt
            dy = (self.beta * y + x * z) * self.dt
            dz = (self.delta * z + (x * y) / 3.0) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 30.0) / 60.0 * width).to(torch.long)
            y_pixel = ((y + 30.0) / 60.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class Lorenz83Attractor:
    def __init__(self, a: float = 0.95, b: float = 7.91, f: float = 4.83, g: float = 4.66, dt: float = 0.01, device: str = "cuda"):
        self.a = a
        self.b = b
        self.f = f
        self.g = g
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 10) - 5
        y = (torch.rand(num_points, device=self.device) * 10) - 5
        z = (torch.rand(num_points, device=self.device) * 10) - 5

        for _ in range(iters_per_point):
            dx = (-self.a * x - y**2 - z**2 + self.a * self.f) * self.dt
            dy = (-y + x * y - self.b * x * z + self.g) * self.dt
            dz = (-z + self.b * x * y + x * z) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 10.0) / 20.0 * width).to(torch.long)
            y_pixel = ((y + 10.0) / 20.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class RosslerAttractor:
    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7, dt: float = 0.01, device: str = "cuda"):
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 20) - 10
        y = (torch.rand(num_points, device=self.device) * 20) - 10
        z = (torch.rand(num_points, device=self.device) * 20) - 10

        for _ in range(iters_per_point):
            dx = -(y + z) * self.dt
            dy = (x + self.a * y) * self.dt
            dz = (self.b + z * (x - self.c)) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 20.0) / 40.0 * width).to(torch.long)
            y_pixel = ((y + 20.0) / 40.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class HalvorsenAttractor:
    def __init__(self, a: float = 1.89, dt: float = 0.01, device: str = "cuda"):
        self.a = a
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 10) - 5
        y = (torch.rand(num_points, device=self.device) * 10) - 5
        z = (torch.rand(num_points, device=self.device) * 10) - 5

        for _ in range(iters_per_point):
            dx = (-self.a * x - 4 * y - 4 * z - y**2) * self.dt
            dy = (-self.a * y - 4 * z - 4 * x - z**2) * self.dt
            dz = (-self.a * z - 4 * x - 4 * y - x**2) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 15.0) / 30.0 * width).to(torch.long)
            y_pixel = ((y + 15.0) / 30.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class RabinovichFabrikantAttractor:
    def __init__(self, alpha: float = 0.14, gamma: float = 0.10, dt: float = 0.01, device: str = "cuda"):
        self.alpha = alpha
        self.gamma = gamma
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 2) - 1
        y = (torch.rand(num_points, device=self.device) * 2) - 1
        z = (torch.rand(num_points, device=self.device) * 2) - 1

        for _ in range(iters_per_point):
            dx = (y * (z - 1 + x**2) + self.gamma * x) * self.dt
            dy = (x * (3 * z + 1 - x**2) + self.gamma * y) * self.dt
            dz = (-2 * z * (self.alpha + x * y)) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 5.0) / 10.0 * width).to(torch.long)
            y_pixel = ((y + 5.0) / 10.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()


class ThreeScrollUnifiedAttractor:
    def __init__(self, a: float = 32.48, b: float = 45.84, c: float = 1.18, d: float = 0.13, e: float = 0.57, f: float = 14.7, dt: float = 0.001, device: str = "cuda"):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.dt = dt
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def generate_density_map(self, width: int, height: int, num_points: int, iters_per_point: int) -> np.ndarray:
        density_map = torch.zeros((height, width), dtype=torch.float32, device="cpu")
        x = (torch.rand(num_points, device=self.device) * 20) - 10
        y = (torch.rand(num_points, device=self.device) * 20) - 10
        z = (torch.rand(num_points, device=self.device) * 20) - 10

        for _ in range(iters_per_point):
            dx = (self.a * (y - x) + self.d * x * z) * self.dt
            dy = (self.b * x - x * z + self.f * y) * self.dt
            dz = (self.c * z + x * y - self.e * (x**2)) * self.dt
            
            x += dx
            y += dy
            z += dz

            x_pixel = ((x + 50.0) / 100.0 * width).to(torch.long)
            y_pixel = ((y + 50.0) / 100.0 * height).to(torch.long)

            valid_mask = (x_pixel >= 0) & (x_pixel < width) & (y_pixel >= 0) & (y_pixel < height)
            x_valid = x_pixel[valid_mask].cpu()
            y_valid = y_pixel[valid_mask].cpu()

            density_map.index_put_((y_valid, x_valid), torch.tensor(1.0), accumulate=True)

        return density_map.numpy()
"""

with open(r"c:\Tensor Field Dynamics\tfd-strange-attractors\tfd-strange-attractors\core\attractor_engine.py", "a", encoding="utf-8") as f:
    f.write(new_classes)

print("Added new classes!")
