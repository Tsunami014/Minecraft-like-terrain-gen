import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from funcs import (
    voronoi, 
    voronoi_map, 
    noise_map, 
    histeq, 
    average_cells, 
    fill_cells, 
    quantize, 
    colour_cells, 
    filter_map, 
    get_boundary, 
    place_trees, 
    apply_height_map
)

im = np.array(Image.open("TP_map.png"))[:, :, :3]
biomes = np.zeros((256, 256))

biome_names = [
  "desert",
  "savanna",
  "tropical_woodland",
  "tundra",
  "seasonal_forest",
  "rainforest",
  "temperate_forest",
  "temperate_rainforest",
  "boreal_forest"
]
biome_colours = [
  [255, 255, 178],
  [184, 200, 98],
  [188, 161, 53],
  [190, 255, 242],
  [106, 144, 38],
  [33, 77, 41],
  [86, 179, 106],
  [34, 61, 53],
  [35, 114, 94]
]

for i, colour in enumerate(biome_colours):
    indices = np.where(np.all(im == colour, axis=-1))
    biomes[indices] = i
    
biomes = np.flip(biomes, axis=0).T

class Map:
    def __init__(self, size=1024, n=256, map_seed=None):
        self.size = size
        self.n = n
        self.map_seed = np.random.randint(0, 1000000) if map_seed is None else map_seed
        np.random.seed(self.map_seed)

    def initialize_voronoi(self):
        points = np.random.randint(0, self.size, (514, 2))
        vor = voronoi(points, self.size)
        return voronoi_map(vor, self.size)

    def apply_boundary_noise(self, vor_map):
        boundary_displacement = 8
        boundary_noise = np.dstack([
            noise_map(self.size, 32, self.map_seed + 200, octaves=8), 
            noise_map(self.size, 32, self.map_seed + 250, octaves=8)
        ])
        boundary_noise = np.indices((self.size, self.size)).T + boundary_displacement * boundary_noise
        boundary_noise = boundary_noise.clip(0, self.size - 1).astype(np.uint32)
        return vor_map[boundary_noise[..., 1], boundary_noise[..., 0]]

    def generate_noise_maps(self):
        temp_map = noise_map(self.size, 2, self.map_seed + 10)
        precip_map = noise_map(self.size, 2, self.map_seed + 20)
        return histeq(temp_map, alpha=0.33), histeq(precip_map, alpha=0.33)

    def quantize_and_fill(self, vor_map, temp_map, precip_map):
        temp_cells = average_cells(vor_map, temp_map)
        precip_cells = average_cells(vor_map, precip_map)
        temp_map = fill_cells(vor_map, temp_cells)
        precip_map = fill_cells(vor_map, precip_cells)
        n = 256
        return quantize(temp_cells, n), quantize(precip_cells, n)

    def assign_biomes(self, vor_map, temp_cells, precip_cells):
        n = len(temp_cells)
        biome_cells = np.zeros(n, dtype=np.uint32)

        for i in range(n):
            temp, precip = temp_cells[i], precip_cells[i]
            biome_cells[i] = biomes[temp, precip]
        
        biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
        return biome_map, colour_cells(biome_map, biome_colours)

    def generate_height_maps(self):
        height_map = noise_map(self.size, 4, self.map_seed, octaves=6, persistence=0.5, lacunarity=2)
        smooth_height_map = noise_map(self.size, 4, self.map_seed, octaves=1, persistence=0.5, lacunarity=2)
        return height_map, smooth_height_map

    def create_masks(self, height_map):
        land_mask = height_map > 0
        blurred_land_mask = binary_dilation(land_mask, iterations=32).astype(np.float64)
        blurred_land_mask = gaussian_filter(blurred_land_mask, sigma=16)
        return land_mask, blurred_land_mask

    def generate_rivers(self, adjusted_height_map, biome_map, vor_map, land_mask):
        biome_bound = get_boundary(biome_map, kernel=5)
        cell_bound = get_boundary(vor_map, kernel=2)
        river_mask = noise_map(self.size, 4, self.map_seed + 4353, octaves=6, persistence=0.5, lacunarity=2) > 0
        new_biome_bound = biome_bound * (adjusted_height_map < 0.5) * land_mask
        new_cell_bound = cell_bound * (adjusted_height_map < 0.05) * land_mask
        rivers = np.logical_or(new_biome_bound, new_cell_bound) * river_mask
        loose_river_mask = binary_dilation(rivers, iterations=8)
        rivers_height = gaussian_filter(rivers.astype(np.float64), sigma=2) * loose_river_mask
        return rivers, rivers_height

    def place_trees_parallel(self, biome_masks, river_land_mask, adjusted_height_river_map):
        tree_densities = [4000, 1500, 8000, 1000, 10000, 25000, 10000, 20000, 5000]
        with ThreadPoolExecutor() as executor:
            return list(executor.map(
                place_trees, 
                tree_densities, 
                biome_masks, 
                [self.size] * len(biome_names), 
                [river_land_mask] * len(biome_names), 
                [adjusted_height_river_map] * len(biome_names)
            ))

    def generate(self):
        with ThreadPoolExecutor() as executor:
            vor_map_future = executor.submit(self.initialize_voronoi)
            temp_precip_future = executor.submit(self.generate_noise_maps)
            height_maps_future = executor.submit(self.generate_height_maps)

            vor_map = vor_map_future.result()
            vor_map = self.apply_boundary_noise(vor_map)

            temp_map, precip_map = temp_precip_future.result()
            temp_map, precip_map = self.quantize_and_fill(vor_map, temp_map, precip_map)

            biome_map, biome_colour_map = self.assign_biomes(vor_map, temp_map, precip_map)

            height_map, smooth_height_map = height_maps_future.result()
            land_mask, blurred_land_mask = self.create_masks(height_map)

            biome_masks = np.zeros((len(biome_names), self.size, self.size))
            for i in range(len(biome_names)):
                biome_masks[i, biome_map == i] = 1
                biome_masks[i] = gaussian_filter(biome_masks[i], sigma=16)
            biome_masks *= blurred_land_mask

            adjusted_height_map = height_map.copy()
            for i in range(len(biome_masks)):
                adjusted_height_map = (1 - biome_masks[i]) * adjusted_height_map + biome_masks[i] * filter_map(
                    height_map, smooth_height_map, 
                    0.75, 0.2, 0.95, 0.2, 0.2, 0.5
                )

            rivers, rivers_height = self.generate_rivers(adjusted_height_map, biome_map, vor_map, land_mask)
            adjusted_height_river_map = adjusted_height_map * (1 - rivers_height) - 0.05 * rivers
            river_land_mask = adjusted_height_river_map >= 0
            rivers_biome_colour_map = np.repeat(river_land_mask[:, :, np.newaxis], 3, axis=-1) * biome_colour_map + (1 - river_land_mask)[:, :, np.newaxis] * np.array([12, 14, 255])

            trees = self.place_trees_parallel(biome_masks, river_land_mask, adjusted_height_river_map)

            colour_map, _ = apply_height_map(rivers_biome_colour_map, adjusted_height_river_map, adjusted_height_river_map, river_land_mask)
            # colour_min, colour_max = colour_map.min(), colour_map.max()
            # colour_map = ((colour_map - colour_min) / (colour_max - colour_min) * 255).astype(np.uint8)
            colour_map = np.clip(colour_map, 0, 255)

            plt.figure(dpi=150, figsize=(5, 5))
            for k in range(len(biome_names)):
                plt.scatter(*trees[k].T, s=0.15, c="red")
            plt.imshow(colour_map)

if __name__ == '__main__':
    m = Map()
    m.generate()
    plt.show()