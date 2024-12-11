import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from multiprocessing import Pool
from PIL import Image
from funcs import (
    voronoi, 
    voronoi_map, 
    relax, 
    noise_map, 
    histeq, 
    average_cells, 
    fill_cells, 
    quantize, 
    color_cells, 
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
biome_colors = [
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

for i, color in enumerate(biome_colors):
    indices = np.where(np.all(im == color, axis=-1))
    biomes[indices] = i
    
biomes = np.flip(biomes, axis=0).T

class Map:
    def __init__(self, size=1024, n=256, map_seed=None): # 762345
        self.size = size
        self.n = n
        if map_seed is None:
            self.map_seed = np.random.randint(0, 1000000)
        else:
            self.map_seed = map_seed

        np.random.seed(self.map_seed)

    def generate(self):
        points = np.random.randint(0, self.size, (514, 2))
        vor = voronoi(points, self.size)
        vor_map = voronoi_map(vor, self.size)

        points = relax(points, self.size, k=100)
        vor = voronoi(points, self.size)
        vor_map = voronoi_map(vor, self.size)

        boundary_displacement = 8
        boundary_noise = np.dstack([noise_map(self.size, 32, self.map_seed+200, octaves=8), noise_map(self.size, 32, self.map_seed+250, octaves=8)])
        boundary_noise = np.indices((self.size, self.size)).T + boundary_displacement * boundary_noise
        boundary_noise = boundary_noise.clip(0, self.size - 1).astype(np.uint32)

        # Use advanced indexing to avoid the nested loops
        vor_map = vor_map[boundary_noise[..., 1], boundary_noise[..., 0]]

        temperature_map = noise_map(self.size, 2, self.map_seed+10)
        precipitation_map = noise_map(self.size, 2, self.map_seed+20)

        temperature_map = histeq(temperature_map, alpha=0.33)
        precipitation_map = histeq(precipitation_map, alpha=0.33)

        temperature_cells = average_cells(vor_map, temperature_map)
        precipitation_cells = average_cells(vor_map, precipitation_map)

        temperature_map = fill_cells(vor_map, temperature_cells)
        precipitation_map = fill_cells(vor_map, precipitation_cells)

        n = 256

        temperature_cells = quantize(temperature_cells, n)
        precipitation_cells = quantize(precipitation_cells, n)

        temperature_map = fill_cells(vor_map, temperature_cells)
        precipitation_map = fill_cells(vor_map, precipitation_cells)

        n = len(temperature_cells)
        biome_cells = np.zeros(n, dtype=np.uint32)

        for i in range(n):
            temp, precip = temperature_cells[i], precipitation_cells[i]
            biome_cells[i] = biomes[temp, precip]
            
        biome_map = fill_cells(vor_map, biome_cells).astype(np.uint32)
        biome_color_map = color_cells(biome_map, biome_colors)

        height_map = noise_map(self.size, 4, self.map_seed, octaves=6, persistence=0.5, lacunarity=2)
        land_mask = height_map > 0

        sea_color = np.array([12, 14, 255])
        land_mask_color = np.repeat(land_mask[:, :, np.newaxis], 3, axis=-1)

        height_map = noise_map(self.size, 4, self.map_seed, octaves=6, persistence=0.5, lacunarity=2)
        smooth_height_map = noise_map(self.size, 4, self.map_seed, octaves=1, persistence=0.5, lacunarity=2)

        biome_height_maps = [
            # Desert
            filter_map(height_map, smooth_height_map, 0.75, 0.2, 0.95, 0.2, 0.2, 0.5),
            # Savanna
            filter_map(height_map, smooth_height_map, 0.5, 0.1, 0.95, 0.1, 0.1, 0.2),
            # Tropical Woodland
            filter_map(height_map, smooth_height_map, 0.33, 0.33, 0.95, 0.1, 0.1, 0.75),
            # Tundra
            filter_map(height_map, smooth_height_map, 0.5, 1, 0.25, 1, 1, 1),
            # Seasonal Forest
            filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.2),
            # Rainforest
            filter_map(height_map, smooth_height_map, 0.5, 0.25, 0.66, 1, 1, 0.5),
            # Temperate forest
            filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
            # Temperate Rainforest
            filter_map(height_map, smooth_height_map, 0.75, 0.5, 0.4, 0.4, 0.33, 0.33),
            # Boreal
            filter_map(height_map, smooth_height_map, 0.8, 0.1, 0.9, 0.05, 0.05, 0.1)
        ]

        biome_count = len(biome_names)
        biome_masks = np.zeros((biome_count, self.size, self.size))

        for i in range(biome_count):
            biome_masks[i, biome_map==i] = 1
            biome_masks[i] = gaussian_filter(biome_masks[i], sigma=16)

        # Remove ocean from masks
        blurred_land_mask = land_mask
        blurred_land_mask = binary_dilation(land_mask, iterations=32).astype(np.float64)
        blurred_land_mask = gaussian_filter(blurred_land_mask, sigma=16)

        biome_masks = biome_masks*blurred_land_mask

        adjusted_height_map = height_map.copy()

        for i in range(len(biome_height_maps)):
            adjusted_height_map = (1-biome_masks[i])*adjusted_height_map + biome_masks[i]*biome_height_maps[i]
        
        biome_bound = get_boundary(biome_map, kernel=5)
        cell_bound = get_boundary(vor_map, kernel=2)

        river_mask = noise_map(self.size, 4, self.map_seed+4353, octaves=6, persistence=0.5, lacunarity=2) > 0

        new_biome_bound = biome_bound*(adjusted_height_map<0.5)*land_mask
        new_cell_bound = cell_bound*(adjusted_height_map<0.05)*land_mask

        rivers = np.logical_or(new_biome_bound, new_cell_bound)*river_mask

        loose_river_mask = binary_dilation(rivers, iterations=8)
        rivers_height = gaussian_filter(rivers.astype(np.float64), sigma=2)*loose_river_mask

        adjusted_height_river_map = adjusted_height_map*(1-rivers_height) - 0.05*rivers

        river_land_mask = adjusted_height_river_map >= 0
        land_mask_color = np.repeat(river_land_mask[:, :, np.newaxis], 3, axis=-1)
        rivers_biome_color_map = land_mask_color*biome_color_map + (1-land_mask_color)*sea_color

        tree_densities = [4000, 1500, 8000, 1000, 10000, 25000, 10000, 20000, 5000]

        # Parallel execution
        with Pool() as pool:
            trees = pool.starmap(
                place_trees, 
                [(tree_densities[i], biome_masks[i], self.size, river_land_mask, adjusted_height_river_map) for i in range(len(biome_names))]
            )
        
        color_map = apply_height_map(rivers_biome_color_map, adjusted_height_river_map, adjusted_height_river_map, river_land_mask)

        plt.figure(dpi=150, figsize=(5, 5))
        for k in range(len(biome_names)):
            plt.scatter(*trees[k].T, s=0.15, c="red")

        plt.imshow(color_map[0])

if __name__ == '__main__':
    m = Map()
    m.generate()
    plt.show()