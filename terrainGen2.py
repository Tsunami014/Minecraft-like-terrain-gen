import json
import pygame
import math
import numpy as np
from scipy.ndimage import gaussian_filter
import random
import noise

clock = pygame.time.Clock()
pygame.init()
pygame.display.set_caption('3D Terrain')
screen = pygame.display.set_mode((500, 500))

FOV = 90
FOG = True
SEED = random.randint(-100000, 100000)
SIZE = 4
CHUNK_SIZE = 40
BIOME_SIZE = 1
OUTLINE = 1
BLUR_AMNT = 3

TP_map = pygame.image.load('TP_map.png')

TRUNK_COLOR = (139, 69, 19)
FOLIAGE_COLOR = (34, 139, 34)

def offset_polygon(polygon, offset):
    for point in polygon:
        point[0] += offset[0]
        point[1] += offset[1]
        point[2] += offset[2]

def rotate_point(point, rot):
    x, y, z = point

    # Rotation around Y-axis (yaw)
    cos_yaw = math.cos(rot[1])
    sin_yaw = math.sin(rot[1])
    x1 = x * cos_yaw + z * sin_yaw
    z1 = -x * sin_yaw + z * cos_yaw

    # Rotation around X-axis (pitch)
    cos_pitch = math.cos(rot[0])
    sin_pitch = math.sin(rot[0])
    y1 = y * cos_pitch - z1 * sin_pitch
    z2 = y * sin_pitch + z1 * cos_pitch


    # Rotation around Z-axis (roll)
    cos_roll = math.cos(rot[2])
    sin_roll = math.sin(rot[2])
    x2 = x1 * cos_roll - y1 * sin_roll
    y2 = x1 * sin_roll + y1 * cos_roll

    return [x2, y2, z2]

def is_polygon_on_screen(polygon):
    if len(polygon) == 0:
        return False
    xs, ys = zip(*polygon)
    scrw, scrh = screen.get_size()
    if max(xs) < 0 or min(xs) > scrw or max(ys) < 0 or min(ys) > scrh:
        return False
    return True

def project_polygon(polygon, player_rot):
    projected_points = []
    scrw, scrh = screen.get_size()
    for point in polygon:
        rotated_point = rotate_point(point, player_rot)
        if rotated_point[2] <= 0:
            continue  # Skip points behind the camera
        factor = scrw / (2 * math.tan(math.radians(FOV) / 2))
        x = (rotated_point[0] * factor) / rotated_point[2] + scrw / 2
        y = (rotated_point[1] * factor) / rotated_point[2] + scrh / 2
        projected_points.append([x, y])
    if is_polygon_on_screen(projected_points):
        return projected_points
    return []

def gen_polygon(polygon_base, player_pos, player_rot):
    generated_polygon = [point[:] for point in polygon_base]  # Manual copy of the polygon
    # Translate polygon to camera space
    offset_polygon(generated_polygon, [-player_pos[0], -player_pos[1], -player_pos[2]])
    projected_points = []
    depths = []
    scrw, scrh = screen.get_size()
    hsw, hsh = scrw / 2, scrh / 2
    factordiv = 2 * math.tan(math.radians(FOV) / 2)
    for point in generated_polygon:
        rotated_point = rotate_point(point, player_rot)
        if rotated_point[2] <= 0:
            continue  # Skip points behind the camera
        factor = scrw / factordiv
        x = (rotated_point[0] * factor) / rotated_point[2] + hsw
        y = (rotated_point[1] * factor) / rotated_point[2] + hsh
        projected_points.append([x, y])
        depths.append(math.sqrt(rotated_point[0]**2 + rotated_point[1]**2 + rotated_point[2]**2))
    if is_polygon_on_screen(projected_points) and projected_points:
        avg_depth = sum(depths) / len(depths)
        return projected_points, avg_depth
    return [], 0

pos = [0, 0, 0]
rot = [0, 0, 0]

BIOME_STATS = json.load(open('biomes.json'))

d = BIOME_STATS['Colours']
COLOUR2NAME = {tuple(d[i]): i for i in d}
arr = pygame.surfarray.array3d(TP_map)
BLURCOLOURS = gaussian_filter(arr, sigma = BLUR_AMNT, mode='nearest')
BLURS = {
    i: gaussian_filter(
        np.apply_along_axis(lambda v: BIOME_STATS[i][COLOUR2NAME[tuple(v)]], 2, arr), 
        sigma = BLUR_AMNT, mode='nearest') for i in BIOME_STATS.keys() if i != 'Colours'
}

def generate_poly(x, y):
    poly = [
        [-0.5, 0, -0.5],
        [0.5, 0, -0.5],
        [0.5, 0, 0.5],
        [-0.5, 0, 0.5],
    ]
    offset_polygon(poly, [x, 0, y])

    water = True
    depth = 0
    temp = 0
    precip = 0

    map_w, map_h = TP_map.get_size()

    for corner in poly:
        thistemp = noise.pnoise2(((corner[0] / CHUNK_SIZE + 1000)/BIOME_SIZE)/SIZE + SEED, ((corner[2] / CHUNK_SIZE + 1000)/BIOME_SIZE)/SIZE + SEED)
        thisprecip = noise.pnoise2(((corner[0] / CHUNK_SIZE + 2000)/BIOME_SIZE)/SIZE + SEED, ((corner[2] / CHUNK_SIZE + 2000)/BIOME_SIZE)/SIZE + SEED)

        pos = (int(min(map_w-1, max(0, (map_w/2) - (thistemp*4) * (map_w/4)))), int(min(map_h-1, max(0, (map_h/2) - (thisprecip*4) * (map_h/4)))))

        v1 = noise.pnoise2((corner[0] / 10)/SIZE + SEED, (corner[2] / 10)/SIZE + SEED, octaves=2) * 2
        v2 = (noise.pnoise2((corner[0] / 10)/SIZE + SEED - 1000, (corner[2] / 10)/SIZE + SEED - 2000, octaves=2)+0.5)
        v2 = (round(v2 * 10) / 10) * 2
        v3 = noise.pnoise2(((corner[0] / 10)/SIZE + SEED + 1000) * BLURS['Mountanicity'][pos], ((corner[2] / 10)/SIZE + SEED - 1000) * BLURS['Mountanicity'][pos], octaves=2)
        v3 = (round(v3 * 3) / 3) * (BLURS['Mountanicity'][pos] / 2)
        v = v1 * v2 + v3

        if v < 0:
            depth -= v
            v = 0
        else:
            water = False
        temp += thistemp
        precip += thisprecip
        corner[1] -= (v * BLURS['Variation'][pos]) * 4.5

    if water:
        c = (0, min(255, max(0, 150 - depth * 25)), min(255, max(0, 255 - depth * 25)))
    else:
        pos = (int(min(map_w-1, max(0, (map_w/2) - temp * (map_w/4)))), int(min(map_h-1, max(0, (map_h/2) - precip * (map_h/4)))))
        biome = BLURCOLOURS[pos]
        # biome_type = get_biome_type(biome)
        c = biome[:3]

    tree_polygons = None
    random.seed(y + SEED)
    random.seed(x * random.randint(-999999, 999999) + y + SEED)
    if not water and random.random() < BLURS['Trees'][pos]:
        tree_height = random.uniform(BLURS['TreeMinHeights'][pos], BLURS['TreeMaxHeights'][pos])
        trunk_height = tree_height * 0.6
        x_base = x
        y_base = max(i[1] for i in poly)
        z_base = y

        trunk = [
            [
                [x_base - 0.1, y_base, z_base - 0.1],
                [x_base + 0.1, y_base, z_base - 0.1],
                [x_base + 0.1, y_base - trunk_height, z_base - 0.1],
                [x_base - 0.1, y_base - trunk_height, z_base - 0.1],
            ],
            [
                [x_base - 0.1, y_base, z_base + 0.1],
                [x_base + 0.1, y_base, z_base + 0.1],
                [x_base + 0.1, y_base - trunk_height, z_base + 0.1],
                [x_base - 0.1, y_base - trunk_height, z_base + 0.1],
            ],
            [
                [x_base - 0.1, y_base, z_base - 0.1],
                [x_base - 0.1, y_base, z_base + 0.1],
                [x_base - 0.1, y_base - trunk_height, z_base + 0.1],
                [x_base - 0.1, y_base - trunk_height, z_base - 0.1],
            ],
            [
                [x_base + 0.1, y_base, z_base - 0.1],
                [x_base + 0.1, y_base, z_base + 0.1],
                [x_base + 0.1, y_base - trunk_height, z_base + 0.1],
                [x_base + 0.1, y_base - trunk_height, z_base - 0.1],
            ],
        ]

        foliage = [
            [
                [x_base, y_base - tree_height, z_base],
                [x_base - 0.5, y_base - trunk_height, z_base - 0.5],
                [x_base + 0.5, y_base - trunk_height, z_base - 0.5],
            ],
            [
                [x_base, y_base - tree_height, z_base],
                [x_base - 0.5, y_base - trunk_height, z_base + 0.5],
                [x_base + 0.5, y_base - trunk_height, z_base + 0.5],
            ],
            [
                [x_base, y_base - tree_height, z_base],
                [x_base + 0.5, y_base - trunk_height, z_base - 0.5],
                [x_base + 0.5, y_base - trunk_height, z_base + 0.5],
            ],
            [
                [x_base, y_base - tree_height, z_base],
                [x_base - 0.5, y_base - trunk_height, z_base - 0.5],
                [x_base - 0.5, y_base - trunk_height, z_base + 0.5],
            ],
            [
                [x_base - 0.5, y_base - trunk_height, z_base - 0.5],
                [x_base + 0.5, y_base - trunk_height, z_base - 0.5],
                [x_base + 0.5, y_base - trunk_height, z_base + 0.5],
                [x_base - 0.5, y_base - trunk_height, z_base + 0.5],
            ],
        ]

        tree_polygons = {'trunk': trunk, 'foliage': foliage}

    return [poly, c, tree_polygons]

def get_biome_type(colour):
    colour = tuple(colour[:3])
    return COLOUR2NAME[colour]

def generate_surround_polys(pos, polygons):
    newpolys = {}
    start_x = int(pos[0] - CHUNK_SIZE / 2)
    start_y = int(pos[2] - CHUNK_SIZE / 2)
    for y in range(CHUNK_SIZE):
        for x in range(CHUNK_SIZE):
            world_x = start_x + x
            world_y = start_y + y
            p = (world_x, world_y)
            if p in polygons:
                newpolys[p] = polygons[p]
            else:
                newpolys[p] = generate_poly(world_x, world_y)
    return newpolys

def move_player(pos, rot, direction, amount, strafe=False):
    if strafe:
        dx = amount * math.cos(rot[1])
        dz = amount * -math.sin(rot[1])
    else:
        dx = amount * math.sin(rot[1])
        dz = amount * math.cos(rot[1])
    dy = amount * -math.sin(rot[0])
    pos[0] -= dx * direction
    pos[1] -= dy * direction
    pos[2] += dz * direction

polygons = generate_surround_polys(pos, {})

pos[1] = -(polygons[(0, 0)][0][0][1] + 2)

run = True
while run:
    screen.fill((100, 200, 250))

    # move
    speed = (2 if pygame.key.get_mods() & pygame.KMOD_CTRL else 1)
    keys = pygame.key.get_pressed()
    moved = False
    movement = 0.25*speed
    if keys[pygame.K_w]:
        move_player(pos, rot, 1, movement)
        moved = True
    if keys[pygame.K_s]:
        move_player(pos, rot, -1, movement)
        moved = True
    if keys[pygame.K_a]:
        move_player(pos, rot, 1, movement, True)
        moved = True
    if keys[pygame.K_d]:
        move_player(pos, rot, -1, movement, True)
        moved = True
    
    if keys[pygame.K_q] or keys[pygame.K_SPACE]:
        pos[1] -= movement
    if keys[pygame.K_e] or pygame.key.get_mods() & pygame.KMOD_SHIFT:
        pos[1] += movement

    rot_by = math.radians(3)*speed
    if keys[pygame.K_UP]:
        rot[0] -= rot_by
    if keys[pygame.K_DOWN]:
        rot[0] += rot_by
    if keys[pygame.K_LEFT]:
        rot[1] += rot_by
    if keys[pygame.K_RIGHT]:
        rot[1] -= rot_by
    if keys[pygame.K_PERIOD]:
        rot[2] += rot_by
    if keys[pygame.K_SLASH]:
        rot[2] -= rot_by

    # generate new polygons if moved
    if moved:
        polygons = generate_surround_polys(pos, polygons)

    # render
    polygons_to_render = []

    for p, poly in polygons.items():
        render_poly, depth = gen_polygon(poly[0], pos, rot)
        if len(render_poly) >= 3:
            polygons_to_render.append((depth, render_poly, poly[1]))
        if len(poly) > 2 and poly[2]:
            tree = poly[2]
            for face in tree['trunk']:
                render_face, face_depth = gen_polygon(face, pos, rot)
                if len(render_face) >= 3:
                    polygons_to_render.append((face_depth, render_face, TRUNK_COLOR))
            for face in tree['foliage']:
                render_face, face_depth = gen_polygon(face, pos, rot)
                if len(render_face) >= 3:
                    polygons_to_render.append((face_depth, render_face, FOLIAGE_COLOR))

    # Sort polygons by depth (from farthest to nearest)
    polygons_to_render.sort(reverse=True)

    if FOG:
        bg = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        bg.fill(0)

    if not polygons_to_render:
        maxDepth = 0
    else:
        maxDepth = max(i[0] for i in polygons_to_render)
    for depth, render_poly, colour in polygons_to_render:
        dpth = 1-(depth/maxDepth)
        dpth = (dpth*2)**2
        colour = (*colour, min(dpth*255, 255))
        if FOG:
            pygame.draw.polygon(bg, colour, render_poly)
            if OUTLINE > 0:
                pygame.draw.polygon(bg, (colour[0]/2, colour[1]/2, colour[2]/2, colour[3]), render_poly, OUTLINE)
        else:
            pygame.draw.polygon(screen, colour, render_poly)
        if OUTLINE > 0:
            col = (colour[0]/2, colour[1]/2, colour[2]/2, colour[3])
            surf = (screen if not FOG else bg)
            pygame.draw.polygon(surf, col, render_poly, OUTLINE)

    if FOG:
        screen.blit(bg, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False

    pygame.display.update()
    clock.tick(60)