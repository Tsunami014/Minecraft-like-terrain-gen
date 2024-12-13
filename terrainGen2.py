import pygame
import math
from copy import deepcopy
import noise

clock = pygame.time.Clock()
pygame.init()
pygame.display.set_caption('3D Terrain')
screen = pygame.display.set_mode((500, 500))

FOV = 90
FOG = False
CHUNK_SIZE = 30

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
    for point in polygon:
        if 0 <= point[0] < screen.get_width() and 0 <= point[1] < screen.get_height():
            return True
    return False

def project_polygon(polygon, player_rot):
    projected_points = []
    for point in polygon:
        rotated_point = rotate_point(point, player_rot)
        if rotated_point[2] <= 0:
            continue  # Skip points behind the camera
        factor = screen.get_width() / (2 * math.tan(math.radians(FOV) / 2))
        x = (rotated_point[0] * factor) / rotated_point[2] + screen.get_width() / 2
        y = (rotated_point[1] * factor) / rotated_point[2] + screen.get_height() / 2
        projected_points.append([x, y])
    if is_polygon_on_screen(projected_points):
        return projected_points
    return []

def gen_polygon(polygon_base, player_pos, player_rot):
    generated_polygon = deepcopy(polygon_base)
    # Translate polygon to camera space
    offset_polygon(generated_polygon, [-player_pos[0], -player_pos[1], -player_pos[2]])
    projected_points = []
    depths = []
    for point in generated_polygon:
        rotated_point = rotate_point(point, player_rot)
        if rotated_point[2] <= 0:
            continue  # Skip points behind the camera
        factor = screen.get_width() / (2 * math.tan(math.radians(FOV) / 2))
        x = (rotated_point[0] * factor) / rotated_point[2] + screen.get_width() / 2
        y = (rotated_point[1] * factor) / rotated_point[2] + screen.get_height() / 2
        projected_points.append([x, y])
        depths.append(rotated_point[2])
    if is_polygon_on_screen(projected_points) and projected_points:
        avg_depth = sum(depths) / len(depths)
        return projected_points, avg_depth
    return [], 0

pos = [0, 0, 0]
rot = [0, 0, 0]

square_polygon = [
    [-0.5, 0, -0.5],
    [0.5, 0, -0.5],
    [0.5, 0, 0.5],
    [-0.5, 0, 0.5],
]

def generate_poly(x, y):
    poly_copy = deepcopy(square_polygon)
    offset_polygon(poly_copy, [x, 0, y])

    water = True
    depth = 0

    for corner in poly_copy:
        v = noise.pnoise2(corner[0] / 10, corner[2] / 10, octaves=2) * 3
        v2 = noise.pnoise2(corner[0] / CHUNK_SIZE + 1000, corner[2] / CHUNK_SIZE)
        if v < 0:
            depth -= v
            v = 0
        else:
            water = False
        corner[1] -= v * 4.5

    if water:
        c = (0, min(255, max(0, 150 - depth * 25)), min(255, max(0, 255 - depth * 25)))
    else:
        c = (CHUNK_SIZE - v * 10 + v2 * CHUNK_SIZE, 50 + v2 * 40 + v * CHUNK_SIZE, 50 + v * 10)

    return [poly_copy, c]

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

    rot_by = math.radians(1)*speed
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

    # Sort polygons by depth (from farthest to nearest)
    polygons_to_render.sort(reverse=True)

    bg = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    bg.fill(0)

    if not polygons_to_render:
        maxDepth = 0
    else:
        maxDepth = max(i[0] for i in polygons_to_render)
    for depth, render_poly, colour in polygons_to_render:
        colour = (*colour, (1-depth/maxDepth)*255)
        pygame.draw.polygon(bg, colour, render_poly)

    screen.blit(bg, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                run = False

    pygame.display.update()
    clock.tick(60)