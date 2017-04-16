from math import sqrt

import pygame as pygame
from pygame.locals import *

pygame.init()


def get_list(l, indexes):
    return [l[i] for i in indexes]


def polarity(n):
    if n == 0:
        return n
    return n / abs(n)


def combine_lists(l1, l2, sign):
    l1 = list(l1)
    for i in range(2):
        if sign == '+':
            l1[i] += l2[i]
        elif sign == '-':
            l1[i] -= l2[i]
        elif sign == '*':
            l1[i] *= l2[i]
        else:
            l1[i] /= l2[i]
    return l1


def opposite(n):
    return abs(n - 1)


def make_tuple(thing):
    if type(thing) not in (list, tuple):
        # noinspection PyRedundantParentheses
        return (thing,)
    return thing


def find_center(d1, d2, c1=(0, 0)):
    return [(c1[i] + (d1[i] / 2 - d2[i] / 2)) for i in range(2)]


def collision(c1, d1, c2, d2, inside_only=False):
    d1 = list(d1)
    d2 = list(d2)
    collisions = [False, False]
    for i in range(2):
        d1[i] -= 1
        d2[i] -= 1
        if (c1[i] <= c2[i] and c1[i] + d1[i] >= c2[i] + d2[i]) or \
                (c1[i] >= c2[i] and c1[i] + d1[i] <= c2[i] + d2[i]):
            collisions[i] = True
        if not inside_only:
            if (c2[i] <= c1[i] <= c2[i] + d2[i]) or \
                    (c2[i] <= c1[i] + d1[i] <= c2[i] + d2[i]):
                collisions[i] = True
    if False not in collisions:
        return True
    elif True in collisions:
        return collisions.index(False)


class SpriteSheet:
    def __init__(self, filename, division_index=1):
        self.sheet = pygame.image.load(filename).convert_alpha()
        self.division_index = division_index
        self.farthest_y_coordinate = 0

    def get_image(self, coordinates, dimensions):
        # noinspection PyArgumentList
        image = pygame.Surface(dimensions, SRCALPHA).convert_alpha()
        image.blit(self.sheet, (0, 0), (coordinates, dimensions))
        return image

    def get_sprites(self, starting_x_coordinate=0, farthest_y_coordinate=None, all_dimensions=None, y_constant=None,
                    x_constant=None, update=True, scale=3, block_number=None, dimensions=None):
        sprites = []
        if block_number:
            y_constant = game.block_size
            x_constant = (game.block_size, block_number)
        elif dimensions:
            y_constant = dimensions[1]
            x_constant = (dimensions[0], 1)

        if not farthest_y_coordinate:
            farthest_y_coordinate = self.farthest_y_coordinate
        if x_constant:
            thing = x_constant[1]
        else:
            thing = len(all_dimensions)
        farthest_x_coordinate = starting_x_coordinate

        for i in range(thing):
            coordinates = [0, 0]
            coordinates[opposite(self.division_index)] = farthest_x_coordinate
            coordinates[self.division_index] = farthest_y_coordinate

            dimensions = [0, 0]
            if x_constant or y_constant:
                if x_constant:
                    dimensions[opposite(self.division_index)] = x_constant[0]
                else:
                    dimensions[opposite(self.division_index)] = all_dimensions[i]
                if y_constant:
                    dimensions[self.division_index] = y_constant
                else:
                    dimensions[self.division_index] = all_dimensions[i]
            else:
                dimensions = all_dimensions[i]

            farthest_x_coordinate += dimensions[opposite(self.division_index)]
            sprite = self.get_image(coordinates, dimensions)
            if scale:
                sprite = pygame.transform.scale(sprite, combine_lists(sprite.get_size(),
                                                                      (scale, scale), '*'))
            sprites.append(sprite)

        if update:
            if y_constant:
                self.farthest_y_coordinate += y_constant
            elif x_constant:
                self.farthest_y_coordinate += max(all_dimensions)
            else:
                self.farthest_y_coordinate += max(
                    [dimensions[self.division_index] for dimensions in all_dimensions])

        return sprites


class Quadratic:
    def __init__(self, sign, y_range, speed):
        self.a = sign
        self.y_range = y_range
        if (sign == 1 and self.y_range[0] > self.y_range[1]) or (sign == -1 and self.y_range[1] > self.y_range[0]):
            x_solution_index = 0
        else:
            x_solution_index = 1
        self.c = self.y_range[opposite(x_solution_index)]
        self.x_range = [self.get_x(self.y_range[i])[x_solution_index] for i in range(2)]
        self.x_change = (self.x_range[1] - self.x_range[0]) / speed
        self.reset()

    def execute(self):
        self.current_x += self.x_change
        if self.current_x - .01 > self.x_range[1]:
            # noinspection PyRedundantParentheses
            return (self.y_change,)
        current_y = int(self.get_y(self.current_x))
        self.y_change = current_y - self.old_y
        self.old_y = current_y
        return self.y_change

    def get_y(self, x):
        return self.a * x ** 2 + self.c

    def get_x(self, y):
        x = sqrt((-self.c + y) / self.a)
        return sorted((-x, x))

    def reset(self):
        self.current_x = self.x_range[0]
        self.old_y = self.y_range[0]


class Game:
    def __init__(self, speed, dimensions, scale_factor, block_size, version):
        self.speed = speed
        self.real_speed = self.speed
        self.dimensions = dimensions
        self.scale_factor = scale_factor
        self.block_size = block_size
        self.version = version
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)
        self.movement_keys = (K_RIGHT, K_LEFT, K_d, K_a)
        self.font = pygame.font.SysFont("Calibri", 100)
        self.count = 0

    def blit(self, thing):
        self.display.blit(thing.current_sprite(),
                          player.generate_display_coordinates(thing.coordinates))


class Background:
    def __init__(self, sprites, level_maps,
                 block_color_values, block_offsets,
                 default_block_info, block_info,
                 block_types,
                 level_transition_height, level_transition_speed,
                 menu_options, menu_heights, menu_selectable_options,
                 selecter_gap, title_transition_time,
                 switch_pairs, difficulty,
                 level_number_margins, level_number_gap):
        self.sprites = sprites
        self.level_maps = level_maps

        self.block_color_values = block_color_values
        self.block_offsets = block_offsets
        for offset_name in self.block_offsets:
            if offset_name not in ('entrance_background', 'exit_background'):
                self.block_offsets[offset_name] = combine_lists(block_offsets[offset_name],
                                                                (game.scale_factor, game.scale_factor), '*')
        self.default_block_info = default_block_info
        self.block_info = block_info
        self.block_size = game.scale_factor * game.block_size
        self.block_types = block_types

        self.level_transition_coordinates = [
            (player.default_fake_coordinates[1], player.default_fake_coordinates[1] + level_transition_height),
            (player.default_fake_coordinates[1] - level_transition_height, player.default_fake_coordinates[1])
        ]
        self.level_transition_quadratics = [
            Quadratic(sign, self.level_transition_coordinates[i], level_transition_speed)
            for sign, i in zip((1, -1), (0, 1))
            ]
        self.level_transition_phase = None

        self.menu_coordinates = {}
        previous_height = 0
        for name in menu_options:
            height = previous_height + menu_heights[name]
            previous_height = height + self.sprites[name][0].get_height()
            self.menu_coordinates[name] = (find_center(game.dimensions, self.sprites[name][0].get_size())[0], height)
        self.menu_selectable_options = menu_selectable_options
        self.selecter_position_index = 0
        self.selecter_gap = selecter_gap * game.scale_factor
        self.update_selecter_coordinates()
        self.title_coordinates = (find_center(game.dimensions, self.sprites['title'][0].get_size()))
        self.title_transition_speed = (self.title_coordinates[1] - self.menu_coordinates['title'][
            1]) / title_transition_time

        self.switch_pairs = switch_pairs
        self.difficulty = difficulty
        self.level_number_margins = level_number_margins
        self.level_number_gap = level_number_gap
        self.number_width = self.sprites['number'][0].get_width()
        self.condition = 'title'
        self.phase = 'menu'
        self.level = 0
        self.block_color = self.sprites['block'][0].get_at((0, 0))
        self.background_color = self.sprites['background'][0].get_at((0, 0))
        self.directions = ((-1, 0), (0, -1), (1, 0), (0, 1))

    def update_selecter_coordinates(self):
        self.selecter_coordinates = (
            self.menu_coordinates[self.menu_selectable_options[self.selecter_position_index]][0] -
            self.sprites['selecter'][0].get_width() - self.selecter_gap,
            find_center(
                self.sprites[self.menu_selectable_options[self.selecter_position_index]][0].get_size(),
                self.sprites['selecter'][0].get_size(),
                self.menu_coordinates[self.menu_selectable_options[self.selecter_position_index]]
            )[1]
        )

    def initiate_level_transition(self):
        self.level_transition_phase = 0
        for quadratic in self.level_transition_quadratics:
            quadratic.reset()

    def block_type(self, grid_coordinates):
        if grid_coordinates in self.blocks:
            return self.blocks[grid_coordinates].kind

    def convert_from_grid(self, grid_coordinates):
        return combine_lists(grid_coordinates, (self.block_size, self.block_size), '*')

    def convert_to_grid(self, coordinates):
        return tuple([int((coordinates[i] - (coordinates[i] % self.block_size)) / self.block_size) for i in range(2)])

    def find_all_grid_coordinates(self, coordinates, dimensions):
        start = self.convert_to_grid(coordinates)
        end = self.convert_to_grid(combine_lists(combine_lists(coordinates, dimensions, '+'), (1, 1), '-'))
        all_coordinates = []

        for x in range(start[0], end[0] + 1):
            for y in range(start[1], end[1] + 1):
                all_coordinates.append((x, y))

        return all_coordinates

    def get_color(self, coordinates):
        try:
            return tuple(self.level_maps[self.difficulty][self.level - 1].get_at(coordinates))

        except:
            return 0, 0, 0, 0

    def update_level(self):
        self.analyze_map()
        self.player_default_coordinates = [
            find_center(self.doors[0].dimensions, player.dimensions,
                        self.doors[0].coordinates)[0],
            self.doors[0].coordinates[1] +
            self.doors[0].dimensions[1] -
            player.dimensions[1]
        ]
        player.default_coordinates = list(self.player_default_coordinates)
        player.total_reset()
        player.fake_coordinates = player.default_fake_coordinates
        player.update_grid_coordinates()
        player.block = self.blocks[
            self.convert_to_grid((player.coordinates[0], player.coordinates[1] + player.dimensions[1]))]

    def surrounding_blocks(self, initial_coordinates, block_types, blocks, return_coordinates=False):
        result = []
        for direction in self.directions:
            grid_coordinates = tuple(combine_lists(initial_coordinates, direction, '+'))
            if grid_coordinates in blocks and blocks[grid_coordinates] in block_types:
                if return_coordinates:
                    result.append(grid_coordinates)
                else:
                    result.append(direction)
        return result

    def rotate_sprites(self, sprite_type, direction):
        rotation = None
        if direction == (-1, 0):
            rotation = 90
        elif direction == (0, 1):
            rotation = 180
        elif direction == (1, 0):
            rotation = 270

        if rotation:
            sprites = []
            for sprite in self.sprites[sprite_type]:
                sprites.append(pygame.transform.rotate(sprite, rotation))

        else:
            sprites = self.sprites[sprite_type]
        return sprites

    def add_offset(self, coordinates, block_type, direction):
        offset = self.block_offsets[block_type]
        if direction[0]:
            offset = offset[::-1]
        return tuple(combine_lists(coordinates, offset, '+'))

    def exception(self, exception, coordinates):
        raise Exception(exception + ' at {0}'.format((coordinates[0] + (self.level - 1) * 25, coordinates[1])))

    def get_block_info(self, block_type, info):
        if self.level in self.block_info[self.difficulty] and block_type in self.block_info[self.difficulty][
            self.level] and info in \
                self.block_info[self.difficulty][self.level][block_type]:
            return self.block_info[self.difficulty][self.level][block_type][info]
        return self.default_block_info[block_type][info]

    def analyze_map(self):
        blocks = {}
        self.level += 1
        entrance = None
        self.count = 0

        for x in range(self.level_maps[self.difficulty][self.level - 1].get_width()):
            for y in range(self.level_maps[self.difficulty][self.level - 1].get_height()):
                color = self.get_color((x, y))
                if color[3] == 255:
                    formatted_color = (color[0], color[1])

                    if formatted_color in self.block_color_values:
                        block_type = self.block_color_values[formatted_color]
                        if block_type == 'entrance':
                            entrance = (x, y)
                        blocks[(x, y)] = block_type

                    elif formatted_color != (0, 0):
                        raise Exception(
                            "Unidentified block_color {0} at {1}".format(color, (x + (self.level - 1) * 25, y)))

        if not entrance:
            raise Exception("Missing door")

        self.backgrounds = {}
        active_coordinates = (entrance,)

        while active_coordinates:
            all_new_coordinates = []
            for coordinates in active_coordinates:
                for direction in self.directions:
                    new_coordinates = tuple(combine_lists(coordinates, direction, '+'))
                    if new_coordinates not in self.backgrounds and (
                                    new_coordinates not in blocks or blocks[new_coordinates] != 'block'):
                        self.backgrounds[new_coordinates] = Block(self.sprites['background'],
                                                                  self.convert_from_grid(
                                                                      new_coordinates), 'background')
                        if new_coordinates in blocks and blocks[new_coordinates] in self.block_types['solid']:
                            continue
                        all_new_coordinates.append(new_coordinates)
            active_coordinates = tuple(all_new_coordinates)

        self.blocks = {}
        self.cannons = []
        self.lasers = []
        platforms = []
        self.rails = []
        self.gate_heads = []
        self.alternating_blocks = []
        self.platform_tunnel_entrances = []
        self.doors = [None, None]
        gate_switch = None
        gate_head = None

        for initial_coordinates in blocks:
            block_type = blocks[initial_coordinates]
            if block_type in self.block_types['multi_block']:
                top_corner = True
                for direction in self.surrounding_blocks(initial_coordinates, (block_type,), blocks):
                    if direction in ((-1, 0), (0, -1)):
                        top_corner = False
                        break
                if not top_corner:
                    continue
            sprites = None
            coordinates = self.convert_from_grid(initial_coordinates)

            if block_type in self.block_types['directional']:
                direction = self.surrounding_blocks(initial_coordinates, ('platform',), blocks)
                if not direction:
                    for direction_2 in self.directions:
                        color = self.get_color(tuple(combine_lists(initial_coordinates, direction_2, '+')))
                        if color[2] == 205 and color[3] == 255:
                            direction = (direction_2,)
                            break
                    if not direction:
                        direction = [(0, 1)]

                direction = list(direction[0])
                direction_index = opposite(direction.index(0))
                direction[direction_index] *= -1
                direction = tuple(direction)

                if block_type in ('platform_tunnel_entrance', 'platform_tunnel_exit'):
                    sprite_type = 'platform_tunnel'
                else:
                    sprite_type = block_type

                sprites = self.rotate_sprites(sprite_type, direction)

                if block_type in self.block_offsets:
                    if block_type != 'spikes' or direction in ((-1, 0), (0, -1)):
                        coordinates = self.add_offset(coordinates, block_type, direction)

                if block_type in ('laser', 'cannon', 'gate_head'):
                    if block_type in ('laser', 'gate_head'):
                        entity_sprite = self.rotate_sprites(block_type + '_entity', direction)[0]
                        if block_type == 'laser':
                            block = EntityBlock(sprites, coordinates, 'laser', entity_sprite, direction,
                                                direction_index)
                            self.lasers.append(block)
                            block.end = None
                            block.active = 0
                            block.active_duration = self.get_block_info('laser', 'active_duration')
                            block.inactive_duration = self.get_block_info('laser', 'inactive_duration')
                            block.frequency = block.active_duration + block.inactive_duration
                            block.sprite_speed = int(block.inactive_duration / len(block.current_sprites()))

                        else:
                            block = GateHead(sprites, coordinates, 'gate_head', entity_sprite, direction,
                                             direction_index,
                                             self.get_block_info('gate_head', 'speed'))
                            self.gate_heads.append(block)

                    else:
                        block = Cannon(sprites, coordinates, self.sprites['cannon_entity'][0], direction,
                                       direction_index,
                                       self.get_block_info('cannon', 'entity_frequency'),
                                       self.get_block_info('cannon', 'entity_speed'))
                        self.cannons.append(block)

                    active_coordinates = list(initial_coordinates)

                    while True:
                        if tuple(active_coordinates) in blocks and blocks[
                            tuple(active_coordinates)
                        ] in self.block_types['solid'] and tuple(active_coordinates) != initial_coordinates:
                            if block_type == 'cannon':
                                block.last_coordinates = tuple(active_coordinates)
                            break

                        if block_type in ('laser', 'gate_head'):
                            # noinspection PyTypeChecker
                            entity = Thing(block.entity_sprite, self.add_offset(
                                self.convert_from_grid(tuple(active_coordinates)), block_type + '_entity',
                                block.direction))
                            if block_type == 'laser':
                                entity.all_grid_coordinates = tuple(active_coordinates)
                            else:
                                entity.all_grid_coordinates = (tuple(active_coordinates),)
                                entity.gate_head = block

                            block.entities.append(entity)
                        active_coordinates[block.direction_index] += block.direction[block.direction_index]

                else:
                    block = DirectionBlock(sprites, coordinates, block_type, direction, direction_index)

                    if block_type in ('fire', 'lava'):
                        block.transforming = True

                    elif block_type == 'platform_tunnel_entrance':
                        self.platform_tunnel_entrances.append(block)

                if block_type == 'gate_switch':
                    gate_switch = block

                elif block_type == 'gate_head':
                    gate_head = block

            elif block_type == 'platform':
                platforms.append(Block(self.sprites['platform'], coordinates, 'platform'))
                continue

            else:
                if block_type in ('block', 'alternating_block'):
                    sprites = [sprite.copy() for sprite in self.sprites['block']]
                    other_sprite = None
                    if block_type == 'alternating_block':
                        other_sprite = self.sprites['alternating_block'][0].copy()
                        sprites.append(other_sprite)
                    far = self.block_size - game.scale_factor
                    directions = []
                    corner_directions = []

                    for direction in self.directions:
                        grid_coordinates = tuple(combine_lists(initial_coordinates, direction, '+'))
                        if (grid_coordinates in blocks and (blocks[
                                                                grid_coordinates] not in self.block_types['solid'] or
                                                                    blocks[
                                                                        grid_coordinates] in self.block_types[
                                                                    'partial_solid'])) or (
                                        grid_coordinates not in blocks and grid_coordinates in self.backgrounds):
                            directions.append(direction)

                            if grid_coordinates not in blocks or blocks[grid_coordinates] not in self.block_types[
                                'no_corner']:
                                corner_directions.append(direction)

                            rect_coordinates = [0, 0]
                            rect_size = [0, 0]
                            direction_index = opposite(direction.index(0))

                            if direction[direction_index] == 1:
                                rect_coordinates[direction_index] = far

                            rect_size[direction_index] = game.scale_factor
                            rect_size[opposite(direction_index)] = self.block_size

                            for sprite in sprites:
                                if sprite == other_sprite:
                                    color = self.get_block_info('alternating_block', 'border_color')
                                else:
                                    color = (0, 0, 0, 255)
                                pygame.draw.rect(sprite, color, (rect_coordinates, rect_size))

                    corner_coordinates = []
                    if (-1, 0) in corner_directions:
                        if (0, -1) in corner_directions:
                            corner_coordinates.append((0, 0))
                        if (0, 1) in corner_directions:
                            corner_coordinates.append((0, far))
                    if (1, 0) in corner_directions:
                        if (0, -1) in corner_directions:
                            corner_coordinates.append((far, 0))
                        if (0, 1) in corner_directions:
                            corner_coordinates.append((far, far))

                    for corner in corner_coordinates:
                        for sprite in sprites:
                            pygame.draw.rect(sprite, self.background_color,
                                             (corner, (game.scale_factor, game.scale_factor)))

                if block_type in self.block_offsets:
                    coordinates = self.add_offset(coordinates, block_type, (0, 1))

                if not sprites:
                    sprites = self.sprites[block_type]

                block = Block(sprites, coordinates, block_type)

                if block_type == 'alternating_block':
                    block.active = 1
                    block.frequency = self.get_block_info('alternating_block', 'frequency')
                    self.alternating_blocks.append(block)

                elif block_type == 'entrance':
                    self.doors[0] = block
                elif block_type == 'exit':
                    self.doors[1] = block

            for grid_coordinates in block.all_grid_coordinates:
                self.blocks[grid_coordinates] = block

            if block_type in self.block_types['delay']:
                color = self.get_color(initial_coordinates)
                if color[2] != 0 and color[2] < 100:
                    block.delay = color[2]
                    if block_type == 'alternating_block' and block.delay >= self.get_block_info('alternating_block',
                                                                                                'frequency'):
                        block.active = opposite(block.active)
                        block.delay -= self.get_block_info('alternating_block', 'frequency')
                else:
                    block.delay = 0

        if gate_switch:
            gate_switch.gate_head = gate_head

        self.platforms = []

        for l in (platforms, self.platform_tunnel_entrances):
            for block in l:
                repeating = True
                other_coordinates = []
                if block.kind == 'platform':
                    start = block.all_grid_coordinates[0]
                else:
                    start = list(block.all_grid_coordinates[0])
                    start[opposite(block.direction_index)] += 1
                    other_coordinates.append(tuple(start))
                    start[block.direction_index] -= 1 * block.direction[block.direction_index]
                    start = tuple(start)
                rails = [start]
                current_coordinates = start

                while True:
                    new_coordinates = []
                    directions = []
                    for direction in self.directions:
                        grid_coordinates = tuple(combine_lists(current_coordinates, direction, '+'))
                        if block.kind == 'platform_tunnel_entrance' and len(other_coordinates) == 1 and \
                                        self.block_type(grid_coordinates) == 'platform_tunnel_exit':
                            end_block = self.blocks[grid_coordinates]
                            end = list(end_block.all_grid_coordinates[0])
                            end[opposite(end_block.direction_index)] += 1
                            other_coordinates.append(tuple(end))
                            end[end_block.direction_index] -= 1 * end_block.direction[end_block.direction_index]
                            other_coordinates.append(tuple(end))

                        color = self.get_color(grid_coordinates)
                        if (color[2] == 255 and color[
                            3] == 255) or grid_coordinates == start or grid_coordinates in other_coordinates or (
                                        grid_coordinates in blocks and blocks[grid_coordinates] == 'platform'):
                            directions.append(direction)
                            if grid_coordinates not in rails:
                                new_coordinates = grid_coordinates

                    if len(directions) == 1:
                        sprites = self.rotate_sprites('end_rail', directions[0])
                        if not new_coordinates:
                            repeating = False

                    elif len(directions) == 2:
                        if ((1, 0) in directions and (-1, 0) in directions) or (
                                        (0, 1) in directions and (0, -1) in directions):
                            if (0, 1) in directions:
                                sprites = []
                                for sprite in self.sprites['straight_rail']:
                                    sprites.append(pygame.transform.rotate(sprite, 90))
                            else:
                                sprites = self.sprites['straight_rail']

                        else:
                            rotation = None
                            if (0, 1) in directions:
                                if (-1, 0) in directions:
                                    rotation = 90
                                else:
                                    rotation = 180
                            elif (1, 0) in directions and (0, -1) in directions:
                                rotation = 270
                            if rotation:
                                sprites = []
                                for sprite in self.sprites['turn_rail']:
                                    sprites.append(pygame.transform.rotate(sprite, rotation))
                            else:
                                sprites = self.sprites['turn_rail']

                    elif len(directions) == 0:
                        self.exception('Single rail', current_coordinates)

                    else:
                        self.exception('Surrounded rail', current_coordinates)

                    self.rails.append(Thing(sprites, self.convert_from_grid(current_coordinates)))
                    current_coordinates = new_coordinates
                    if not new_coordinates:
                        break
                    rails.append(new_coordinates)

                if block.kind == 'platform':
                    if repeating:
                        direction = self.directions[self.get_color(block.all_grid_coordinates[0])[2]]
                    else:
                        direction = None
                    platform = Platform(self.sprites['platform'], rails, self.get_block_info('platform', 'speed'),
                                        repeating, direction)
                    self.platforms.append(platform)
                    for grid_coordinates in self.surrounding_blocks(platform.all_grid_coordinates[0],
                                                                    self.block_types['directional'], blocks, True):
                        platform.blocks.append(self.blocks[grid_coordinates])
                        del self.blocks[grid_coordinates]

                else:
                    block.rails = rails
                    block.frequency = self.get_block_info('platform_tunnel', 'frequency')
                    block.platforms = []


class Thing:
    def __init__(self, sprites, coordinates=(0, 0)):
        self.sprites = sprites
        self.coordinates = list(coordinates)
        self.reset()
        self.dimensions = self.current_sprite().get_size()

    def update_sprites(self, speed=4, reset=True):
        self.sprite_count += 1
        if self.sprite_count == speed:
            self.sprite_count = 0
            if self.sprite_index == len(self.current_sprites()) - 1:
                if reset:
                    self.sprite_index = 0
                return 'completed'
            self.sprite_index += 1

    def current_sprites(self):
        return make_tuple(self.sprites)

    def current_sprite(self):
        return self.current_sprites()[self.sprite_index]

    def reset(self):
        self.sprite_count = 0
        self.sprite_index = 0


class Block(Thing):
    def __init__(self, sprites, coordinates, kind):
        super().__init__(sprites, coordinates=coordinates)
        self.all_grid_coordinates = background.find_all_grid_coordinates(self.coordinates, self.dimensions)
        self.kind = kind
        self.transforming = False


class DirectionBlock(Block):
    def __init__(self, sprites, coordinates, kind, direction, direction_index):
        super().__init__(sprites, coordinates, kind)
        self.direction = direction
        self.direction_index = direction_index


class EntityBlock(DirectionBlock):
    def __init__(self, sprites, coordinates, kind, entity_sprite, direction, direction_index):
        super().__init__(sprites, coordinates, kind, direction, direction_index)
        self.entity_sprite = entity_sprite
        self.direction = direction
        self.direction_index = opposite(self.direction.index(0))
        self.entity_dimensions = self.entity_sprite.get_size()
        self.entities = []


class GateHead(EntityBlock):
    def __init__(self, sprites, coordinates, kind, entity_sprite, direction, direction_index, speed):
        super().__init__(sprites, coordinates, kind, entity_sprite, direction, direction_index)
        self.speed = speed * direction[self.direction_index] * game.scale_factor * -1
        self.retracting = False
        self.final_coordinate = self.coordinates[self.direction_index] - background.block_size * self.direction[
            self.direction_index]

    def retract(self):
        for entity in self.entities:
            entity.coordinates[self.direction_index] += self.speed
            entity.all_grid_coordinates = background.find_all_grid_coordinates(entity.coordinates, entity.dimensions)
            if entity.coordinates[self.direction_index] == self.final_coordinate:
                self.entities.remove(entity)

        if len(self.entities) == 0:
            self.retracting = False


class Cannon(EntityBlock):
    def __init__(self, sprites, coordinates, entity_sprite, direction, direction_index, entity_frequency,
                 entity_speed):
        super().__init__(sprites, coordinates, 'cannon', entity_sprite, direction, direction_index)
        self.entity_speed = entity_speed * self.direction[self.direction_index] * game.scale_factor
        self.update_initial_coordinates()
        self.entity_frequency = entity_frequency

    def update_initial_coordinates(self):
        if self.direction[self.direction_index] == 1:
            entity_x = self.coordinates[self.direction_index]
        else:
            entity_x = self.coordinates[self.direction_index] + self.dimensions[self.direction_index] - \
                       self.entity_dimensions[self.direction_index]
        self.entity_initial_coordinates = find_center(self.dimensions, self.entity_dimensions, self.coordinates)
        self.entity_initial_coordinates[self.direction_index] = entity_x


class Platform(Block):
    def __init__(self, sprites, rails, speed, repeating, direction):
        self.speed = speed * game.scale_factor
        self.repeating = repeating
        self.rail_direction = -1
        if self.repeating:
            if tuple(combine_lists(rails[1], rails[0], '-')) == direction:
                self.rail_direction = 1
                self.end_index = 1
            else:
                self.end_index = 0
        self.rail_coordinates = [background.convert_from_grid(rail) for rail in rails]
        self.current_rail_number = 0
        self.blocks = [self]
        self.ends = (0, len(self.rail_coordinates) - 1)
        super().__init__(sprites, self.rail_coordinates[self.current_rail_number], 'moving_platform')

    def move(self):
        if self.coordinates in self.rail_coordinates:
            if (not self.repeating and self.current_rail_number in self.ends) or (
                        self.repeating and self.current_rail_number == self.ends[self.end_index]):
                if self.repeating:
                    self.current_rail_number = self.ends[opposite(self.end_index)] - self.rail_direction
                else:
                    self.rail_direction *= -1
            self.current_rail_number += self.rail_direction
            self.direction = [int(polarity(self.rail_coordinates[self.current_rail_number][i] - self.coordinates[i]))
                              for i
                              in range(2)]
            self.direction_index = opposite(self.direction.index(0))

        for block in self.blocks:
            block.coordinates[self.direction_index] += self.speed * self.direction[self.direction_index]
            block.all_grid_coordinates = background.find_all_grid_coordinates(block.coordinates, block.dimensions)


class Mob(Thing):
    def __init__(self, sprites, conditions_info, coordinates=list((0, 0)),
                 default_sprite_type='stagnant', visible_direction=True, gravity=(0, 1)):
        self.conditions_info = conditions_info
        self.default_coordinates = coordinates
        self.default_sprite_type = default_sprite_type
        self.visible_direction = visible_direction
        self.default_gravity = gravity
        self.current_sprite_type = self.default_sprite_type
        super().__init__(sprites, coordinates)
        self.total_reset()
        self.velocity = [0, 0]
        self.direction = 1

    def current_sprites(self):
        if self.visible_direction:
            return make_tuple(self.sprites[self.direction][self.current_sprite_type])
        else:
            return make_tuple(self.sprites[self.current_sprite_type])

    def reset(self, conditions=None):
        super().reset()
        if conditions:
            for condition in conditions:
                self.conditions[condition] = False

    def total_reset(self):
        self.reset()
        self.current_sprite_type = self.default_sprite_type
        self.velocity = [0, 0]
        self.conditions = {name: self.conditions_info[name]['active'] for name in self.conditions_info}
        self.coordinates = list(self.default_coordinates)
        self.gravity = list(self.default_gravity)
        self.gravity_index = opposite(self.gravity.index(0))

    def update_grid_coordinates(self, velocity=True):
        if velocity:
            coordinates = combine_lists(self.coordinates, self.velocity, '+')
        else:
            coordinates = self.coordinates
        self.all_grid_coordinates = background.find_all_grid_coordinates(coordinates, self.dimensions)


class Player(Mob):
    def __init__(self, sprites, conditions_info):
        super().__init__(sprites, conditions_info, visible_direction=False)
        self.default_fake_coordinates = find_center(game.dimensions, self.dimensions)
        self.fake_coordinates = [0, 0]
        self.gravity_switch = None

    def generate_display_coordinates(self, coordinates):
        return combine_lists(
            combine_lists(coordinates, self.fake_coordinates, '+'),
            self.coordinates, '-')

    def die(self):
        self.current_sprite_type = 'dying'
        background.condition = 'reset'
        if self.gravity_switch:
            self.gravity_switch.sprite_index = 0
        self.gravity_switch = None
        self.block = None

    def process_collision(self, thing):
        for i in range(2):
            velocity = [0, 0]
            velocity[i] = self.velocity[i]
            if collision(combine_lists(self.coordinates, velocity, '+'), self.dimensions,
                         thing.coordinates, thing.dimensions) is True:
                player.align_velocity(thing, i)

    def align_velocity(self, thing, i):
        if i == player.gravity_index:
            self.reset(('jumping', 'falling'))
            if (self.gravity[self.gravity_index] == 1 and self.velocity[i] > 0) or (
                            self.gravity[self.gravity_index] == -1 and self.velocity[i] < 0):
                self.block = thing

        if self.velocity[i] > 0:
            self.velocity[i] = thing.coordinates[i] - (
                self.coordinates[i] + self.dimensions[i])
        elif self.velocity[i] < 0:
            self.velocity[i] = (thing.coordinates[i] + thing.dimensions[i]) - \
                               self.coordinates[i]
        else:
            self.die()
            self.velocity = [0, 0]

    def set_gravity(self, thing):
        if self.gravity_switch:
            self.gravity_switch.sprite_index = 0
        self.gravity_switch = thing
        thing.sprite_index = 1
        self.gravity = list(thing.direction)
        self.gravity_index = thing.direction_index
        self.gravity[self.gravity_index] *= -1


# sheet input

game = Game(30, (1200, 900), 3, 10, 14)
block_sheet = SpriteSheet('block_sheet_V{0}.png'.format(game.version))
misc_sheet = SpriteSheet('misc_sheet_V{0}.png'.format(game.version))
map_sheet = SpriteSheet('map_sheet_V{0}.png'.format(game.version))
# background_map_sheet = SpriteSheet('Ivan\'s_levels.png', 1)
background_maps = map_sheet.get_sprites(y_constant=25, x_constant=(25, 30), scale=False)
background_maps_hard = map_sheet.get_sprites(y_constant=25, x_constant=(25, 30), scale=False)

# sprite input

single_block_names = (
    'block', 'background', 'cannon', 'platform', 'straight_rail', 'turn_rail', 'end_rail', 'alternating_block')
single_blocks = block_sheet.get_sprites(block_number=len(single_block_names))
background_sprites = {name: [single_blocks[single_block_names.index(name)]]
                      for name in single_block_names}

misc_names = ('gate_head_entity', 'laser_entity', 'cannon_entity', 'spikes')
misc = block_sheet.get_sprites(all_dimensions=((6, 10), (4, 10), (4, 4), (10, 3)))
misc_sprites = {name: [misc[misc_names.index(name)]] for name in misc_names}
background_sprites.update(misc_sprites)

background_doors = block_sheet.get_sprites(y_constant=12, x_constant=(16, 10))
background_sprites['door_background'] = [background_doors[-1]]
del background_doors[-1]
background_sprites['entrance'] = background_doors
background_sprites['exit'] = background_doors[::-1]
background_sprites['checkpoint'] = block_sheet.get_sprites(y_constant=10, x_constant=(8, 3))
background_sprites['gate_head'] = block_sheet.get_sprites(block_number=2)
background_sprites['gravity_switch'] = block_sheet.get_sprites(block_number=2)
background_sprites['laser'] = block_sheet.get_sprites(block_number=4)  # y_constant=10, x_constant=(8, 4))
player_sprites = block_sheet.get_sprites(block_number=8)
background_sprites['gate_switch'] = block_sheet.get_sprites(y_constant=10, x_constant=(9, 4))
background_sprites['lava'] = block_sheet.get_sprites(block_number=9)
background_sprites['fire'] = block_sheet.get_sprites(block_number=4)
background_sprites['platform_tunnel'] = block_sheet.get_sprites(all_dimensions=((30, 10),))

# background_sprites['number'] = misc_sheet.get_sprites(y_constant=40, x_constant=(20, 10))
background_sprites['number'] = misc_sheet.get_sprites(y_constant=7, x_constant=(7, 10))
background_sprites['title'] = misc_sheet.get_sprites(dimensions=(81, 31), scale=4)
player_option_sprites = misc_sheet.get_sprites(y_constant=7, x_constant=(61, 2))
background_sprites['1 player'] = [player_option_sprites[0]]
background_sprites['2 player'] = [player_option_sprites[1]]
background_sprites['selecter'] = misc_sheet.get_sprites(dimensions=(4, 5), scale=4)

for sprites in background_sprites:
    if type(background_sprites[sprites]) != list:
        raise Exception('Non-list sprite')

# sprite processing

player_sprites = {
    'stagnant': player_sprites[0],
    'dying': get_list(player_sprites, range(1, 8))
}

player = Player(player_sprites, {
    'moving': {'active': False, 'velocity': int(1.7 * game.scale_factor)},
    'jumping': {'active': False, 'velocity': 30 * game.scale_factor, 'quadratic': None, 'speed': .5 * game.speed},
    'falling': {'active': False, 'velocity': 60 * game.scale_factor, 'quadratic': None, 'speed': .7 * game.speed},
})


def convert_to_color(number, base):
    b = int(number / base)
    number -= (b * base)
    factor = 255 / (base - 1)
    return int(number * factor), int(b * factor)


block_names = (
    'block', 'lava', 'fire', 'checkpoint', 'cannon', 'entrance', 'exit', 'laser', 'gate_head', 'gate_switch',
    'gravity_switch', 'platform', 'spikes', 'platform_tunnel_entrance',
    'platform_tunnel_exit', 'alternating_block'
)

background_color_values = {color: name for color, name in zip(
    [convert_to_color(number, 16) for number in
     range(1, len(block_names) + 1)
     ], block_names)}

background_offsets = {
    'checkpoint': (1, 0),
    'laser_entity': (3, 0),
    'entrance': (2, 8),
    'exit': (2, 8),
    'gate_head_entity': (2, 0),
    'spikes': (0, 7)
}

default_background_block_info = {
    'platform': {'speed': 1},
    'cannon': {'entity_speed': 2, 'entity_frequency': 1.6 * game.speed},
    'laser': {'inactive_duration': 2 * game.speed, 'active_duration': 1 * game.speed},
    'gate_head': {'speed': 1},
    'alternating_block': {'frequency': 2 * game.speed, 'border_color': (77, 64, 64, 255)},
    'platform_tunnel': {'frequency': 1.5 * game.speed, 'platform_offset': 3}
}

background_block_info = {
    'easy': {
    },
    'hard': {
    }
}

background_block_types = {
    'solid': ('block', 'cannon', 'gate_head', 'gravity_switch', 'laser',
              'alternating_block', 'platform_tunnel_entrance', 'platform_tunnel_exit', 'moving_platform'),
    'partial_solid': (),
    'foreground': ('fire', 'lava', 'spikes'),
    'directional': (
        'lava', 'fire', 'checkpoint', 'cannon', 'laser', 'gate_head', 'gate_switch', 'gravity_switch', 'spikes',
        'platform_tunnel_entrance', 'platform_tunnel_exit'),
    'dangerous': ('lava', 'fire', 'spikes'),
    'multi_block': ('exit', 'entrance', 'platform_tunnel_entrance', 'platform_tunnel_exit'),
    'no_corner': ('spikes', 'fire', 'lava'),
    'delay': ('alternating_block', 'laser', 'cannon', 'platform_tunnel_entrance')
}

background_maps = {'easy': background_maps, 'hard': background_maps_hard}

background_switch_pairs = None

background = Background(background_sprites, background_maps,
                        background_color_values, background_offsets,
                        default_background_block_info, background_block_info,
                        background_block_types,
                        750 + player.fake_coordinates[1], 1.5 * game.speed,
                        ('title', '1 player', '2 player'), {'title': 300, '1 player': 50, '2 player': 25},
                        ('1 player', '2 player'), 3, 1.5 * game.speed,
                        background_switch_pairs, 'easy',
                        [10, 10], game.scale_factor,
                        )

# developer tools
#
background.level = 30
background.level -= 1
background.condition = None
background.update_level()
background.doors[0].sprite_index = len(background.doors[0].current_sprites()) - 1
background.phase = 'levels'

# for block in background.blocks:
#         if background.blocks[block].kind == 'checkpoint':
#             player.coordinates = background.convert_from_grid(background.blocks[block].all_grid_coordinates[0])
#        break

# player.coordinates = background.convert_from_grid(background.doors[1].all_grid_coordinates[0])

while True:
    print(player.coordinates)
    events = pygame.event.get()
    # print(events)
    for event in events:
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_SPACE):
            pygame.quit()
            quit()
        elif event.type == KEYDOWN:
            if event.key == K_z:
                if game.real_speed == 30:
                    game.real_speed = 1
                else:
                    game.real_speed = 30

            elif event.key == K_x:
                # player.coordinates = background.convert_from_grid((5, 14))
                if game.real_speed == 30:
                    game.real_speed = 100
                else:
                    game.real_speed = 30

    if background.level_transition_phase is not None:
        result = background.level_transition_quadratics[background.level_transition_phase].execute()

        if type(result) == tuple:
            if background.level_transition_phase == 0:
                background.update_level()
                background.phase = 'levels'
                player.fake_coordinates[1] = background.level_transition_coordinates[1][0]
                background.level_transition_phase = 1
            else:
                background.condition = 'doors_opening'
                background.level_transition_phase = None
            continue

        else:
            player.fake_coordinates[1] += result

    if background.phase == 'menu':
        if background.condition == 'title':
            game.display.fill(background.block_color)
            game.display.blit(background.sprites['title'][0], background.title_coordinates)
            pygame.display.update()
            pygame.time.wait(1000)
            background.condition = 'title_transition'
            continue

        elif background.condition == 'title_transition':
            background.title_coordinates[1] -= background.title_transition_speed
            if round(background.title_coordinates[1]) <= background.menu_coordinates['title'][1]:
                background.condition = 'menu'
                continue

        elif background.condition == 'menu':
            for event in events:
                if event.type == KEYDOWN:
                    if event.key == K_DOWN:
                        if background.selecter_position_index != len(background.menu_selectable_options) - 1:
                            background.selecter_position_index += 1
                            background.update_selecter_coordinates()

                    elif event.key == K_UP:
                        if background.selecter_position_index != 0:
                            background.selecter_position_index -= 1
                            background.update_selecter_coordinates()

                    elif event.key == K_RETURN:
                        if background.menu_selectable_options[background.selecter_position_index] == '1 player':
                            background.condition = None
                            background.initiate_level_transition()
                            break
            if background.level_transition_phase:
                continue

        game.display.fill(background.block_color)

        if background.condition != 'title_transition':
            for option in background.menu_coordinates:
                game.display.blit(background.sprites[option][0], (
                    player.generate_display_coordinates(background.menu_coordinates[option])))

            game.display.blit(background.sprites['selecter'][0],
                              player.generate_display_coordinates(background.selecter_coordinates))

        else:
            game.display.blit(background.sprites['title'][0], background.title_coordinates)

    else:
        if background.condition:
            if background.condition == 'reset':
                if player.update_sprites(3) == 'completed':
                    player.total_reset()
                    background.condition = None
                    continue

            elif background.condition == 'doors_closing':
                if background.doors[1].update_sprites(4, False) == 'completed':
                    background.initiate_level_transition()
                    background.condition = 'transitioning'
                    continue

            elif background.condition == 'doors_opening':
                if background.doors[0].update_sprites(4, False) == 'completed':
                    background.condition = None
                    continue

        for platform_tunnel_entrance in background.platform_tunnel_entrances:
            for platform in platform_tunnel_entrance.platforms:
                if platform.coordinates in platform.rail_coordinates and platform.current_rail_number in platform.ends:
                    platform_tunnel_entrance.platforms.remove(platform)
                    background.platforms.remove(platform)
                    break

            if (game.count - platform_tunnel_entrance.delay) % platform_tunnel_entrance.frequency == 0:
                platform = Platform(background.sprites['platform'], platform_tunnel_entrance.rails,
                                    background.get_block_info('platform', 'speed'), False, None)
                platform_tunnel_entrance.platforms.append(platform)
                background.platforms.append(platform)

        for platform in background.platforms:
            platform.move()

        for gate_head in background.gate_heads:
            if gate_head.retracting:
                gate_head.retract()

        for alternating_block in background.alternating_blocks:
            if (game.count - alternating_block.delay) % alternating_block.frequency == 0:
                alternating_block.sprite_index = alternating_block.active
                alternating_block.active = opposite(alternating_block.active)

                if alternating_block.active == 0:
                    del background.blocks[alternating_block.all_grid_coordinates[0]]
                    if player.block == alternating_block:
                        player.block = None

                else:
                    background.blocks[alternating_block.all_grid_coordinates[0]] = alternating_block

        if not background.condition:
            background.count += 1
            player.velocity = [0, 0]

            if player.block:
                dimensions = list(player.dimensions)
                dimensions[player.gravity_index] = 1

                coordinates_1 = list(player.coordinates)
                if player.gravity[player.gravity_index] == -1:
                    coordinates_1[player.gravity_index] -= 1
                else:
                    coordinates_1[player.gravity_index] += player.dimensions[player.gravity_index]
                coordinates_2 = list(coordinates_1)
                coordinates_2[player.gravity_index] -= player.gravity[player.gravity_index]
                coordinates = (coordinates_1, coordinates_2)

                velocity = [0, 0]
                for platform in background.platforms:
                    if player.block in platform.blocks:
                        velocity[platform.direction_index] = platform.speed * platform.direction[
                            platform.direction_index]
                        break

                if type(player.block) == Thing and player.block.gate_head.retracting:
                    velocity[player.block.gate_head.direction_index] = player.block.gate_head.speed * \
                                                                       player.block.gate_head.direction[
                                                                           player.block.gate_head.direction_index] * -1

                coordinates = [combine_lists(coordinates[i], velocity, '+') for i in range(2)]

                if (
                                collision(coordinates[0], player.dimensions, player.block.coordinates,
                                          player.block.dimensions) is True and
                                collision(coordinates[1], dimensions, player.block.coordinates,
                                          player.block.dimensions) is not True
                ):
                    player.reset(('jumping', 'falling'))
                    if velocity != [0, 0]:
                        player.velocity = velocity

                else:
                    player.block = None

            keys = pygame.key.get_pressed()

            if keys[K_RIGHT] or keys[K_LEFT] or keys[K_d] or [K_a]:
                if not player.conditions['moving']:
                    player.conditions['moving'] = True

            if (keys[K_UP] or keys[K_w]) and player.block and not (
                        player.conditions['jumping'] or player.conditions['falling']):
                player.conditions['jumping'] = True
                player.block = None
                # noinspection PyTypeChecker
                player.conditions_info['jumping']['quadratic'] = Quadratic(-1, (0,
                                                                                player.conditions_info['jumping'][
                                                                                    'velocity']),
                                                                           player.conditions_info['jumping'][
                                                                               'speed'])

            if not (keys[K_RIGHT] and keys[K_d]) and player.direction == 1 or not (
                        keys[K_LEFT] and keys[K_a]) and player.direction == -1:
                player.direction = 0

            if keys[K_RIGHT] or keys[K_d]:
                player.direction = 1
            elif keys[K_LEFT] or keys[K_a]:
                player.direction = -1

            if player.gravity == [1, 0]:
                player.direction *= -1

            if player.direction == 0:
                player.reset(('moving',))
                player.direction = 1

            if not (player.block or player.conditions['falling'] or player.conditions['jumping']):
                player.conditions['falling'] = True
                # noinspection PyTypeChecker
                player.conditions_info['falling']['quadratic'] = Quadratic(1,
                                                                           (0,
                                                                            player.conditions_info['falling'][
                                                                                'velocity']),
                                                                           player.conditions_info['falling']['speed'])

            for condition in player.conditions:
                if player.conditions[condition]:
                    if condition == 'moving':
                        # noinspection PyTypeChecker
                        player.velocity[opposite(player.gravity_index)] += player.conditions_info['moving'][
                                                                               'velocity'] * player.direction

                    if condition == 'jumping':
                        # noinspection PyUnresolvedReferences
                        result = player.conditions_info['jumping']['quadratic'].execute()
                        if type(result) == tuple:
                            player.reset(('jumping',))
                        else:
                            player.velocity[player.gravity_index] -= result * player.gravity[player.gravity_index]

                    elif condition == 'falling':
                        result = player.conditions_info['falling']['quadratic'].execute()
                        player.velocity[player.gravity_index] += make_tuple(result)[0] * player.gravity[
                            player.gravity_index]

            for collision_type in ('platform', 'block', 'gate'):
                player.update_grid_coordinates()
                collided_object = None

                if collision_type == 'platform':
                    for platform in background.platforms:
                        for block in platform.blocks:
                            for grid_coordinates in block.all_grid_coordinates:
                                if grid_coordinates in player.all_grid_coordinates:
                                    if collision(combine_lists(player.coordinates, player.velocity, '+'),
                                                 player.dimensions,
                                                 block.coordinates, block.dimensions) is True:
                                        if block.kind in background.block_types['dangerous']:
                                            player.die()
                                            break

                                        elif block.kind in background.block_types['solid']:
                                            if collision(player.coordinates, player.dimensions, block.coordinates,
                                                         block.dimensions) is True:
                                                player.velocity[platform.direction_index] = 0
                                                player.block = block
                                                if platform.direction[platform.direction_index] == 1:
                                                    player.coordinates[platform.direction_index] = block.coordinates[
                                                                                                       platform.direction_index] + \
                                                                                                   block.dimensions[
                                                                                                       platform.direction_index]
                                                else:
                                                    player.coordinates[platform.direction_index] = block.coordinates[
                                                                                                       platform.direction_index] - \
                                                                                                   player.dimensions[
                                                                                                       platform.direction_index]

                                            else:
                                                collided_object = block
                                                player.process_collision(collided_object)

                elif collision_type == 'block':
                    for grid_coordinates in player.all_grid_coordinates:
                        if background.block_type(grid_coordinates) in background.block_types['solid']:
                            collided_object = background.blocks[grid_coordinates]
                            if collision(player.coordinates, player.dimensions, collided_object.coordinates,
                                         collided_object.dimensions) is True:
                                player.die()
                                break
                            player.process_collision(collided_object)
                            if collided_object.kind == 'gravity_switch' and player.gravity_switch != collided_object:
                                player.set_gravity(collided_object)
                                player.reset(('jumping', 'falling'))

                else:
                    for gate_head in background.gate_heads:
                        for entity in gate_head.entities:
                            for grid_coordinates in entity.all_grid_coordinates:
                                if grid_coordinates in player.all_grid_coordinates:
                                    if collision(combine_lists(player.coordinates, player.velocity, '+'),
                                                 player.dimensions,
                                                 entity.coordinates,
                                                 entity.dimensions) is True:
                                        collided_object = entity
                                        player.process_collision(collided_object)

                if collided_object and player.velocity != [0, 0] and player.current_sprite_type != 'dying':
                    if collision(combine_lists(player.coordinates, player.velocity, '+'), player.dimensions,
                                 collided_object.coordinates, collided_object.dimensions) is True:
                        player.align_velocity(collided_object, opposite(player.gravity_index))

            player.update_grid_coordinates()

            if not background.condition:
                for grid_coordinates in player.all_grid_coordinates:
                    block_type = background.block_type(grid_coordinates)

                    if block_type:
                        block = background.blocks[grid_coordinates]
                        if collision(combine_lists(player.coordinates, player.velocity, '+'), player.dimensions,
                                     block.coordinates, block.dimensions) is True:
                            if block_type == 'exit' and player.gravity == [0, 1]:
                                if collision(
                                        combine_lists(player.coordinates, player.velocity, '+'),
                                        player.dimensions,
                                        block.coordinates,
                                        block.dimensions,
                                        True) is True:
                                    player.coordinates = (
                                        find_center(
                                            block.dimensions,
                                            player.dimensions,
                                            c1=block.coordinates)[0],
                                        block.coordinates[1] +
                                        block.dimensions[1] -
                                        player.dimensions[1]
                                    )

                                    background.condition = 'doors_closing'
                                    break

                            elif block_type in background.block_types['dangerous']:
                                player.die()
                                break

                            elif block_type == 'checkpoint' and block.sprite_index == 0:
                                player.default_coordinates = background.convert_from_grid(grid_coordinates)
                                player.default_gravity = player.gravity
                                block.transforming = True

                            elif block_type == 'gate_switch' and block.gate_head.sprite_index == 0:
                                player.default_coordinates = background.convert_from_grid(grid_coordinates)
                                player.default_gravity = player.gravity
                                block.gate_head.retracting = True
                                block.gate_head.sprite_index += 1
                                block.transforming = True

            if background.condition != 'doors_closing':
                player.coordinates = combine_lists(player.velocity, player.coordinates, '+')

        for cannon in background.cannons:
            for entity in cannon.entities:
                entity.coordinates[cannon.direction_index] += cannon.entity_speed
                all_grid_coordinates = background.find_all_grid_coordinates(entity.coordinates, entity.dimensions)
                collided = False

                if cannon.last_coordinates in all_grid_coordinates:
                    cannon.entities.remove(entity)
                    break

                for grid_coordinates in all_grid_coordinates:
                    if not background.condition:
                        if grid_coordinates in player.all_grid_coordinates:
                            if collision(entity.coordinates, entity.dimensions, player.coordinates,
                                         player.dimensions) is True:
                                player.die()
                                collided = True

                    if not collided:
                        for gate_head in background.gate_heads:
                            for gate in gate_head.entities:
                                if grid_coordinates in gate.all_grid_coordinates:
                                    if collision(entity.coordinates, entity.dimensions,
                                                 gate.coordinates,
                                                 gate.dimensions) is True:
                                        collided = True
                                        break

                    if not collided:
                        for platform in background.platforms:
                            for block in platform.blocks:
                                if block.kind in background.block_types['solid'] and \
                                                grid_coordinates in block.all_grid_coordinates:
                                    if collision(entity.coordinates, entity.dimensions, block.coordinates,
                                                 block.dimensions) is True:
                                        collided = True
                                        break

                    if collided and grid_coordinates not in cannon.all_grid_coordinates:
                        break

                if collided:
                    cannon.entities.remove(entity)

            if (game.count - cannon.delay) % cannon.entity_frequency == 0:
                cannon.entities.append(Thing(cannon.entity_sprite, cannon.entity_initial_coordinates))

        for laser in background.lasers:
            cycle = (game.count - laser.delay) % laser.frequency
            if cycle == 0:
                laser.active = 0
                laser.reset()

            if cycle == laser.inactive_duration:
                laser.active = 1
                laser.sprite_index = len(laser.current_sprites()) - 1

            if laser.active:
                laser.end = None
                for entity in laser.entities:
                    if not background.condition:
                        if entity.all_grid_coordinates in player.all_grid_coordinates:
                            if collision(entity.coordinates, entity.dimensions,
                                         player.coordinates,
                                         player.dimensions) is True:
                                player.die()

                    for gate_head in background.gate_heads:
                        for gate in gate_head.entities:
                            if entity.all_grid_coordinates in gate.all_grid_coordinates:
                                if collision(entity.coordinates, entity.dimensions,
                                             gate.coordinates,
                                             gate.dimensions) is True:
                                    laser.end = (entity, gate)
                                    break

                    for platform in background.platforms:
                        for block in platform.blocks:
                            if block.kind in background.block_types['solid'] and \
                                            entity.all_grid_coordinates in block.all_grid_coordinates:
                                if collision(entity.coordinates, entity.dimensions,
                                             block.coordinates,
                                             block.dimensions) is True:
                                    laser.end = (entity, block)
                                    break
                    if laser.end:
                        break

            else:
                laser.update_sprites(laser.sprite_speed, False)

        game.display.fill(background.block_color)

        for grid_coordinates in background.backgrounds:
            game.blit(background.backgrounds[grid_coordinates])

        for door in background.doors:
            game.display.blit(background.sprites['door_background'][0],
                              player.generate_display_coordinates(door.coordinates))

        for rail in background.rails:
            game.blit(rail)

        for gate_head in background.gate_heads:
            for entity in gate_head.entities:
                game.blit(entity)

        for platform in background.platforms:
            for block in platform.blocks:
                game.blit(block)

        for grid_coordinates in background.blocks:
            block = background.blocks[grid_coordinates]
            if block.kind not in ('cannon', 'entrance', 'exit', 'laser', 'alternating_block') and block.kind not in \
                    background.block_types['foreground']:
                if block.transforming:
                    if block.kind in ('checkpoint', 'gate_switch'):
                        if block.update_sprites(5, reset=False):
                            block.transforming = False
                    else:
                        block.update_sprites()
                game.blit(block)

        for block in background.alternating_blocks:
            game.blit(block)

        if background.condition in ('doors_closing', 'doors_opening') or background.level_transition_phase is not None:
            game.display.blit(player.current_sprite(), player.fake_coordinates)
            player.display_after = False
        else:
            player.display_after = True

        for door in background.doors:
            # noinspection PyTypeChecker
            game.blit(door)

        for cannon in background.cannons:
            for entity in cannon.entities:
                game.blit(entity)
            game.blit(cannon)

        for laser in background.lasers:
            if laser.active:
                for entity in laser.entities:
                    if laser.end and entity == laser.end[0]:
                        area_dimensions = [0, 0]
                        area_dimensions[opposite(laser.direction_index)] = laser.end[0].dimensions[
                            opposite(laser.direction_index)]
                        blit_coordinates = list(laser.end[0].coordinates)

                        if laser.direction[laser.direction_index] == 1:
                            area_dimensions[laser.direction_index] = laser.end[1].coordinates[
                                                                         laser.direction_index] - \
                                                                     laser.end[0].coordinates[laser.direction_index]
                        else:
                            blit_coordinates[laser.direction_index] = laser.end[1].coordinates[
                                                                          laser.direction_index] + \
                                                                      laser.end[1].dimensions[laser.direction_index]
                            area_dimensions[laser.direction_index] = laser.end[0].coordinates[
                                                                         laser.direction_index] - (
                                                                         laser.end[1].coordinates[
                                                                             laser.direction_index] +
                                                                         laser.end[1].dimensions[
                                                                             laser.direction_index])

                        game.display.blit(laser.entity_sprite,
                                          player.generate_display_coordinates(blit_coordinates),
                                          ((0, 0), area_dimensions))
                        break

                    else:
                        game.blit(entity)
            game.blit(laser)

        if player.display_after:
            game.display.blit(player.current_sprite(), player.fake_coordinates)

        for grid_coordinates in background.blocks:
            block = background.blocks[grid_coordinates]
            if block.kind in background.block_types['foreground']:
                if block.transforming:
                    block.update_sprites()
                game.blit(block)

        level = str(background.level)
        for number, i in zip(level, reversed(range(len(level)))):
            game.display.blit(background.sprites['number'][int(number)],
                              (game.dimensions[0] - (i + 1) * (background.number_width + background.level_number_gap) -
                               background.level_number_margins[0], background.level_number_margins[1]))
        game.count += 1

    pygame.display.update()
    game.clock.tick(game.real_speed)
