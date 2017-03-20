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
    def __init__(self, filename, division_index):
        self.sheet = pygame.image.load(filename).convert_alpha()
        self.division_index = division_index
        self.farthest_y_coordinate = 0

    def get_image(self, coordinates, dimensions):
        # noinspection PyArgumentList
        image = pygame.Surface(dimensions, SRCALPHA).convert_alpha()
        image.blit(self.sheet, (0, 0), (coordinates, dimensions))
        return image

    def get_sprites(self, farthest_x_coordinate=0, farthest_y_coordinate=None, all_coordinates=None, y_constant=None,
                    x_constant=None, update=True,
                    scale=True):
        sprites = []
        if not farthest_y_coordinate:
            farthest_y_coordinate = self.farthest_y_coordinate
        if y_constant and x_constant:
            thing = range(x_constant[1])
        else:
            thing = all_coordinates
        for i in thing:
            coordinates_1 = [0, 0]
            coordinates_1[self.division_index] = farthest_y_coordinate
            coordinates_1[opposite(self.division_index)] = farthest_x_coordinate
            if x_constant:
                farthest_x_coordinate = (i + 1) * x_constant[0] - 1
            elif y_constant:
                farthest_x_coordinate = i
            else:
                farthest_x_coordinate = i[opposite(self.division_index)]
            coordinates_2 = [None, None]
            if x_constant or y_constant:
                if x_constant:
                    coordinates_2[opposite(self.division_index)] = farthest_x_coordinate
                else:
                    coordinates_2[opposite(self.division_index)] = i
                if y_constant:
                    coordinates_2[self.division_index] = farthest_y_coordinate + y_constant - 1
                else:
                    coordinates_2[self.division_index] = i
            else:
                coordinates_2 = i
            farthest_x_coordinate += 1
            dimensions = combine_lists((1, 1), combine_lists(coordinates_2, coordinates_1, '-'), '+')
            sprite = self.get_image(coordinates_1, dimensions)
            if scale:
                sprite = pygame.transform.scale(sprite, combine_lists(sprite.get_size(),
                                                                      (game.scale_factor, game.scale_factor), '*'))
            sprites.append(sprite)
        if update:
            if y_constant:
                self.farthest_y_coordinate += y_constant
            elif x_constant:
                self.farthest_y_coordinate = max(all_coordinates) + 1
            else:
                self.farthest_y_coordinate = max(
                    [coordinates[self.division_index] for coordinates in all_coordinates]) + 1
        return sprites


class Quadratic:
    def __init__(self, sign, y_range, speed):
        self.a = sign
        if (sign == 1 and y_range[0] > y_range[1]) or (sign == -1 and y_range[1] > y_range[0]):
            x_solution_index = 0
        else:
            x_solution_index = 1
        self.c = y_range[opposite(x_solution_index)]
        self.x_range = [self.get_x(y_range[i])[x_solution_index] for i in range(2)]
        self.x_change = (self.x_range[1] - self.x_range[0]) / (speed * game.speed)
        self.current_x = self.x_range[0]
        self.old_y = y_range[0]

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

        # return sorted(
        #     [sqrt( -1 * (4 * self.a * self.c) + (4 * self.a * y)) / (2 * self.a) for i in
        #      (1, -1)]
        # )


class Game:
    def __init__(self, speed, dimensions, scale_factor, version):
        self.speed = speed
        self.real_speed = self.speed
        self.dimensions = dimensions
        self.scale_factor = scale_factor
        self.version = version
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)
        self.movement_keys = (K_RIGHT, K_LEFT, K_d, K_a)
        self.font = pygame.font.SysFont("Calibri", 100)
        self.condition = 'title'
        self.entering_phase = 1
        self.entering_quadratic = None
        self.count = 0

    @staticmethod
    def exit():
        pygame.quit()
        quit()


class Background:
    def __init__(self, maps, block_sprites, block_color_values, block_offsets, level_times):
        self.maps = maps
        self.block_sprites = block_sprites
        self.block_color_values = block_color_values
        self.block_offsets = block_offsets
        for offset_name in self.block_offsets:
            if offset_name not in ('entrance_background', 'exit_background'):
                self.block_offsets[offset_name] = combine_lists(block_offsets[offset_name],
                                                                (game.scale_factor, game.scale_factor), '*')
        self.block_size = game.scale_factor * 10
        self.level = 0
        self.special_block_types = ('entrance', 'exit')
        self.color = self.block_sprites['block'].get_at((0, 0))
        self.sides = ((0, 1), (0, -1), (1, 0), (-1, 0))
        self.side_dict = {side: False for side in self.sides}
        self.level_times = level_times
        self.number_width = self.block_sprites['numbers'][0].get_width()
        self.reset_delay = 2

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
        return tuple(self.maps[self.level - 1].get_at(coordinates))

    def update_level(self):
        self.analyze_map()
        self.player_default_coordinates = [
            find_center(self.blocks[self.entrance].dimensions, player.dimensions,
                        self.blocks[self.entrance].coordinates)[0],
            self.blocks[self.entrance].coordinates[1] +
            self.blocks[self.entrance].dimensions[1] -
            player.dimensions[1]
        ]
        player.default_coordinates = list(self.player_default_coordinates)
        player.total_reset()

    def reset_time(self):
        self.delay = 0
        self.time = background.level_times[background.level - 1][0]
        self.display_time = 1
        self.count = 0

    def analyze_map(self):
        blocks = {}
        self.level += 1
        self.entrance = None
        self.exit = None
        self.reset_time()

        for x in range(self.maps[self.level - 1].get_width()):
            for y in range(self.maps[self.level - 1].get_height()):
                color = self.get_color((x, y))

                if color in self.block_color_values:
                    block_type = self.block_color_values[color]

                    if block_type == 'entrance':
                        self.entrance = (x, y)

                    elif block_type == 'exit':
                        self.exit = (x, y)

                    blocks[(x, y)] = block_type

                elif color != (0, 0, 0, 0):
                    raise Exception("Unidentified block_color {0} at {1}".format(color, (x, y)))

        if not self.entrance or not self.exit:
            raise Exception("Missing door")

        self.doors = (self.entrance, self.exit)

        self.backgrounds = {}
        active_coordinates = (self.entrance,)

        while active_coordinates:
            all_new_coordinates = []
            for coordinates in active_coordinates:
                for direction in self.sides:
                    new_coordinates = tuple(combine_lists(coordinates, direction, '+'))
                    if new_coordinates not in self.backgrounds and (
                                    new_coordinates not in blocks or blocks[new_coordinates] != 'block'):
                        self.backgrounds[new_coordinates] = Block(self.block_sprites['background'],
                                                                  self.convert_from_grid(
                                                                      new_coordinates), 'background')
                        all_new_coordinates.append(new_coordinates)

            active_coordinates = tuple(all_new_coordinates)

        self.blocks = {}
        self.weapons = {'cannon': [], 'laser': []}
        left_lasers = {}
        end_rails = {}
        self.platforms = []
        self.tracks = []
        self.gate_heads = []

        for initial_coordinates in blocks:
            block_type = blocks[initial_coordinates]
            coordinates = self.convert_from_grid(initial_coordinates)

            if block_type in self.block_offsets and block_type != 'gate':
                coordinates = combine_lists(coordinates, self.block_offsets[block_type], '+')

            if block_type in self.weapons:
                direction = 0
                for sign in (1, -1):
                    color = self.get_color(combine_lists(initial_coordinates, (sign, 0), '+'))

                    if color in self.block_color_values and self.block_color_values[color] == 'block':
                        direction = -1 * sign

                        if direction == -1:
                            sprites = [
                                pygame.transform.flip(sprite, True, False) for sprite in
                                make_tuple(self.block_sprites[block_type])
                                ]
                        else:
                            sprites = self.block_sprites[block_type]
                        break

                if direction == 0:
                    raise Exception("Unsupported weapon at {0}".format(initial_coordinates))

                if block_type == 'laser':
                    block = Weapon(sprites, self.block_sprites['laser_projectile'], coordinates, block_type, direction,
                                   game.speed * .5)
                    if direction == -1:
                        left_lasers[initial_coordinates] = block
                    else:
                        self.weapons[block_type].append(block)

                else:
                    block = Cannon(sprites, self.block_sprites['cannon_projectile'], coordinates, direction,
                                   game.speed * 2.5)
                    self.weapons[block_type].append(block)

            elif block_type == 'gate_head':
                direction = None
                gate_coordinates = []
                for side in self.sides:
                    grid_coordinates = tuple(combine_lists(initial_coordinates, side, '+'))
                    if grid_coordinates in blocks and blocks[grid_coordinates] == 'gate':
                        direction = side
                        direction_index = opposite(direction.index(0))
                        gate_coordinates.append(tuple(grid_coordinates))
                        break

                if not direction:
                    raise Exception('Unsupported gate head')

                # noinspection PyUnboundLocalVariable
                grid_coordinates = list(grid_coordinates)

                while True:
                    # noinspection PyUnboundLocalVariable
                    grid_coordinates[direction_index] += 1
                    if blocks[tuple(grid_coordinates)] == 'gate':
                        gate_coordinates.append(tuple(grid_coordinates))

                    else:
                        break

                sprites = []

                for sprite in self.block_sprites['gate_head']:
                    rotation = None
                    if direction == (0, 1):
                        rotation = 90
                    elif direction == (1, 0):
                        rotation = 180
                    elif direction == (0, -1):
                        rotation = 270
                    if rotation:
                        sprite = pygame.transform.rotate(sprite.copy(), rotation)
                    sprites.append(sprite)

                block = GateHead(sprites, coordinates, gate_coordinates, direction_index)
                self.gate_heads.append(block)

            else:
                if block_type == 'rail':
                    if initial_coordinates in end_rails:
                        block_type = 'end_rail'
                        sprites = end_rails[initial_coordinates]

                    else:
                        sides = []

                        for side in self.sides:
                            grid_coordinates = tuple(combine_lists(initial_coordinates, side, '+'))
                            if grid_coordinates in blocks and blocks[grid_coordinates] == 'rail':
                                sides.append(side)

                        if len(sides) == 1:
                            block_type = 'end_rail'
                            sprites = self.block_sprites['end_rail'].copy()
                            rotation = None
                            if sides[0] == (0, 1):
                                rotation = 90
                            elif sides[0] == (1, 0):
                                rotation = 180
                            elif sides[0] == (0, -1):
                                rotation = 270
                            if rotation:
                                sprites = pygame.transform.rotate(sprites, rotation)

                            current_coordinates = tuple(combine_lists(initial_coordinates, sides[0], '+'))
                            rails = [initial_coordinates, current_coordinates]

                            while current_coordinates:
                                new_coordinates = []
                                for side in self.sides:
                                    grid_coordinates = tuple(combine_lists(current_coordinates, side, '+'))

                                    if grid_coordinates in blocks and blocks[
                                        grid_coordinates] == 'rail' and grid_coordinates not in rails:
                                        rails.append(grid_coordinates)
                                        new_coordinates = grid_coordinates
                                        break

                                current_coordinates = tuple(new_coordinates)

                            end_rail_sprite = self.block_sprites['end_rail'].copy()
                            rotation = None
                            if rails[-2][1] > rails[-1][1]:
                                rotation = 90
                            elif rails[-2][0] > rails[-1][0]:
                                rotation = 180
                            elif rails[-2][1] < rails[-1][1]:
                                rotation = 270
                            if rotation:
                                end_rail_sprite = pygame.transform.rotate(end_rail_sprite, rotation)
                            end_rails[rails[-1]] = end_rail_sprite

                            self.platforms.append(Platform(self.block_sprites['platform'], rails, 3))

                        elif len(sides) == 2:
                            if ((1, 0) in sides and (-1, 0) in sides) or ((0, 1) in sides and (0, -1) in sides):
                                block_type = 'straight_rail'
                                sprites = self.block_sprites['straight_rail'].copy()
                                if (0, 1) in sides:
                                    sprites = pygame.transform.rotate(sprites, 90)

                            else:
                                block_type = 'turn_rail'
                                sprites = self.block_sprites['turn_rail'].copy()
                                rotation = None
                                if (0, 1) in sides:
                                    if (-1, 0) in sides:
                                        rotation = 90
                                    else:
                                        rotation = 180
                                elif (1, 0) in sides and (0, -1) in sides:
                                    rotation = 270
                                if rotation:
                                    sprites = pygame.transform.rotate(sprites, rotation)

                        else:
                            raise Exception('Undefined rail at {0}'.format(initial_coordinates))

                    self.tracks.append(initial_coordinates)

                elif block_type == 'block':
                    sprites = self.block_sprites[block_type].copy()

                    sides = dict(self.side_dict)
                    far = self.block_size - game.scale_factor

                    for side in sides:
                        grid_coordinates = tuple(combine_lists(initial_coordinates, side, '+'))

                        if (grid_coordinates in blocks and blocks[grid_coordinates] not in ('block', 'cannon')) or (
                                        grid_coordinates not in blocks and grid_coordinates in self.backgrounds):
                            # if grid_coordinates not in blocks or blocks[grid_coordinates] != 'cannon':
                            sides[side] = True

                            rect_coordinates = [0, 0]
                            rect_size = [0, 0]
                            index = opposite(side.index(0))

                            if side[index] == 1:
                                rect_coordinates[index] = far

                            rect_size[index] = game.scale_factor
                            rect_size[opposite(index)] = self.block_size

                            pygame.draw.rect(sprites, (0, 0, 0, 255), (rect_coordinates, rect_size))

                    def draw(coords):
                        pygame.draw.rect(sprites, self.block_sprites['background'].get_at((0, 0)),
                                         (coords, (game.scale_factor, game.scale_factor)))

                    if sides[(-1, 0)]:
                        if sides[(0, -1)]:
                            draw((0, 0))
                        if sides[(0, 1)]:
                            draw((0, far))

                    if sides[(1, 0)]:
                        if sides[(0, -1)]:
                            draw((far, 0))
                        if sides[(0, 1)]:
                            draw((far, far))

                else:
                    sprites = self.block_sprites[block_type]

                block = Block(sprites, coordinates, block_type)

            for grid_coordinates in block.all_grid_coordinates:
                self.blocks[grid_coordinates] = block

            if initial_coordinates in self.doors:
                if block_type == 'entrance':
                    offset = 'entrance_background'
                else:
                    offset = 'exit_background'
                block.background_coordinates = combine_lists(self.block_offsets[offset], block.coordinates, '+')

        for laser in self.weapons['laser']:
            x = laser.all_grid_coordinates[0][0] + 1
            counterpart = None
            all_coordinates = []

            while x < self.maps[self.level - 1].get_width():
                coordinates = (x, laser.all_grid_coordinates[0][1])

                if coordinates in left_lasers:
                    # noinspection PyTypeChecker
                    counterpart = left_lasers[coordinates]
                    break

                all_coordinates.append(
                    combine_lists(self.convert_from_grid(coordinates), self.block_offsets['laser_projectile'], '+'))
                x += 1

            if not counterpart:
                raise Exception('Single laser at {0}'.format(laser.all_grid_coordinates[0]))

            laser.counterpart = counterpart
            laser.projectile_coordinates = all_coordinates
            laser.active = 1

        for gate_head in self.gate_heads:
            for gate_coordinates in gate_head.gate_coordinates:
                gate =  self.blocks[gate_coordinates]
                if gate_head.direction_index == 1:
                    gate.sprites = pygame.transform.rotate(gate.sprites, 90)
                    offset = (self.block_offsets['gate'][1], 0)
                else:
                    offset = self.block_offsets['gate']
                gate.coordinates = combine_lists(gate.coordinates, offset, '+')


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


class GateHead(Block):
    def __init__(self, sprites, coordinates, gate_coordinates, direction_index):
        super().__init__(sprites, coordinates, 'gate_head')
        self.gate_coordinates = gate_coordinates
        self.direction_index = direction_index

    def retract(self):
        pass


class Button(Block):
    def __init__(self, sprites, coordinates, kind):
        super().__init__(sprites, coordinates, kind)
        self.pushed = False


class Weapon(Block):
    def __init__(self, sprites, projectile_sprite, coordinates, kind, direction, projectile_frequency):
        super().__init__(sprites, coordinates, kind)
        self.projectile_sprite = projectile_sprite
        self.direction = direction
        self.projectile_dimensions = self.projectile_sprite.get_size()
        self.projectile_frequency = projectile_frequency


class Cannon(Weapon):
    def __init__(self, sprites, projectile_sprite, coordinates, direction, projectile_frequency):
        super().__init__(sprites, projectile_sprite, coordinates, 'cannon', direction, projectile_frequency)
        if self.direction == 1:
            x = self.coordinates[0]
        else:
            x = self.coordinates[0] + self.dimensions[0] - self.projectile_dimensions[0]
        self.projectile_initial_coordinates = (
            x, find_center(self.dimensions, self.projectile_dimensions, self.coordinates)[1])
        self.projectile_velocity = [1 * game.scale_factor * self.direction, 0]
        self.projectile_coordinates = []


class Platform(Thing):
    def __init__(self, sprites, rails, speed):
        self.speed = speed
        if background.block_size % self.speed != 0:
            raise Exception('Bad platform speed')
        self.rail_coordinates = [background.convert_from_grid(rail) for rail in rails]
        self.current_rail_number = 0
        self.rail_direction = -1
        super().__init__(sprites, coordinates=self.rail_coordinates[self.current_rail_number])

    def move(self):
        if self.coordinates in self.rail_coordinates:
            if self.current_rail_number in (0, len(self.rail_coordinates) - 1):
                self.rail_direction *= -1
            self.current_rail_number += self.rail_direction
            self.direction = [polarity(self.rail_coordinates[self.current_rail_number][i] - self.coordinates[i]) for i
                              in range(2)]
            self.direction_index = opposite(self.direction.index(0))

        self.coordinates[0] += self.speed * self.direction[0]
        self.coordinates[1] += self.speed * self.direction[1]

        self.all_grid_coordinates = background.find_all_grid_coordinates(self.coordinates, self.dimensions)


class Mob(Thing):
    def __init__(self, sprites, conditions_info, coordinates=list((0, 0)),
                 default_sprite_type='stagnant', visible_direction=True):
        self.conditions_info = conditions_info
        self.default_coordinates = coordinates
        self.default_sprite_type = default_sprite_type
        self.visible_direction = visible_direction
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
        self.all_grid_coordinates = background.find_all_grid_coordinates(self.coordinates, self.dimensions)

    def process_keys(self, keys):
        if (keys[K_RIGHT] or keys[K_d]) and self.direction != 1:
            self.direction = 1
        elif (keys[K_LEFT] or keys[K_a]) and self.direction != -1:
            self.direction = -1

    def grid_coordinates(self, velocity=True):
        if velocity:
            coordinates = combine_lists(self.coordinates, self.velocity, '+')
        else:
            coordinates = self.coordinates
        return background.find_all_grid_coordinates(coordinates, self.dimensions)


class Player(Mob):
    def __init__(self, sprites, conditions_info):
        super().__init__(sprites, conditions_info, visible_direction=False)
        self.default_fake_coordinates = find_center(game.dimensions, self.dimensions)
        self.fake_coordinates = list(self.default_fake_coordinates)
        self.block = None

    def generate_display_coordinates(self, coordinates):
        return combine_lists(
            combine_lists(coordinates, self.fake_coordinates, '+'),
            self.coordinates, '-')

    def die(self, total=False):
        self.current_sprite_type = 'dying'
        if total:
            game.condition = 'total_reset'
        else:
            game.condition = 'reset'


game = Game(30, (1000, 800), 3, 9)

# background level_maps processing

background_map_sheet = SpriteSheet('background_map_sheet_V{0}.png'.format(game.version), 1)
background_maps_raw = background_map_sheet.get_sprites(y_constant=50, x_constant=(50, 3), scale=False)

# other sprites processing

sprite_sheet = SpriteSheet('sprite_sheet_V{0}.png'.format(game.version), 1)

# background object processing

block_names = ('lava', 'fire', 'block', 'background', 'cannon', 'platform', 'straight_rail', 'turn_rail', 'end_rail')
blocks = sprite_sheet.get_sprites(y_constant=10, x_constant=(10, len(block_names)))

background_sprites = {name: blocks[block_names.index(name)]
                      for name in block_names}

misc_object_names = ('cannon_projectile', 'laser_projectile', 'gate')
misc_objects = sprite_sheet.get_sprites(farthest_x_coordinate=46, update=False,
                                        all_coordinates=((49, 13), (59, 13), (69, 13)))

misc_object_dict = {name: misc_objects[misc_object_names.index(name)]
                    for name in misc_object_names}
background_sprites.update(misc_object_dict)

background_door_signs = sprite_sheet.get_sprites(y_constant=9, all_coordinates=(29, 45))
background_doors = sprite_sheet.get_sprites(y_constant=11, x_constant=(16, 10))
background_sprites['door_background'] = background_doors[-1]
del background_doors[-1]

door_types = ('entrance', 'exit')
door_background_offsets = {}

for door_sign_index in range(2):
    if door_sign_index == 0:
        iterator = background_doors
    else:
        iterator = reversed(background_doors)

    for door_sprite in iterator:
        door_sign_sprite = background_door_signs[door_sign_index]

        surface_size = (max(
            door_sign_sprite.get_width(),
            door_sprite.get_width()
        ), door_sign_sprite.get_height() + door_sprite.get_height())

        # noinspection PyArgumentList
        surface = pygame.Surface(surface_size, SRCALPHA).convert_alpha()

        surface.blit(door_sign_sprite, (find_center(
            surface_size, door_sign_sprite.get_size())[0], 0))
        door_coordinates = (find_center(
            surface_size, door_sprite.get_size())[0], door_sign_sprite.get_height())
        surface.blit(door_sprite, door_coordinates)

        background_sprites.setdefault(door_types[door_sign_index], []).append(surface)

    # noinspection PyUnboundLocalVariable
    door_background_offsets[door_types[door_sign_index]] = door_coordinates

flag_sprites_raw = sprite_sheet.get_sprites(y_constant=10, x_constant=(8, 3))
background_sprites['flag'] = flag_sprites_raw

laser_sprites_raw = sprite_sheet.get_sprites(y_constant=8, x_constant=(10, 5))
background_sprites['laser'] = laser_sprites_raw

number_sprites = sprite_sheet.get_sprites(y_constant=40, x_constant=(20, 10))
background_sprites['numbers'] = number_sprites

button_sprites = sprite_sheet.get_sprites(all_coordinates=((19, 94), (39, 93), (59, 92), (79, 91)))
background_sprites['button'] = button_sprites

gate_head_sprites = sprite_sheet.get_sprites(y_constant=10, x_constant=(10, 2))
background_sprites['gate_head'] = gate_head_sprites

title_sprites_raw = sprite_sheet.get_sprites(farthest_y_coordinate=210, x_constant=(318, 1), y_constant=53,
                                             update=False)
background_sprites['title'] = title_sprites_raw


def convert_to_color(number, base=6):
    c = int(number / base ** 2)
    number -= (c * base ** 2)
    b = int(number / base)
    number -= (b * base)
    a = number
    return a * 51, b * 51, c * 51, 255


background_block_names = (
    'block', 'lava', 'fire', 'flag', 'cannon', 'entrance', 'exit', 'laser', 'rail', 'button', 'gate', 'gate_head')

background_color_values = {color: name for color, name in zip(
    [
        convert_to_color(number) for number in
        range(len(background_block_names))
        ], background_block_names)}

background_offsets = {
    'flag': (1, 0),
    'exit': (2, 0),
    'laser': (0, 1),
    'laser_projectile': (0, 3),
    'entrance_background': door_background_offsets['entrance'],
    'exit_background': door_background_offsets['exit'],
    'button': (0, 3),
    'gate': (0, 3)
}

level_times = [(1000, 5), (30, 5), (120, 20)]

background = Background(background_maps_raw, background_sprites, background_color_values, background_offsets,
                        level_times)

# player sprites processing

player_sprites_raw = sprite_sheet.get_sprites(y_constant=10, x_constant=(10, 8), farthest_y_coordinate=200)

player_sprites = {
    'stagnant': player_sprites_raw[0],
    'dying': get_list(player_sprites_raw, range(1, 8))
}

player = Player(player_sprites, {
    'moving': {'active': False, 'velocity': int(1.7 * game.scale_factor)},
    'jumping': {'active': False, 'velocity': -30 * game.scale_factor, 'quadratic': None, 'speed': .5},
    'falling': {'active': False, 'velocity': 60 * game.scale_factor, 'quadratic': None, 'speed': .7},
})

# debugging  tools
background.level = 0
game.condition = None
background.update_level()
# player.coordinates = background.convert_from_grid((15, 16))

while True:
    for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_SPACE):
            game.exit()
        elif event.type == KEYDOWN and event.key == K_z:
            if game.real_speed == 30:
                game.real_speed = 3
            else:
                game.real_speed = 30

        elif event.type == KEYDOWN and event.key == K_x:
            player.coordinates = background.convert_from_grid((15, 16))

    if background.level != 0:
        for platform in background.platforms:
            platform.move()

    if game.condition:
        if game.condition == 'reset':
            if player.update_sprites(3) == 'completed':
                player.total_reset()
                game.condition = None

        elif game.condition == 'total_reset':
            if player.update_sprites(3, False) == 'completed':
                player.default_coordinates = list(background.player_default_coordinates)
                player.total_reset()
                background.blocks[background.entrance].reset()
                background.reset_time()
                game.condition = 'doors_opening'
                continue

        elif game.condition == 'doors_closing':
            if background.blocks[background.exit].update_sprites(5, False) == 'completed':
                game.entering_quadratic = Quadratic(-1, (
                    player.fake_coordinates[1], player.fake_coordinates[1] - game.dimensions[1]), 1)
                game.condition = 'transitioning'
                continue

        elif game.condition == 'doors_opening':
            if background.blocks[background.entrance].update_sprites(5, False) == 'completed':
                game.condition = None

        elif game.condition == 'transitioning':
            result = game.entering_quadratic.execute()

            if type(result) == tuple:
                if game.entering_phase == 1:
                    background.update_level()
                    player.fake_coordinates[1] = player.default_fake_coordinates[1] + game.dimensions[1]
                    game.entering_quadratic = Quadratic(1, (
                        player.fake_coordinates[1], player.default_fake_coordinates[1]), 3)

                    game.entering_phase = 2

                else:
                    game.condition = 'doors_opening'
                    game.entering_phase = 1

                continue

            else:
                player.fake_coordinates[1] += result

        elif game.condition == 'title':
            if not game.entering_quadratic:
                player.fake_coordinates[1] = player.default_fake_coordinates[1] + game.dimensions[1] * 2
                game.entering_quadratic = Quadratic(-1, (
                    player.fake_coordinates[1], player.default_fake_coordinates[1] + game.dimensions[1]), 1)

            else:
                pygame.time.wait(2000)
                game.condition = 'transitioning'
                continue

    else:
        background.count += 1
        if background.count % game.speed == 0:
            background.time -= 1
            if background.time == 0:
                player.die(True)

        player.velocity = [0, 0]
        previous_block = player.block
        player.block = None

        for grid_coordinates in background.find_all_grid_coordinates(
                (player.coordinates[0], player.coordinates[1] + player.dimensions[1]),
                (player.dimensions[0], 1)):
            if background.block_type(grid_coordinates) == 'block':
                player.block = background.blocks[grid_coordinates]
                break

        if not player.block:
            for platform in background.platforms:
                if collision((player.coordinates[0], player.coordinates[1] + 1), player.dimensions,
                             platform.coordinates, platform.dimensions) is True or (
                                previous_block == platform and collision(
                            (player.coordinates[0], player.coordinates[1] + 1 + platform.speed), player.dimensions,
                            platform.coordinates, platform.dimensions)) is True:
                    player.block = platform
                    player.velocity[platform.direction_index] = platform.speed * platform.direction[
                        platform.direction_index]
                    break

        keys = pygame.key.get_pressed()

        if keys[K_RIGHT] or keys[K_LEFT] or keys[K_d] or [K_a]:
            if not player.conditions['moving']:
                player.conditions['moving'] = True

        if (keys[K_UP] or keys[K_w]) and player.block and not (
                    player.conditions['jumping'] or player.conditions['falling']):
            player.conditions['jumping'] = True
            # noinspection PyTypeChecker
            player.conditions_info['jumping']['quadratic'] = Quadratic(1, (0,
                                                                           player.conditions_info['jumping'][
                                                                               'velocity']),
                                                                       player.conditions_info['jumping'][
                                                                           'speed'])

        if not (keys[K_RIGHT] and keys[K_d]) and player.direction == 1 or not (
                    keys[K_LEFT] and keys[K_a]) and player.direction == -1:
            player.direction = 0

        player.process_keys(keys)

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
                    player.velocity[0] += player.conditions_info['moving']['velocity'] * player.direction

                if condition == 'jumping':
                    # noinspection PyUnresolvedReferences
                    result = player.conditions_info['jumping']['quadratic'].execute()
                    if type(result) == tuple:
                        player.reset(('jumping',))
                    else:
                        player.velocity[1] += result

                elif condition == 'falling':
                    # noinspection PyUnresolvedReferences
                    result = player.conditions_info['falling']['quadratic'].execute()
                    player.velocity[1] += make_tuple(result)[0]

        player.all_grid_coordinates = player.grid_coordinates()

        for platform in background.platforms:
            if collision(combine_lists(player.coordinates, player.velocity, '+'), player.dimensions,
                         platform.coordinates, platform.dimensions) is True:
                if collision(player.coordinates, player.dimensions, platform.coordinates, platform.dimensions) is True:
                    player.velocity[platform.direction_index] = 0
                    if platform.direction_index == 1:
                        player.reset(('jumping', 'falling'))
                    if platform.direction[platform.direction_index] == 1:
                        player.coordinates[platform.direction_index] = platform.coordinates[platform.direction_index] + \
                                                                       platform.dimensions[platform.direction_index]
                    else:
                        player.coordinates[platform.direction_index] = platform.coordinates[platform.direction_index] - \
                                                                       player.dimensions[platform.direction_index]

                else:
                    for i in range(2):
                        velocity = [0, 0]
                        velocity[i] = player.velocity[i]
                        if collision(combine_lists(player.coordinates, velocity, '+'), player.dimensions,
                                     platform.coordinates, platform.dimensions) is True:
                            if velocity[i] > 0:
                                player.velocity[i] = platform.coordinates[i] - (
                                    player.coordinates[i] + player.dimensions[i])
                            else:
                                player.velocity[i] = (platform.coordinates[i] + platform.dimensions[i]) - \
                                                     player.coordinates[i]
                            if i == 1:
                                player.reset(('jumping', 'falling'))

                    if collision(combine_lists(player.coordinates, player.velocity, '+'), player.dimensions,
                                 platform.coordinates, platform.dimensions) is True:
                        for i in range(2):
                            if player.velocity[i] > 0:
                                player.velocity[i] = platform.coordinates[i] - (
                                    player.coordinates[i] + player.dimensions[i])
                            else:
                                player.velocity[i] = (platform.coordinates[i] + platform.dimensions[i]) - \
                                                     player.coordinates[i]
                            player.reset(('jumping', 'falling'))

        player.all_grid_coordinates = player.grid_coordinates()

        for grid_coordinates in player.all_grid_coordinates:
            block_type = background.block_type(grid_coordinates)
            if block_type in ('block', 'cannon', 'gate_head', 'gate'):
                block = background.blocks[grid_coordinates]
                for i in range(2):
                    velocity = [0, 0]
                    velocity[i] = player.velocity[i]
                    if grid_coordinates in background.find_all_grid_coordinates(
                            combine_lists(player.coordinates, velocity, '+'),
                            player.dimensions):
                        if velocity[i] > 0:
                            player.velocity[i] = block.coordinates[i] - (
                                player.coordinates[i] + player.dimensions[i])
                        elif velocity[i] < 0:
                            player.velocity[i] = (block.coordinates[i] + block.dimensions[i]) - player.coordinates[i]
                        else:
                            player.velocity = (0, 0)
                            player.die()
                            break
                        if i == 1:
                            player.reset(('jumping', 'falling'))
                if game.condition == 'reset':
                    break

                if collision(combine_lists(player.coordinates, player.velocity, '+'), player.dimensions,
                             block.coordinates, block.dimensions) is True:
                    for i in range(2):
                        if player.velocity[i] > 0:
                            player.velocity[i] = block.coordinates[i] - (
                                player.coordinates[i] + player.dimensions[i])
                        elif player.velocity[i] < 0:
                            player.velocity[i] = (block.coordinates[i] + block.dimensions[i]) - player.coordinates[
                                i]
                        player.reset(('jumping', 'falling'))

        player.all_grid_coordinates = player.grid_coordinates()

        if not game.condition:
            for grid_coordinates in player.all_grid_coordinates:
                block_type = background.block_type(grid_coordinates)
                if block_type:
                    if block_type == 'exit':
                        if collision(
                                combine_lists(player.coordinates, player.velocity, '+'),
                                player.dimensions,
                                background.blocks[grid_coordinates].coordinates,
                                background.blocks[grid_coordinates].dimensions,
                                True) is True:
                            player.coordinates = (
                                find_center(
                                    background.blocks[grid_coordinates].dimensions,
                                    player.dimensions,
                                    c1=background.blocks[grid_coordinates].coordinates)[0],
                                background.blocks[grid_coordinates].coordinates[1] +
                                background.blocks[grid_coordinates].dimensions[1] -
                                player.dimensions[1]
                            )

                            game.condition = 'doors_closing'
                            break

                    elif block_type in ('fire', 'lava'):
                        player.die()
                        break

                    elif block_type == 'flag' and background.blocks[grid_coordinates].sprite_index == 0:
                        player.default_coordinates = background.blocks[grid_coordinates].coordinates
                        background.blocks[grid_coordinates].transforming = True

        if game.condition != 'doors_closing':
            player.coordinates = combine_lists(player.velocity, player.coordinates, '+')

        if not player.block:
            for platform in background.platforms:
                if collision((player.coordinates[0], player.coordinates[1] + 1), player.dimensions,
                             platform.coordinates, platform.dimensions) is True:
                    player.block = platform
                    player.velocity[platform.direction_index] = platform.speed * platform.direction[
                        platform.direction_index]
                    break

    if background.level != 0:
        for cannon in background.weapons['cannon']:
            for coordinates in cannon.projectile_coordinates:
                coordinates[0] += cannon.projectile_velocity[0]
                coordinates[1] += cannon.projectile_velocity[1]

                for grid_coordinates in background.find_all_grid_coordinates(coordinates, cannon.projectile_dimensions):
                    if grid_coordinates in player.all_grid_coordinates:
                        if collision(coordinates, cannon.projectile_dimensions, player.coordinates,
                                     player.dimensions) is True:
                            player.die()
                            cannon.projectile_coordinates.remove(coordinates)
                            break

                    elif (background.block_type(
                            grid_coordinates) == 'cannon' and grid_coordinates not in cannon.all_grid_coordinates) or background.block_type(
                        grid_coordinates) == 'block':
                        cannon.projectile_coordinates.remove(coordinates)
                        break

                    for platform in background.platforms:
                        if grid_coordinates in platform.all_grid_coordinates:
                            if collision(coordinates, cannon.projectile_dimensions, platform.coordinates,
                                         platform.dimensions) is True:
                                cannon.projectile_coordinates.remove(coordinates)

            if game.count % cannon.projectile_frequency == 0:
                cannon.projectile_coordinates.append(list(cannon.projectile_initial_coordinates))

        for laser in background.weapons['laser']:
            if game.count % laser.projectile_frequency == 0:
                laser.active = opposite(laser.active)
            if laser.active:
                laser.update_sprites()
                laser.counterpart.update_sprites()
                for coordinates in laser.projectile_coordinates:
                    for grid_coordinates in background.find_all_grid_coordinates(coordinates,
                                                                                 laser.projectile_dimensions):
                        if grid_coordinates in player.all_grid_coordinates:
                            if collision(coordinates, laser.projectile_dimensions,
                                         player.coordinates,
                                         player.dimensions) is True:
                                player.die()
                                break

    game.display.fill(background.color)

    if background.level != 0:
        for grid_coordinates in background.backgrounds:
            block = background.backgrounds[grid_coordinates]
            game.display.blit(block.current_sprite(), player.generate_display_coordinates(block.coordinates))

        for door in background.doors:
            block = background.blocks[door]
            game.display.blit(background.block_sprites['door_background'],
                              player.generate_display_coordinates(block.background_coordinates))

        for track in background.tracks:
            block = background.blocks[track]
            game.display.blit(block.current_sprite(),
                              player.generate_display_coordinates(block.coordinates))

        for laser in background.weapons['laser']:
            if laser.active:
                for coordinates in laser.projectile_coordinates:
                    game.display.blit(laser.projectile_sprite, player.generate_display_coordinates(coordinates))
        for platform in background.platforms:
            game.display.blit(platform.current_sprite(),
                              player.generate_display_coordinates(platform.coordinates))

        for grid_coordinates in background.blocks:
            if grid_coordinates not in background.doors and grid_coordinates not in background.tracks:
                block = background.blocks[grid_coordinates]

                if block.kind == 'cannon':
                    for projectile_coordinates in block.projectile_coordinates:
                        game.display.blit(block.projectile_sprite,
                                          player.generate_display_coordinates(projectile_coordinates))

                elif block.kind == 'flag':
                    if block.transforming:
                        if block.update_sprites(5, reset=False):
                            block.transforming = False

                game.display.blit(block.current_sprite(),
                                  player.generate_display_coordinates(block.coordinates))

        if game.condition in ('doors_closing', 'doors_opening', 'transitioning'):
            game.display.blit(player.current_sprite(), player.fake_coordinates)
            player.display_after = False
        else:
            player.display_after = True

        for door in background.doors:
            block = background.blocks[door]
            game.display.blit(block.current_sprite(),
                              player.generate_display_coordinates(block.coordinates))

        if player.display_after:
            game.display.blit(player.current_sprite(), player.fake_coordinates)

        if background.time <= background.level_times[background.level - 1][1]:
            if background.count % (int(game.speed / 2)) == 0:
                background.display_time = opposite(background.display_time)

        if background.display_time:
            time = str(background.time)
            for number, i in zip(time, reversed(range(len(time)))):
                game.display.blit(background.block_sprites['numbers'][int(number)],
                                  (game.dimensions[0] - (i + 1) * (background.number_width + 10) - 40, 50))

    else:
        game.display.blit(background.block_sprites['title'][0], (
            find_center(game.dimensions, background.block_sprites['title'][0].get_size())[0],
            -game.dimensions[1] * 2 + player.fake_coordinates[1]
        ))

    pygame.display.update()
    game.clock.tick(game.real_speed)
    game.count += 1
