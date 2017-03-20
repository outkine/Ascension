from math import sqrt

import pygame as pygame
from pygame.locals import *

pygame.init()


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


def find_center(c1, d1, d2):
    return [(c1[i] + (d1[i] / 2 - d2[i] / 2)) for i in range(2)]


def collision(c1, d1, c2, d2):
    d1 = list(d1)
    d2 = list(d2)
    collisions = [False, False]
    for i in range(2):
        d1[i] -= 1
        d2[i] -= 1
        if (c1[i] <= c2[i] and c1[i] + d1[i] >= c2[i] + d2[i]) or \
                (c1[i] >= c2[i] and c1[i] + d1[i] <= c2[i] + d2[i]) or \
                (c2[i] <= c1[i] <= c2[i] + d2[i]) or \
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
        image = pygame.Surface(dimensions, SRCALPHA).convert_alpha()
        image.blit(self.sheet, (0, 0), (coordinates, dimensions))
        return image

    def get_sprites(self, farthest_x_coordinate=0, all_coordinates=None, y_constant=None, x_constant=None, update=True):
        sprites = []
        if y_constant and x_constant:
            thing = range(x_constant[1])
        else:
            thing = all_coordinates
        for i in thing:
            coordinates_1 = [0, 0]
            coordinates_1[self.division_index] = self.farthest_y_coordinate
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
                    coordinates_2[self.division_index] = self.farthest_y_coordinate + y_constant - 1
                else:
                    coordinates_2[self.division_index] = i
            else:
                coordinates_2 = i
            farthest_x_coordinate += 1
            dimensions = combine_lists((1, 1), combine_lists(coordinates_2, coordinates_1, '-'), '+')
            sprite = self.get_image(coordinates_1, dimensions)
            sprites.append(pygame.transform.scale(sprite, combine_lists(sprite.get_size(),
                                                                        (game.scale_factor, game.scale_factor), '*')))
        if update:
            if y_constant:
                self.farthest_y_coordinate += y_constant
            elif x_constant:
                self.farthest_y_coordinate = max(all_coordinates) + 1
            else:
                self.farthest_y_coordinate = max(
                    [coordinates[self.division_index] for coordinates in all_coordinates]) + 1
        return sprites


class Game:
    def __init__(self):
        self.speed = 30
        self.dimensions = (1000, 800)
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)
        self.scale_factor = 3
        self.movement_keys = (K_RIGHT, K_LEFT, K_d, K_a)


class Background:
    def __init__(self, background_map, sprites, color_values):
        self.background_map = background_map
        self.sprites = sprites
        self.coordinates = {
            'block': []
        }
        self.color_values = color_values
        self.block_size = (30, 30)

    def analyze_background(self, origin):
        for x in range(self.background_map.get_width()):
            for y in range(self.background_map.get_height()):
                color = tuple(self.background_map.get_at((x, y)))
                if color == (0, 0, 0, 0):
                    continue
                elif color in self.color_values:
                    self.coordinates[self.color_values[color]].append(
                        combine_lists(self.block_size, (combine_lists((x, y), origin, '-')), '*'))
                else:
                    raise Exception("Unidentified block_color")


class Player:
    def __init__(self, sprites, coordinates=(0, 0)):
        self.sprites = sprites
        self.coordinates = list(coordinates)
        self.reset()
        self.display_coordinates = find_center((0, 0), game.dimensions, self.dimensions())
        self.velocity = [0, 0]
        self.total_velocity = [0, 0]
        self.conditions = {
            'moving': {'active': False, 'velocity': 5},
            'jumping': {'active': False, 'velocity': -100, 'quadratic': None},
            'falling': {'active': False, 'velocity': 25, 'quadratic': None}
        }
        self.direction = 1
        self.on_block = True

    def dimensions(self):
        return self.current_sprite().get_size()

    def update_sprites(self, speed=4):
        self.sprite_count += 1
        if self.sprite_count == speed:
            self.sprite_index += 1
            self.sprite_count = 0
            if self.sprite_index == len(self.current_sprites()):
                self.sprite_index = 0
                return 'completed'

    def current_sprites(self):
        return make_tuple(self.sprites)

    def current_sprite(self):
        return self.current_sprites()[self.sprite_index]

    def reset(self, conditions=None):
        self.sprite_count = 0
        self.sprite_index = 0
        if conditions:
            for condition in conditions:
                self.conditions[condition]['active'] = False

    def process_keys(self, keys):
        if (keys[K_RIGHT] or keys[K_d]) and self.direction != 1:
            self.direction = 1
        elif (keys[K_LEFT] or keys[K_a]) and self.direction != -1:
            self.direction = -1

    def update_coordinates(self):
        self.velocity = [int(self.velocity[i]) for i in range(2)]
        self.coordinates = combine_lists(self.velocity, self.coordinates, '+')
        self.total_velocity = combine_lists(self.velocity, self.total_velocity, '+')


class Quadratic:
    def __init__(self, reset, duration, z, velocity=0, ac=2, speed=1, c=None):
        self.b = 0
        self.reset = reset
        if c is not None:
            self.c = c
        else:
            self.c = velocity
        if self.c == 0:
            c = 1
        else:
            c = self.c
        self.a = abs(ac / c) * z
        self.x_change = (sqrt(abs(c)) / 10) * speed
        self.duration = duration
        for point in duration:
            index = duration.index(point)
            if point == 'solution':
                self.duration[index] = self.get_x()
            elif point == 'velocity':
                if index == 1:
                    sign = -1
                else:
                    sign = 1
                self.duration[index] = self.get_x(y=velocity) * sign
        self.current_x = self.duration[0]
        self.y = None
        self.old_y = None

    def get_a(self, x):
        return -(self.c / x ** 2)

    def get_y(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def get_x(self, y=0):
        return abs(sqrt(-4 * self.a * (self.c - y)) / (2 * self.a))

    def execute(self):
        self.old_y = self.get_y(self.current_x)
        self.current_x -= self.x_change
        if self.duration[1] is not None and self.current_x <= self.duration[1]:
            return 'completed'
        self.y = self.get_y(self.current_x)
        if self.reset is True:
            return self.y
        return self.y - self.old_y


game = Game()

background_map = pygame.image.load('background_map.png').convert_alpha()

sprite_sheet = SpriteSheet('sprite_sheet.png', 1)

background_sprites_raw = sprite_sheet.get_sprites(all_coordinates=((9, 9),))

background_sprites = {
    'block': background_sprites_raw[0]
}

background = Background(background_map, background_sprites,
                        {
                            (0, 0, 0, 255): 'block'
                        })

background.analyze_background((15, 92))

player_sprites_raw = sprite_sheet.get_sprites(all_coordinates=((9, 19),))

player = Player(player_sprites_raw)
#
# print(player.blocks)
# print(background.blocks['block'])

while True:
    player.velocity = [0, 0]

    player.on_block = False
    for block_coordinates in background.coordinates['block']:
        if collision((player.coordinates[0], player.coordinates[1] + 1), player.dimensions(), block_coordinates,
                     background.block_size) is True:
            player.on_block = True

    # print(player.on_block)

    for event in pygame.event.get():
        if event.type == QUIT or event.type == KEYDOWN and event.key == K_SPACE:
            pygame.quit()
            quit()

        if event.type == KEYDOWN:
            if event.key in game.movement_keys:
                if not player.conditions['moving']['active']:
                    player.conditions['moving']['active'] = True
                player.process_keys(pygame.key.get_pressed())

            if event.key in (K_UP, K_w) and player.on_block is True:
                player.conditions['jumping']['active'] = True
                # noinspection PyTypeChecker
                player.conditions['jumping']['quadratic'] = Quadratic(False, ['solution', 0], 1,
                                                                      velocity=player.conditions['jumping']['velocity'],
                                                                      speed=3)

        elif event.type == KEYUP:
            if event.key in game.movement_keys:
                keys = pygame.key.get_pressed()
                if not (keys[K_RIGHT] or keys[K_d]) and player.direction == 1 or not (
                            keys[K_LEFT] or keys[K_a]) and player.direction == -1:
                    player.direction = 0
                player.process_keys(keys)
                if player.direction == 0:
                    player.reset(('moving',))
                    player.direction = 1

    if not player.on_block and not player.conditions['falling']['active'] and not player.conditions['jumping'][
        'active']:
        player.conditions['falling']['active'] = True
        player.conditions['falling']['quadratic'] = Quadratic(False, [0, None], 1, speed=3, ac=4)

    for condition in player.conditions:
        if player.conditions[condition]['active']:
            if condition == 'moving':
                # noinspection PyTypeChecker
                player.velocity[0] += player.conditions['moving']['velocity'] * player.direction

            if condition == 'jumping':
                # noinspection PyUnresolvedReferences
                result = player.conditions['jumping']['quadratic'].execute()
                if result == 'completed':
                    player.reset(('jumping',))
                else:
                    player.velocity[1] += result

            elif condition == 'falling':
                # noinspection PyUnresolvedReferences
                result = player.conditions['falling']['quadratic'].execute()
                if result < 30:
                    player.velocity[1] += result
                else:
                    player.velocity[1] = result

    if player.velocity != [0, 0]:
        velocity_2 = list(player.velocity)
        result = None
        while True:
            collided = False
            repeat = False
            for block_coordinates in background.coordinates['block']:
                if collision(combine_lists(player.coordinates, velocity_2, '+'), player.dimensions(),
                             block_coordinates, background.block_size) is True:
                    collided = True
                    divisor = max(abs(velocity_2[0]), abs(velocity_2[1]))
                    velocity = list(velocity_2)
                    velocity_change = (velocity_2[0] / divisor, velocity_2[1] / divisor)
                    while True:
                        velocity = combine_lists(velocity, velocity_change, '-')
                        result = collision(combine_lists(player.coordinates, velocity, '+'), player.dimensions(),
                                           block_coordinates, background.block_size)
                        if result is True:
                            continue
                        if result == 0:
                            velocity_2[0] = velocity[0]
                        elif result == 1:
                            velocity_2[1] = velocity[1]
                        break
                    if repeat:
                        break
            if not collided:
                if result == 0:
                    player.velocity[0] = velocity[0]
                elif result == 1:
                    player.velocity[1] = velocity[1]
                    player.reset(('jumping', 'falling'))
                break

    player.update_coordinates()

    game.display.fill(pygame.Color('white'))

    for block_type in background.coordinates:
        for coordinates in background.coordinates[block_type]:
            game.display.blit(background.sprites[block_type],
                              combine_lists(combine_lists(coordinates, player.display_coordinates, '+'),
                                            player.total_velocity, '-'))

    game.display.blit(player.current_sprite(), player.display_coordinates)

    pygame.display.update()
    game.clock.tick(game.speed)
