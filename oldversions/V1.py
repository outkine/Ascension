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
                self.farthest_y_coordinate = max([coordinates[self.division_index] for coordinates in all_coordinates]) + 1
        return sprites


class Game:
    def __init__(self):
        self.speed = 30
        self.dimensions = (1000, 800)
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(self.dimensions)
        self.scale_factor = 3


class Background:
    def __init__(self, background_map, sprites, color_values):
        self.background_map = background_map
        self.sprites = sprites
        self.coordinates = {
            'block': []
        }
        self.color_values = color_values
        self.block_size = (10, 10)

    def analyze_background(self, origin):
        for x in range(self.background_map.get_width()):
            for y in range(self.background_map.get_height()):
                color = tuple(self.background_map.get_at((x, y)))
                print(color)
                if color == (0, 0, 0, 0):
                    continue
                elif color in self.color_values:
                    self.coordinates[self.color_values[color]].append(
                        combine_lists(combine_lists(self.block_size, (combine_lists((x, y), origin, '-')), '*'),
                                      (game.scale_factor, game.scale_factor), '*'))
                else:
                    raise Exception("Unidentified block_color")


class Player:
    def __init__(self, sprites, coordinates=(0, 0)):
        self.sprites = sprites
        self.coordinates = list(coordinates)
        self.reset()
        self.display_coordinates = self.coordinates = find_center((0, 0), game.dimensions, self.dimensions())
        print(self.dimensions())
        self.velocity = [0, 0]
        self.total_velocity = [0, 0]

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

    def reset(self):
        self.sprite_count = 0
        self.sprite_index = 0


# class BasicMob(Thing):
#     def __init__(self, sprites, sprites_info=None, default_sprite_type='stagnant', blocks=(0, 0)):
#         self.default_sprite_type = default_sprite_type
#         self.current_sprite_type = default_sprite_type
#         super().__init__(sprites, blocks)
#         self.sprites_info = sprites_info
#         self.direction = [0, 1]
#
#     def current_sprites(self):
#         return make_tuple(self.sprites[tuple(self.direction)][self.current_sprite_type])
#
#     def current_sprite_info(self, category):
#         if self.sprites_info:
#             if category in self.sprites_info:
#                 if self.current_sprite() in self.sprites_info[category]:
#                     return self.sprites_info[category][self.current_sprite()]
#
#     def display_coordinates(self):
#         if self.current_sprite_info('offsets'):
#             return combine_lists(self.blocks, self.current_sprite_info('offsets'), '+')
#         else:
#             return self.blocks
#
#     def dimensions(self, override=False):
#         if self.current_sprite_info('dimensions') and not override:
#             return self.current_sprite_info('dimensions')
#         else:
#             return super().dimensions()
#
#     def current_center(self):
#         return find_center(self.blocks, self.dimensions(), (1, 1))
#
#
# class Mob(BasicMob):
#     def __init__(self, sprites, sprites_info=None, default_sprite_type='stagnant', blocks=(0, 0)):
#         super().__init__(sprites, sprites_info, default_sprite_type, blocks)
#         self.conditions = {}
#         self.movement_direction = [0, 0]
#         self.velocity = [0, 0]
#
#     def add_velocity(self, list1):
#         return combine_lists(self.velocity, list1, '+')
#
#     def update_coordinates(self):
#         self.velocity = [int(self.velocity[i]) for i in range(2)]
#         self.blocks = self.add_velocity(self.blocks)
#
#     def initialize(self, condition):
#         self.reset()
#         self.conditions[condition]['active'] = True
#
#     def reset(self, conditions=None):
#         super().reset()
#         if conditions:
#             for condition in conditions:
#                 self.conditions[condition]['active'] = False
#                 if self.current_sprite_type == condition:
#                     self.current_sprite_type = self.default_sprite_type


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

background.analyze_background((0, 98))

player_sprites_raw = sprite_sheet.get_sprites(all_coordinates=((9, 19),))

player = Player(player_sprites_raw)


while True:
    for event in pygame.event.get():
        if event.type == QUIT or event.type == KEYDOWN and event.key == K_SPACE:
            pygame.quit()
            quit()

    game.display.fill(pygame.Color('white'))

    for block_type in background.coordinates:
        for coordinates in background.coordinates[block_type]:
            game.display.blit(background.sprites[block_type], combine_lists(combine_lists(coordinates, player.display_coordinates, '+'), player.total_velocity, '-'))

    game.display.blit(player.current_sprite(), player.coordinates)

    pygame.display.update()
    game.clock.tick(game.speed)
