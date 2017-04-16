import pygame
# from genericfunctions import *

class SpriteSheet:
    def __init__(self, filename, default_scale):
        self.sheet = pygame.image.load(filename).convert_alpha()
        if default_scale <= 1:
            self.default_scale = None
        else:
            self.default_scale = default_scale
        self.farthest_y = 0

    def scale_sprite(self, scale, sprite):
        if (not scale and self.default_scale) or scale:
            if not scale:
                scale = self.default_scale
            return pygame.transform.scale(sprite, (sprite.get_width() * scale, sprite.get_height() * scale))

    def get_image(self, dimensions, x, scale=None):
        sprite = pygame.Surface(dimensions, pygame.SRCALPHA).convert_alpha()
        sprite.blit(self.sheet, (0, 0), ((x, self.farthest_y), dimensions))
        sprite = self.scale_sprite(scale, sprite)
        return sprite

    def get_sprite(self, dimensions, x=0, scale=None):
        return [self.get_image(dimensions, x, scale)]

    def get_sprites(self, dimensions, x=0, scale=None, update=True):
        sprites = []
        for i_dimensions in dimensions:
            sprites.append(self.get_image(i_dimensions, x, scale))
            x += i_dimensions[0]
        if update:
            self.farthest_y += max(
                [i_dimensions[1] for i_dimensions in dimensions]
            )
        return sprites

    def get_custom_blocks(self, dimensions, number, x=0, scale=None, update=True):
        sprites = []
        for i in range(number):
            sprites.append(self.get_image(dimensions, x, scale))
            x += dimensions[0]
        if update:
            self.farthest_y += dimensions[1]
        return sprites

    def get_custom(self, dimensions, constant, index, x=0, scale=None, update=True):
        sprites = []
        for i_dimensions in dimensions:
            current_dimensions = [0, 0]
            current_dimensions[index] = constant
            current_dimensions[abs(index - 1)] = i_dimensions
            sprites.append(self.get_image(current_dimensions, x, scale))
            x += current_dimensions[0]
        if update:
            if index == 1:
                self.farthest_y += constant
            else:
                self.farthest_y += max(dimensions)
        return sprites

class BlockSheet(SpriteSheet):
    def __init__(self, filename, default_scale, block_size):
        super().__init__(filename, default_scale)
        self.block_size = block_size

    def get_blocks(self, number, x=0, scale=None, update=True):
        sprites = []
        for i in range(number):
            sprites.append(self.get_image(self.block_size, x, scale))
            x += self.block_size[0]
        if update:
            self.farthest_y += self.block_size[1]
        return sprites
