import pygame
from math import sqrt


def opposite(n):
    return abs(n - 1)


class Quadratic:
    def __init__(self, sign, y_range, speed):
        self.b = 0
        self.a = 1
        if (sign == 1 and y_range[0] > y_range[1]) or (sign == -1 and y_range[1] > y_range[0]):
            x_solution_index = 0
        else:
            x_solution_index = 1
        self.c = y_range[opposite(x_solution_index)]
        self.x_range = [self.get_x(y_range[i])[x_solution_index] for i in range(2)]
        self.x_change = (self.x_range[1] - self.x_range[0]) / speed
        self.current_x = self.x_range[0]
        self.old_y = y_range[0]
        print(self.a, self.b, self.c)

    def execute(self):
        self.current_x += self.x_change
        if self.current_x > self.x_range[1]:
            return (self.y_change,)
        current_y = self.get_y(self.current_x)
        self.y_change = current_y - self.old_y
        self.old_y = current_y
        return self.y_change

    def get_y(self, x):
        return self.a * x ** 2 + self.b * x + self.c

    def get_x(self, y):
        return sorted(
            [(-self.b + i * sqrt((self.b ** 2) - (4 * self.a * self.c) + (4 * self.a * y))) / (2 * self.a) for i in
             (1, -1)]
        )


pygame.init()
display = pygame.display.set_mode((1000, 900))
clock = pygame.time.Clock()
coordinates = ([500, 0], [700, 0])
quadratics = (Quadratic(1, (0, -800), .5 * 30), Quadratic(1, (0, -800), 10 * 30))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    display.fill(pygame.Color('white'))
    for x in range(2):
        result = quadratics[x].execute()
        if type(result) != tuple:
            coordinates[x][1] -= result
            print(quadratics[x].y_change, x)
    pygame.draw.circle(display, pygame.Color('black'), [int(i) for i in coordinates[0]], 10)
    pygame.draw.circle(display, pygame.Color('black'), [int(i) for i in coordinates[0]], 10)
    pygame.display.update()
    clock.tick(30)
