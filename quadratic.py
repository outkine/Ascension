from math import sqrt

class Quadratic:
    def __init__(self, vertex_y, time_to_vertex):
        if time_to_vertex < 0:
            raise Exception("Time to vertex below 0")
        self.h = time_to_vertex
        self.k = vertex_y
        self.a = self.find_a(0, 0, self.h, self.k)
        self.reset()

    def reset(self):
        self.current_x = 0
        self.old_y = 0
        self.done = False

    def execute(self):
        if self.current_x >= self.h:
            self.done = True
        self.current_x += 1
        current_y = self.find_y(self.current_x, self.a, self.h, self.k)
        self.y_change = current_y - self.old_y
        self.old_y = current_y
        return self.y_change

    @staticmethod
    def find_a(x, y, h, k):
        return (y - k) / (x - h) ** 2

    @staticmethod
    def find_x(y, a, h, k):
        return sqrt((y - k) / a) + h

    @staticmethod
    def find_y(x, a, h, k):
        return a * (x - h) ** 2 + k
