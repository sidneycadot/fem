#! /usr/bin/env python3

import time

import numpy as np
import matplotlib.pyplot as plt

class Coord2D:

    def __init__(self, h_pixels: int, v_pixels, h_size: float, v_size):
        self.h_pixels = h_pixels
        self.v_pixels = v_pixels
        self.h_size = h_size
        self.v_size = v_size

    def physical_to_grid(self, x: float, y: float) -> tuple[float, float]:

        gx = 0.5 * (self.h_pixels - 1 + (2 * self.h_pixels * x) / self.h_size)
        gy = 0.5 * (self.v_pixels - 1 - (2 * self.v_pixels * y) / self.v_size)

        return (gx, gy)

    def physical_to_integer_grid(self, x: float, y: float) -> tuple[int, int]:
        (gx, gy) = self.physical_to_grid(x, y)
        return (round(gx), round(gy))

    def grid_to_physical(self, gx: float|int, gy: float|int) -> tuple[float, float]:
        x = +self.h_size * (1 - self.h_pixels + 2 * gx) / (2 * self.h_pixels)
        y = -self.v_size * (1 - self.v_pixels + 2 * gy) / (2 * self.v_pixels)
        return (x, y)

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Disc:
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def inside(self, x: float, y: float):
        return (self.x - x) ** 2 + (self.y - y) ** 2 < self.radius ** 2

def run_sim(resolution: int):

    coord = Coord2D(resolution, resolution, 0.50, 0.50)

    disc = Disc(0.0, 0.0, 0.20)

    source = Point(-0.10, 0)
    sink  = Point(+0.10, 0)

    source_grid_point = coord.physical_to_integer_grid(source.x, source.y)
    sink_grid_point = coord.physical_to_integer_grid(sink.x, sink.y)

    # Walk all voxels.

    index_to_grid_point = {}
    grid_point_to_index = {}

    n = 0
    for gy in range(coord.v_pixels):
        for gx in range(coord.h_pixels):
            (x, y) = coord.grid_to_physical(gx, gy)
            if disc.inside(x, y):
                index_to_grid_point[n] = (gx, gy)
                grid_point_to_index[(gx, gy)] = n
                n += 1

    print("number of voxels:", n)

    assert source_grid_point in grid_point_to_index
    assert sink_grid_point in grid_point_to_index

    print("source and sink are both in the grid.")

    # set up equations.

    source_voltage = 1.0e-3
    sink_voltage = 0.0e-3

    a = np.zeros((n, n))
    b = np.zeros(n)

    for (index, grid_point) in index_to_grid_point.items():
        if grid_point == source_grid_point:
            print("adding equation for source:", index)
            a[index, index] = 1.0
            b[index] = source_voltage
        elif grid_point == sink_grid_point:
            print("adding equation for sink:", index)
            a[index, index] = 1.0
            b[index] = sink_voltage
        else:
            count_neighbors = 0
            for (dx, dy) in ((+1, 0), (0, +1), (-1, 0), (0, -1)):
                xx = grid_point[0] + dx
                yy = grid_point[1] + dy
                neighbor_index = grid_point_to_index.get((xx, yy))
                if neighbor_index is not None:
                    a[index, neighbor_index] = -1.0
                    count_neighbors += 1
                a[index, index] = count_neighbors
                b[index] = 0.0

    print("solving...")
    t1 = time.monotonic()
    voltages = np.linalg.solve(a, b)
    t2 = time.monotonic()
    print("solved.")

    image = np.full((coord.v_pixels, coord.h_pixels), np.nan)
    for (index, v) in enumerate(voltages):
        (gx, gy) = index_to_grid_point[index]
        image[gy, gx] = v

    plt.clf()
    plt.imshow(image)
    plt.pause(1e-3)

    current_probes = {
        "source": source_grid_point,
        "sink": sink_grid_point,
    }

    conductance = 2.65e-8 # Ohm·m    aluminium
    thickness   = 16.0e-6 # 16 um    aluminium foil

    element_resistance = conductance/thickness

    la_resistance = None
    for (name, grid_point) in current_probes.items():
        index = grid_point_to_index[grid_point]
        current = 0.0
        for (dx, dy) in ((+1, 0), (0, +1), (-1, 0), (0, -1)):
            xx = grid_point[0] + dx
            yy = grid_point[1] + dy
            neighbor_index = grid_point_to_index.get((xx, yy))
            if neighbor_index is not None:
                current += (voltages[neighbor_index] - voltages[index]) / element_resistance
        print("resolution {} name {} current {} resistance {}".format(resolution, name, current, (source_voltage - sink_voltage) / current))
        if name == "sink":
            la_resistance = float((source_voltage - sink_voltage) / current)

    return (t2 - t1, la_resistance)

def main():

    resolutions = np.arange(50, 101, 10)
    resistances = []
    durations = []

    for resolution in resolutions:
        (duration, resistance) = run_sim(resolution)
        durations.append(duration)
        resistances.append(resistance)

    print("data:")
    print(resolutions)
    print(resistances)
    print(durations)

    plt.clf()
    plt.subplot(211)
    plt.title("calculated resistance")
    plt.xlabel("resolution [px]")
    plt.ylabel("resistance [Ohm]")
    plt.plot(resolutions, resistances)
    plt.subplot(212)
    plt.title("calculation time")
    plt.xlabel("resolution [px]")
    plt.ylabel("time [s]")
    plt.plot(resolutions, durations)
    plt.pause(1e-2)
    plt.show()

if __name__ == "__main__":
    main()
