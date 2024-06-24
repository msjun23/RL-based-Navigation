# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
import random
import datetime
import Queue
import math
# import tkinter as tk
# from method1 import ObstacleMap
from world_writer import WorldWriter
from difficulty_quant import DifficultyMetrics
from pgm_writer import PGMWriter
from yaml_writer import YamlWriter

# jackal takes up 2 extra grid squares on each side in addition to center square
jackal_radius = 2

# pgm file resolution
pgm_res = 0.15 # meters per pixel

# inflation radius found in planner params
infl_rad = 0.3 # meters

# radius of cylinders in the .world file
cyl_radius = 0.075

# length of containment wall, in meters
contain_wall_length = 5

def expand_path(path, width=4.0):
    expanded_path_left = []
    expanded_path_right = []

    for i in range(len(path) - 1):
        current_point = path[i]
        next_point = path[i + 1]
        vector = next_point - current_point
        orthogonal_vector = np.array([-vector[1], vector[0]], dtype=float)
        norm = np.linalg.norm(orthogonal_vector)
        if norm != 0:
            orthogonal_vector /= norm
            left_point = current_point + orthogonal_vector * width
            right_point = current_point - orthogonal_vector * width
            expanded_path_left.append(left_point)
            expanded_path_right.append(right_point)

    if len(expanded_path_left) > 0:
        expanded_path_left.append(expanded_path_left[-1] + vector)
        expanded_path_right.append(expanded_path_right[-1] + vector)

    return expanded_path_left, expanded_path_right

def getBoundary(expanded_path_left, expanded_path_right):
    boundary = expanded_path_left[:]
    for i in range(len(expanded_path_right)):
        boundary.append(expanded_path_right[len(expanded_path_right) - i - 1])
    boundary.append(boundary[0])
    return np.array(boundary)

def isInside(polygon, point):
    N = len(polygon) - 1
    counter = 0
    p1 = polygon[0]
    for i in range(1, N + 1):
        p2 = polygon[i % N]
        if point[1] > min(p1[1], p2[1]) and point[1] <= max(p1[1], p2[1]) and point[0] <= max(p1[0], p2[0]) and p1[1] != p2[1]:
            xinters = (point[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
            if p1[0] == p2[0] or point[0] <= xinters:
                counter += 1
        p1 = p2 
    return counter % 2 != 0

class ObstacleMap():
    def __init__(self, world,before_map ,boundary, rand_fill_pct, smooth_iter=10, fill_threshold=5, clear_threshold=1, seed=None):
        self.world = world
        self.before_map = before_map
        self.boundary = boundary
        self.rows = len(world)
        self.cols = len(world[0])
        self.rand_fill_pct = rand_fill_pct
        self.seed = seed
        self.smooth_iter = smooth_iter
        self.fill_threshold = fill_threshold
        self.clear_threshold = clear_threshold
        # space index를 따로 저장 (00은 free space, 01은 restricted space)
        self.space_index = [[cell for cell in row] for row in world]
        # 초기 맵 설정 (free space는 0, restricted space는 1)
        self.map = [[0 if cell == 0 else 1 for cell in row] for row in world]

    def __len__(self):
        return len(self.world)
    
    # 맵을 채우고 smooth_iter만큼 반복 실행
    def __call__(self):
        self._random_fill()
        for n in range(self.smooth_iter):
            self._smooth()

    # 초기 채우기 비율을 사용하여 맵을 랜덤하게 채움
    def _random_fill(self):
        if self.seed:
            random.seed(self.seed)
        # 시드가 주어지면 랜덤 시드를 설정
        # Seed 고정 시 random.random() 함수에서 나타나는 숫자 및 순서는 정해짐

        for r in range(self.rows):
            for c in range(self.cols):
                if self.space_index[r][c] == 0:  # free space인 경우에만
                    # random.random()이 초기 채우기 비율보다 작으면 셀을 채움
                    # random.random()은 0~1 값을 가지므로 결국 rand_fill_pct는 확률이며 실제 world의 fill percentage는 다를 수 있음
                    self.map[r][c] = 1 if random.random() < self.rand_fill_pct else 0
                else:
                    self.map[r][c] = 0  # space index가 01인 경우 항상 1로 고정

    # 채우기 임계값이 5이고 비우기 임계값이 1인 하나의 부드럽게 만드는 반복을 실행
    def _smooth(self):
        # 동일한 반복 내에서 진화가 서로 영향을 미치지 않도록 버퍼 맵을 사용
        # 셀들 순차적으로 업데이트하며 이미 업데이트 된 셀이 다른 셀의 업데이트에 영향 미칠 수 있기 때문
        newmap = [[self.map[r][c] for c in range(self.cols)] for r in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if self.space_index[r][c] == 0:  # free space인 경우에만 부드럽게 만듦
                    filled_neighbors = self._tile_neighbors(r, c)
                    # 이웃 셀이 fill_threshold 이상 채워져 있으면 이 셀을 채움
                    if filled_neighbors >= self.fill_threshold:
                        newmap[r][c] = 1
                    # 이웃 셀이 clear_threshold 이하로 채워져 있으면 이 셀을 비움
                    elif filled_neighbors <= self.clear_threshold:
                        newmap[r][c] = 0
                else:
                    newmap[r][c] = 1  # restricted space는 항상 1로 유지
        # 모든 셀의 상태 업데이트가 끝난 후, 버퍼 맵을 현재 맵으로 설정
        self.map = newmap

    # 채워진 이웃의 수를 반환 (8개의 이웃)
    def _tile_neighbors(self, r, c):
        count = 0
        # 이웃의 수를 세기 위한 변수를 초기화
        for i in range(r - 1, r + 2):
            for j in range(c - 1, c + 2):
                # print(self.world[i][j])
                point = [i,j]
                if (self._in_map(i, j) and (i != r or j != c)):
                    if self.before_map[i][j] != True and isInside(self.boundary, point):
                        count += self.map[i][j]
        return count
        # 이웃의 수를 반환

    # 좌표가 맵 안에 있는지 확인
    def _in_map(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    # 현재 맵을 반환
    def get_map(self):
        return self.map

    # 맵을 출력하는 메서드 추가
    def display_map(self):
        for row in self.map:
            print(' '.join(map(str, row)))
    def to_numpy(self):
        return np.array(self.map)

# define boilerplate code needed to write to .world file
with open('./world-boilerplate/world_boiler_start.txt') as f:
    world_boiler_start = f.read()
with open('./world-boilerplate/world_boiler_mid.txt') as f:
    world_boiler_mid = f.read()
with open('./world-boilerplate/world_boiler_end.txt') as f:
    world_boiler_end = f.read()
with open('./world-boilerplate/cylinder_define.txt') as f:
    cylinder_define = f.read()
with open('./world-boilerplate/cylinder_place.txt') as f:
    cylinder_place = f.read()

wall_rgb = [0.152, 0.379, 0.720]
obs_rgb = [0.648, 0.192, 0.192]

def Obstacle_Expand(free, obstacle_map, before_map,after_map, Boundary, rows, cols):
    for r in range(rows):
        for c in range(cols):
            if after_map[r][c] == True:
                for r2 in range(r-free, r+free+1):
                    for c2 in range(c-free, c+free+1):
                        if (0 <= r2 < rows and 0 <= c2 < cols) and before_map[r2][c2] == True:  # 범위 확인
                            obstacle_map[r2][c2] = False
    
    return obstacle_map

def main(iteration=0, seed=0, smooth_iter=4, fill_pct=.5, rows=60, cols=30, show_metrics=1):
    world_file = 'train_data/world_files/world_%d.world' % iteration
    path_file = 'test_data/path_files/path_%d.npy' % iteration
    grid_file = 'test_data/grid_files/grid_%d.npy' % iteration
    free_space = 8

    # 예제 패스 데이터 불러오기
    path = np.load(path_file)
    path = np.delete(path,0,axis=0)
    # print(type(path))
    # print(path[:,0])
    path[:, 0] += 10
    # path[:, 1] = (path[:, 1] + float(free_space) * cyl_radius * 2).astype(int)
    # path[:,1] += int(math.ceil((free_space) * cyl_radius * 2))
    # print(path[:,0])
    
    # 패스 확장 함수 호출
    expanded_path_left, expanded_path_right = expand_path(path)
    Boundary = getBoundary(expanded_path_left, expanded_path_right)

    # 맵 크기 설정 (path 데이터에 따라 크기를 조정)
    map_size = max(rows, cols)  # path 데이터 크기에 따라 조정

    # 장애물 맵 생성 (모든 위치를 장애물로 초기화)
    obstacle_map = np.ones((rows, cols))
    # Boundary 내부를 자유 공간으로 설정
    for i in range(rows):
        for j in range(cols):
            if (isInside(Boundary, [i, j])):
                obstacle_map[i, j] = 0
            if  (i == rows+free_space+1):
                obstacle_map[i, j] = 1

    before_map = obstacle_map
    # print(len(obstacle_map))
    obstacle_map = ObstacleMap(obstacle_map,before_map,Boundary ,0.5, smooth_iter=10, fill_threshold=8, clear_threshold=2)
    # print(len(obstacle_map))
    obstacle_map()
    obstacle_map=obstacle_map.to_numpy()
    # print("obstacle_map : ", obstacle_map)

    after_map = obstacle_map.copy()

    for r in range(rows):
        for c in range(cols):
            if (after_map[r][c] == before_map[r][c]) and (after_map[r][c] == before_map[r][c]):
                after_map[r][c] = False

    # print("after_map : ", after_map)
    # print("obstacle_map : ", obstacle_map)
    obstacle_map = Obstacle_Expand(8,obstacle_map,before_map,after_map,Boundary,rows,cols)
                           
                        
                        
    
    # .world 파일로 저장
    world_writer = WorldWriter(world_file, obstacle_map, cyl_radius=0.075, contain_wall_length=5)
    world_writer()

    print('World file saved as', world_file)

    # # ArbitPoint를 통해 경계 내부 점 확인
    # points = []
    # while True:
    #     try:
    #         ArbitPoint_X = float(input("임의의 X값 입력 : "))
    #         ArbitPoint_Y = float(input("임의의 Y값 입력 : "))
    #         ArbitPoint = [ArbitPoint_X, ArbitPoint_Y]
    #         points.append(ArbitPoint)
    #         # print(Boundary)
    #         plt.figure(figsize=(5, 5))
    #         plt.plot(path[:, 0], path[:, 1], 'bo-', label='Path')
    #         plt.plot(Boundary[:, 0], Boundary[:, 1], 'go--')
    #         for pt in points:
    #             if isInside(Boundary, pt):
    #                 print("자유공간 : True")
    #                 plt.plot(pt[0], pt[1], 'ko', label='Inner Point')
    #             else:
    #                 print("자유공간 : False")
    #                 plt.plot(pt[0], pt[1], 'rx', label='Outer Point')
    #         plt.xlabel('X')
    #         plt.ylabel('Y')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.title('Path Expansion Visualization')
    #         plt.show()
    #     except ValueError:
    #         print("잘못된 입력입니다. 다시 시도하세요.")
    #     except KeyboardInterrupt:
    #         print("\n프로그램을 종료합니다.")
    #         break

if __name__ == "__main__":
    for i in range (300):
        main(iteration=i, seed=0, fill_pct=0.2, smooth_iter=4)

