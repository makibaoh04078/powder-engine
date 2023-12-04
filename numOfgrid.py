import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib import animation, rc
from IPython.display import HTML
import math
from matplotlib.ticker import FormatStrFormatter

# 定数
dt = 0.01                         # 時間刻み幅
nx=100
ny=100
nt = 100                          # 計算ステップ数

x = np.linspace(-10, 10, nx)  # x座標
y = np.linspace(-10, 10, ny)  # y座標
X, Y = np.meshgrid(x, y)

R = 0.4  # 円の半径
center_x = np.random.uniform(-10, 10)  # 円の中心のx座標
center_y = np.random.uniform(-10, 10)  # 円の中心のy座標

fig, ax = plt.subplots()  # グラフの作成
circle = Circle((center_x, center_y), R, fill=False)  # 円の作成
ax.add_patch(circle)  # 円をグラフに追加
ax.set_aspect('equal')  # グラフを正方形にする

indices = np.where((X - center_x)**2 + (Y - center_y)**2 <= R**2)  # 円内に存在するグリッドのインデックス
print(f'Grid indices in circle: {indices}')  # グリッドのインデックスを表示

num_grids = len(indices[0])  # 円内に存在するグリッドの数
print(f'Number of grids in circle: {num_grids}')  # グリッドの数を表示
plt.text(0, 0, f'Number of grids: {num_grids}', fontsize=12)  # グラフにグリッドの数を表示

for i, j in zip(*indices):  # 円内に存在するグリッドの座標をひとつずつ表示
    print(f'Grid coordinates: ({X[i, j]}, {Y[i, j]})')

plt.show()  # グラフを表示





