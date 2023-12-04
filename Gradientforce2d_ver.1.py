import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from matplotlib import animation, rc
from IPython.display import HTML
import math
from matplotlib.ticker import FormatStrFormatter
# 定数
dt = 0.01                         # 時間刻み幅
Lx=10
Ly=10
nx=100
ny=100
nt = 100                          # 計算ステップ数

m, eps_0, eps_r, R = 1.0, 1.0, 3.0, 1.0
x_0 = 0                           # 初期のx座標
y_0 = 0                           # 初期のy座標
vx_0 = 0                          # 初期のx方向の速度
vy_0 = 0                          # 初期のy方向の速度
x,y = x_0,y_0                     # 初期位置
vx,vy = vx_0,vy_0                 # 初期速度

k = 9 * 10**9  # 静電定数
rx = np.linspace(-Lx, Lx, nx+1)  # x座標
ry = np.linspace(-Ly, Ly, ny+1)  # y座標
dx = 2*Lx/nx
dy = 2*Ly/ny
X, Y = np.meshgrid(rx, ry)

# 点電荷のリスト
charges = [{'q': -0.5, 'pos': np.array([-10000, 10000])},  # 負電荷
           {'q': -0.5, 'pos': np.array([-10000, -10000])},  # 負電荷
           {'q': -0.5, 'pos': np.array([10000, -10000])}, # 負電荷
           {'q': 1.0, 'pos': np.array([10000, 10000])}]  # 正電荷

# 電場の計算
E_x = np.zeros_like(X)
E_y = np.zeros_like(Y)
for charge in charges:
    r_x = X - charge['pos'][0]
    r_y = Y - charge['pos'][1]
    r = np.sqrt(r_x**2 + r_y**2)
    E_x += k * charge['q'] * r_x / r**3
    E_y += k * charge['q'] * r_y / r**3

E = np.sqrt(E_x**2 + E_y**2)

# 電場ベクトル場の描画
plt.figure(figsize=(6,4))
strm = plt.streamplot(X, Y, E_x, E_y, color=E_x**2 + E_y**2, cmap='hot')
plt.title('Electric field^2 streamlines')
plt.xlabel('x')
plt.ylabel('y')
cbar = plt.colorbar(strm.lines, label='Electric field^2 strength')
cbar.formatter = FormatStrFormatter("%.2f")
cbar.update_ticks()
plt.grid()
plt.show()

# グラフを作成
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
patch = Circle(xy=(x, y), radius=R)
im = ax.add_patch(patch)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')

def init():
    global x,y,vx,vy
    x,y = x_0,y_0
    vx,vy = vx_0,vy_0
    im.set_center((x,y))
    return im,

def inner_indexes(x,y):
    # 円内に存在するグリッドの数を計算
    grid_in_circle = np.sum((X - x)**2 + (Y - y)**2 <= R**2)
    # 物体内に存在するグリッドのインデックスを取得
    indexes = np.where((X - x)**2 + (Y - y)**2 <= R**2)

    return grid_in_circle,indexes

def grad_E2_func(x, y, E_x, E_y, E):
    grids,indexes = inner_indexes(x,y) #対象のグリッド数とグリッドの座標を代入

    # 円内に存在するグリッドの座標(i,j)の電場をひとつづつ取り出して二乗の勾配を計算
    grad_x, grad_y = np.zeros_like(E), np.zeros_like(E)
    grad_x_sum, grad_y_sum = 0, 0  # 勾配のx成分とy成分の合計
    for i, j in zip(*indexes):
        E_squared = E[i,j]**2
        if i > 0 and i < nx-1 and j > 0 and j < ny-1:  # 境界を除く
            grad_x[i,j] = (E[i+1,j]**2 - E[i-1,j]**2) / (2*dx)
            grad_y[i,j] = (E[i,j+1]**2 - E[i,j-1]**2) / (2*dy) 
            grad_x_sum += grad_x[i,j]
            grad_y_sum += grad_y[i,j]

    # 勾配の平均を計算
    grad_x_avg = grad_x_sum / grids
    grad_y_avg = grad_y_sum / grids

    return grad_x_avg, grad_y_avg

def runge_kutta(x, y, vx, vy, F_x, F_y, dt):
    k1_x = dt * vx
    k1_y = dt * vy
    k1_vx = dt * F_x
    k1_vy = dt * F_y

    k2_x = dt * (vx + 0.5 * k1_vx)
    k2_y = dt * (vy + 0.5 * k1_vy)
    k2_vx = dt * (F_x + 0.5 * k1_vx)
    k2_vy = dt * (F_y + 0.5 * k1_vy)

    k3_x = dt * (vx + 0.5 * k2_vx)
    k3_y = dt * (vy + 0.5 * k2_vy)
    k3_vx = dt * (F_x + 0.5 * k2_vx)
    k3_vy = dt * (F_y + 0.5 * k2_vy)

    k4_x = dt * (vx + k3_vx)
    k4_y = dt * (vy + k3_vy)
    k4_vx = dt * (F_x + k3_vx)
    k4_vy = dt * (F_y + k3_vy)

    x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
    y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
    vx += (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
    vy += (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6

    return x, y, vx, vy

def animate(frame):
    global x, y, vx, vy
    grad_E2_x,grad_E2_y = grad_E2_func(x, y, E_x, E_y, E)
    F_x = 2 * math.pi * eps_0 * (eps_r-1) * (R**3) * grad_E2_x / (m * (eps_r + 2))
    F_y = 2 * math.pi * eps_0 * (eps_r-1) * (R**3) * grad_E2_y / (m * (eps_r + 2))
    x, y, vx, vy = runge_kutta(x, y, vx, vy, F_x, F_y, dt)
    im.set_center((x, y))
    return im,

ani = FuncAnimation(
    fig,  # Figureオブジェクト
    animate,  # グラフ更新関数
    init_func=init,
    frames = np.arange(0, 10, dt),  # フレームを設定
    interval = int(dt*1000),  # 更新間隔(ms)
    repeat = True,  # 描画を繰り返す
    blit = True,
    save_count = 500,
    )

ani.save("gradient_2dim(R=1.0,-0.5,-0.5,-0.5,1.0).gif", writer='pillow')
plt.show()