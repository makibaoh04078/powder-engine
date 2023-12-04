import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# 定数
dt = 0.01                         # 時間刻み幅
Lx = 20
Ly = 20
nx = 101
ny = 101
nt = 100                          # 計算ステップ数

#物体に関する定数
m = 1.0                       # 質量
q = 1.0                      # 電荷量
k = 8.99e9                        # クーロン定数
R = 1.0
x_0 = 10                          # 初期のx座標
y_0 = 10                          # 初期のy座標
vx_0 = 0                          # 初期のx方向の速度
vy_0 = 0                          # 初期のy方向の速度
x,y = x_0,y_0                     # 初期位置
vx,vy = vx_0,vy_0                 # 初期速度


rx = np.linspace(0, Lx, nx)  # x座標
ry = np.linspace(0, Ly, ny)  # y座標
X, Y = np.meshgrid(rx, ry)

charges = [{'q': -0.2, 'pos': np.array([-5000, -5000])},  # 負電荷
           {'q': -0.2, 'pos': np.array([5000,  -5000])},   # 負電荷
           {'q': -0.2, 'pos': np.array([-5000, 5000])}, # 負電荷
           {'q': 1.0, 'pos': np.array([5000, 5000])}]  # 正電荷

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

dx = 2*Lx/(nx-1)
dy = 2*Ly/(ny-1)

# グラフを作成
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
patch = Circle(xy=(x, y), radius=R)
im = ax.add_patch(patch)
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_aspect('equal')

def init():
    global x,y,vx,vy
    x,y = x_0,y_0
    vx,vy = vx_0,vy_0
    im.set_center((x,y))
    return im,

#ルンゲクッタ法
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
    F_x = q * E[0] / m
    F_y = q * E[1] / m
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

ani.save("Charged_particle_motion01.gif", writer='pillow')
plt.show()