import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# 定数
dt = 0.01                         # 時間刻み幅
Lx = 20
Ly = 20
nx = 101
ny = 101
nt = 100                          # 計算ステップ数
k = 8.99e9                        # クーロン定数

rx = np.linspace(0, Lx, nx)  # x座標
ry = np.linspace(0, Ly, ny)  # y座標
X, Y = np.meshgrid(rx, ry)
dx = 2*Lx/(nx-1)
dy = 2*Ly/(ny-1)
#点電荷のリスト

# x軸正の向き
'''
charges = [{'q': -0.2, 'pos': np.array([-5000, -5000])},  # 負電荷
           {'q': 1.0, 'pos': np.array([2500,  -2500])},   # 負電荷
           {'q': -0.2, 'pos': np.array([-5000, 5000])}, # 負電荷
           {'q': 1.0, 'pos': np.array([2500, 2500])}]  # 正電荷
'''
# y軸正の向き
'''
charges = [{'q': -0.1, 'pos': np.array([0, -5000])},  # 負電荷
           {'q': 0.1, 'pos': np.array([0,  2500])},   # 正電荷
           {'q': 0, 'pos': np.array([-5000, 5000])}, # 負電荷
           {'q': 0, 'pos': np.array([2500, 2500])}]  # 正電荷
'''
# 第一象限の向き

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

# 電場ベクトル場の描画(Cursorでは不要)
'''
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
'''
#任意の数の物体を設定
num_particles = 2  # 物体の数を設定

particles = []
for i in range(num_particles):
    particle = {
        "id": i,  # 物体にIDを付ける
        "m": 1.0,
        "q": -1.0 if i % 2 == 0 else 1.0,  # 偶数IDの物体には負の電荷、奇数IDの物体には正の電荷を付ける
        "R": 1.0,
        "x": 10 ,  # 初期位置をIDに基づいて設定
        "y": 8 + i*4,  # 初期位置をIDに基づいて設定
        "vx": 0,
        "vy": 0,
        "color": "blue" if i % 2 == 0 else "red",  # 偶数IDの物体は青、奇数IDの物体は赤にする
    }
    particles.append(particle)

# 物体の初期位置と初速度を保存
for p in particles:
    p["x_initial"] = p["x"]
    p["y_initial"] = p["y"]
    p["vx_initial"] = p["vx"]
    p["vy_initial"] = p["vy"]


# パッチと物体の情報を紐づける辞書を作成
patches = {i: Circle(xy=(p["x"], p["y"]), radius=p["R"], color=p["color"]) for i, p in enumerate(particles)}


# グラフを作成
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)

# グラフにパッチを追加
for patch in patches.values():
    ax.add_patch(patch)
    
ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_aspect('equal')

# 初期化関数
def init():
    for p, patch in zip(particles, patches.values()):
        p["x"], p["y"] = p["x_initial"], p["y_initial"]  # 初期位置を物体の初期位置に設定
        p["vx"], p["vy"] = p["vx_initial"], p["vy_initial"]  # 初速度を物体の初速度に設定
        patch.set_center((p["x"], p["y"]))
    return list(patches.values())  

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

#物体の中心を囲む４つのグリッドのインデックスを取得するために使用する関数。
def findGrid(x, y, dx, dy):
    i = min(int(x / dx), nx-1)
    j = min(int(y / dy), ny-1)
    return i, j

#物体を囲む四つのグリッドの電場の平均を計算
def calc_E(x, y, E_x, E_y, dx, dy):
    Ex_ave, Ey_ave = 0, 0

    # グリッドの座標を取得
    i,j = findGrid(x, y, dx, dy)

    if 0 <= i and i+1 < nx and 0 <= j and j+1 < ny:  # インデックスが範囲内にあることを確認
        #4つの電場を足して平均をとる。
        Ex_ave = ( E_x[j,i] + E_x[j+1,i] + E_x[j,i+1] + E_x[j+1,i+1] ) / 4
        Ey_ave = ( E_y[j,i] + E_y[j+1,i] + E_y[j,i+1] + E_y[j+1,i+1] ) / 4

    return Ex_ave, Ey_ave

# アニメーション関数
def animate(frame):
    for p, patch in zip(particles, patches.values()):
        
        #物体を囲む四つのグリッドの電場の平均を計算
        Ex, Ey = calc_E(p["x"],p["y"],E_x,E_y,dx,dy)
        
        # 力を計算
        F_x = p["q"] * Ex / p["m"]
        F_y = p["q"] * Ey / p["m"]
        
        # ルンゲクッタ法で次の位置と速度を計算
        x_next, y_next, vx_next, vy_next = runge_kutta(p["x"], p["y"], p["vx"], p["vy"], F_x, F_y, dt)
        
        # 次の位置が範囲内に収まるように制限
        x_next = max(0, min(Lx, x_next))
        y_next = max(0, min(Ly, y_next))
        
        # パッチの位置を更新
        p["x"], p["y"], p["vx"], p["vy"] = x_next, y_next, vx_next, vy_next
        patch.set_center((p["x"], p["y"]))
        
    # 剛体の長さを保つための調整
    ddx = particles[1]["x"] - particles[0]["x"]
    ddy = particles[1]["y"] - particles[0]["y"]
    distance = np.sqrt(ddx**2 + ddy**2)
    rigid_length = 4  # 剛体の長さ
    if distance > rigid_length:
        correction = (distance - rigid_length) / 2
        angle = np.arctan2(ddy, ddx)
        particles[0]["x"] += correction * np.cos(angle)
        particles[0]["y"] += correction * np.sin(angle)
        particles[1]["x"] -= correction * np.cos(angle)
        particles[1]["y"] -= correction * np.sin(angle)
        patches[0].set_center((particles[0]["x"], particles[0]["y"]))
        patches[1].set_center((particles[1]["x"], particles[1]["y"]))
        
    return list(patches.values())  # リストに変換

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

ani.save("charged_particle_motion02.gif", writer='pillow')
plt.show()