import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.linalg as la
import matplotlib.pyplot as plt

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Web UI Setup) & Storytelling
# ==========================================
st.set_page_config(page_title="AGV Control Simulation", layout="wide")

st.title("🤖 หุ่นยนต์ส่งของอัจฉริยะ (Self-Balancing AGV)")

# เพิ่มส่วน Storytelling 
st.markdown("""
### ภารกิจ: พนักงานส่งของอัจฉริยะแห่งโรงงานชิป (The Smart Delivery Bot)
ลองจินตนาการว่าคุณเป็นผู้จัดการโรงงานผลิตชิปคอมพิวเตอร์ (Semiconductor Fab) สินค้าของคุณคือ "แผ่นเวเฟอร์ (Wafer)" ที่บอบบางมาก เป็นรอยนิดเดียวคือพังทั้งแผ่น!

ปัญหาคือ เราต้องขนส่งกล่องใส่เวเฟอร์พวกนี้ไปตามเครื่องจักรต่างๆ ในห้องคลีนรูม (ห้องที่สะอาดกว่าห้องผ่าตัด) รถเข็นแบบ 4 ล้อทั่วไปก็เทอะทะ วงเลี้ยวแคบๆ ก็ลำบาก เราจึงสร้าง **"หุ่นยนต์สองล้อ (Self-Balancing AGV)"** ขึ้นมา

**ทำไมต้อง 2 ล้อ?** เพราะมันหมุนตัวกลับหลังหันได้ในที่แคบๆ (เหมือนเรายืนหมุนตัว) แต่ปัญหาใหญ่ที่สุดของรถ 2 ล้อคือ... **"มันล้มง่ายมาก!"** (ลองพยายามเอาไม้บรรทัดตั้งบนนิ้วมือดูสิครับ ถ้านิ้วไม่ขยับ ไม้บรรทัดก็ร่วง) ระบบที่ไม่มีความสมดุลในตัวเองแบบนี้ วิศวกรเรียกว่า *Open-loop Unstable* ซึ่งเขียนเป็นสมการความเปลี่ยนแปลงของสถานะ (State-space) ได้ว่า:
$$ \dot{x} = A_cx + B_cu $$

เราจึงต้องมี "สมองกล (Controller)" มาคอยสั่งมอเตอร์ให้ขยับเดินหน้า-ถอยหลังเพื่อเลี้ยงสมดุลตลอดเวลา!

### 🧠 สมองกลทำงานยังไง? (The Magic of Control System)
การจะทำให้รถสองล้อวิ่งไปข้างหน้าโดยไม่ทำของหล่นแตก เราใช้คณิตศาสตร์ 2 สูตรมาช่วย คือ:

**1. DLQR (The Master Balancer)** 
เวลาขี่จักรยาน เราต้องรักษาสมดุล 2 เรื่องพร้อมกัน คือต้องปั่นให้ถึงจุดหมาย (คุมตำแหน่ง) และต้องไม่ล้ม! (คุมมุมเอียง) 

DLQR คือ "คนขี่จักรยานที่เก่งที่สุด" มันจะคำนวณหาวิธีจ่ายไฟให้มอเตอร์แบบพอดีเป๊ะ! โดยลดค่าความผิดพลาดและการใช้พลังงานให้เหลือน้อยที่สุด ผ่านฟังก์ชันต้นทุน (Cost Function) นี้:
$$ J = \sum_{k=0}^{\infty} (x_k^T Q x_k + u_k^T R u_k) $$
*(อธิบายง่ายๆ: ตัว Q คือโดนหักคะแนนถ้าหุ่นยนต์เอียงหรือไปไม่ตรงจุด ส่วนตัว R คือโดนหักคะแนนถ้ามอเตอร์กินไฟเยอะเกินไป)*

เมื่อหาจุดที่เสียคะแนนน้อยที่สุดได้แล้ว สมองกลก็จะสั่งการมอเตอร์ ($u_k$) ด้วยสมการ:
$$ u_k = -K x_k $$

**2. Luenberger Observer (The Blindfolded Ninja)**
ในชีวิตจริง หุ่นยนต์อาจจะวัดความเร็วไม่ได้เป๊ะๆ (เซนเซอร์วัดได้แค่ระยะทางกับมุมเอียง) 
Observer เปรียบเสมือนนินจาที่ถูกปิดตา แต่หูไวมาก! มันสามารถใช้คณิตศาสตร์ "คาดเดา" ความเร็วได้อย่างแม่นยำภายในเสี้ยววินาที ด้วยสมการนี้:
$$ \hat{x}_{k+1} = A\hat{x}_k + Bu_k + L(y_k - C\hat{x}_k) $$
*(อธิบายง่ายๆ: ตัว y คือสิ่งที่เซนเซอร์มองเห็น และตัว x-hat คือความเร็วที่นินจาคาดเดาขึ้นมา)*

---
### 🎮 ลองเล่นเป็นผู้จัดการโรงงาน!
ด้านซ้ายมือของคุณคือแผงควบคุม ลองปรับค่าต่างๆ ดูสิ:
*   **Q: น้ำหนักมุมเอียง:** ยิ่งปรับสูง หุ่นยนต์จะยิ่งเกร็งตัวตรงเป๊ะเหมือนทหาร!
*   **R: ข้อจำกัดมอเตอร์:** ถ้าน้อยไป หุ่นยนต์จะกระชากไฟแรงจนมอเตอร์อาจพัง (สังเกตแถบเตือนสีเหลือง!)

มาดูกันว่าคุณจะหาค่าที่เหมาะสม (Sweet spot) ให้หุ่นยนต์วิ่งไปส่งของได้ปลอดภัยที่สุดได้ไหม!
""")

# ==========================================
# 2. แถบเครื่องมือด้านข้าง (Sidebar Controls)
# ==========================================
st.sidebar.header("⚙️ ปรับแต่งพารามิเตอร์ตัวควบคุม")

st.sidebar.subheader("DLQR Weights")
st.sidebar.markdown("ปรับค่าน้ำหนักเพื่อดูผลกระทบต่อการตอบสนองและการใช้พลังงาน")
q_theta = st.sidebar.slider("Q: น้ำหนักมุมเอียง (Theta)", min_value=100, max_value=2000, value=1000, step=100)
q_p = st.sidebar.slider("Q: น้ำหนักตำแหน่ง (Position)", min_value=10, max_value=500, value=100, step=10)
r_weight = st.sidebar.slider("R: ข้อจำกัดมอเตอร์ (Control Effort)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)

st.sidebar.subheader("Initial Conditions")
init_angle = st.sidebar.slider("มุมเอียงเริ่มต้น (rad)", min_value=-0.2, max_value=0.2, value=0.1, step=0.01)

# ==========================================
# 3. แบบจำลองทางคณิตศาสตร์ (System Modeling)
# ==========================================
M, m, l, I, b, g = 5.0, 2.0, 0.3, 0.006, 0.1, 9.81
p_den = I*(M + m) + M*m*l**2

Ac = np.array([
    [0, 1, 0, 0],
    [0, -(I + m*l**2)*b/p_den, (m**2 * g * l**2)/p_den, 0],
    [0, 0, 0, 1],
    [0, -(m*l*b)/p_den, m*g*l*(M + m)/p_den, 0]
])
Bc = np.array([[0], [(I + m*l**2)/p_den], [0], [m*l/p_den]])
Cc = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
Dc = np.array([[0], [0]])

Ts = 0.01
sys_c = signal.StateSpace(Ac, Bc, Cc, Dc)
sys_d = sys_c.to_discrete(Ts)
A, B, C = sys_d.A, sys_d.B, sys_d.C

# ==========================================
# 4. ออกแบบตัวควบคุมและตัวประมาณค่า (Controller & Observer)
# ==========================================
Q = np.diag([q_p, 1, q_theta, 10])
R = np.array([[r_weight]])
P = la.solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

observer_poles = np.array([0.4, 0.45, 0.5, 0.55])
place_obj = signal.place_poles(A.T, C.T, observer_poles)
L = place_obj.gain_matrix.T

# ==========================================
# 5. ซิมูเลชัน (Closed-Loop Simulation)
# ==========================================
N = 400
x = np.array([[0.0], [0.0], [init_angle], [0.0]])
x_hat = np.array([[0.0], [0.0], [0.0], [0.0]])
x_ref = np.array([[1.0], [0.0], [0.0], [0.0]])

x_hist, x_hat_hist, u_hist = np.zeros((4, N)), np.zeros((4, N)), np.zeros((1, N))

for k in range(N):
    y = C @ x
    u = -K @ (x_hat - x_ref)
    
    x_hist[:, k] = x.flatten()
    x_hat_hist[:, k] = x_hat.flatten()
    u_hist[:, k] = u.flatten()
    
    x = A @ x + B @ u
    x_hat = A @ x_hat + B @ u + L @ (y - C @ x_hat)

# ==========================================
# 6. แสดงผลกราฟบนเว็บไซต์
# ==========================================
time = np.arange(N) * Ts
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

ax1.plot(time, x_hist[0, :], 'b-', label='Actual Position')
ax1.plot(time, x_hat_hist[0, :], 'r--', label='Estimated Position')
ax1.axhline(1.0, color='gray', linestyle=':', label='Target (1m)')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

ax2.plot(time, x_hist[2, :], 'g-', label='Actual Angle')
ax2.plot(time, x_hat_hist[2, :], 'orange', linestyle='--', label='Estimated Angle')
ax2.axhline(0, color='gray', linestyle=':')
ax2.set_ylabel('Angle (rad)')
ax2.legend()
ax2.grid(True)

ax3.plot(time, u_hist[0, :], 'purple', label='Control Effort (u)')
ax3.set_ylabel('Voltage (V)')
ax3.set_xlabel('Time (s)')
ax3.legend()
ax3.grid(True)

# ไฮไลต์เรื่องความปลอดภัย
max_u = np.max(np.abs(u_hist))
if max_u > 24.0:
    st.warning(f"⚠️ **แจ้งเตือนความปลอดภัย:** ระบบใช้แรงดันไฟฟ้าสูงสุดถึง {max_u:.2f} V ซึ่งอาจเกินขีดจำกัดของมอเตอร์ 24V ทั่วไป (Actuator Saturation) แนะนำให้ปรับเพิ่มค่า R ในแถบด้านข้าง")
else:
    st.success(f"✅ **ปลอดภัย:** ระบบใช้แรงดันไฟฟ้าสูงสุด {max_u:.2f} V (อยู่ในขีดจำกัดมอเตอร์ 24V)")

st.pyplot(fig)
