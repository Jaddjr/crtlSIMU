import streamlit as st
import numpy as np
import scipy.signal as signal
import scipy.linalg as la
import matplotlib.pyplot as plt

# ==========================================
# 1. ตั้งค่าหน้าเว็บ (Web UI Setup)
# ==========================================
st.set_page_config(page_title="AGV Control Simulation", layout="wide")
st.title("ระบบควบคุมหุ่นยนต์ล้อคู่ (Self-Balancing AGV) สำหรับขนส่งเวเฟอร์")
st.markdown("โปรเจกต์รายวิชา Modern Control Systems: **การประยุกต์ใช้ DLQR และ Luenberger Observer**")

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
# พารามิเตอร์ฟิสิกส์
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

# Discretization
Ts = 0.01
sys_c = signal.StateSpace(Ac, Bc, Cc, Dc)
sys_d = sys_c.to_discrete(Ts)
A, B, C = sys_d.A, sys_d.B, sys_d.C

# ==========================================
# 4. ออกแบบตัวควบคุมและตัวประมาณค่า (Controller & Observer)
# ==========================================
# DLQR
Q = np.diag([q_p, 1, q_theta, 10])
R = np.array([[r_weight]])
P = la.solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

# Luenberger Observer
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

# กราฟตำแหน่ง
ax1.plot(time, x_hist[0, :], 'b-', label='Actual Position')
ax1.plot(time, x_hat_hist[0, :], 'r--', label='Estimated Position')
ax1.axhline(1.0, color='gray', linestyle=':', label='Target (1m)')
ax1.set_ylabel('Position (m)')
ax1.legend()
ax1.grid(True)

# กราฟมุม
ax2.plot(time, x_hist[2, :], 'g-', label='Actual Angle')
ax2.plot(time, x_hat_hist[2, :], 'orange', linestyle='--', label='Estimated Angle')
ax2.axhline(0, color='gray', linestyle=':')
ax2.set_ylabel('Angle (rad)')
ax2.legend()
ax2.grid(True)

# กราฟสัญญาณควบคุม
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

st.markdown("""
---
**วิเคราะห์ผลการจำลอง:**
- **Observer:** สามารถประเมินค่ามุมและความเร็วที่มองไม่เห็นได้อย่างรวดเร็ว (เส้นประลู่เข้าหาเส้นทึบ)
- **Controller:** ระบบสามารถรักษาสมดุลของหุ่นยนต์ (มุมกลับมาที่ 0) และเคลื่อนที่ไปยังเป้าหมาย (1 เมตร) ได้สำเร็จ 
""")
