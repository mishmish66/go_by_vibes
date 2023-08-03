using Symbolics
using LinearAlgebra
using Latexify

rotmat(θ) =
    [
        cos(θ) -sin(θ)
        sin(θ) cos(θ)
    ]

dot_self(x) = dot(x, x)

@variables θ_body x_body y_body q_lhip q_lknee q_rhip q_rknee θ_body_dot x_body_dot y_body_dot q_lhip_dot q_lknee_dot q_rhip_dot q_rknee_dot θ_body_ddot x_body_ddot y_body_ddot q_lhip_ddot q_lknee_ddot q_rhip_ddot q_rknee_ddot m_body m_thigh m_shank I_body I_thigh I_shank l_body l_thigh l_shank g

q = [θ_body x_body y_body q_lhip q_lknee q_rhip q_rknee]'
q_dot = [θ_body_dot x_body_dot y_body_dot q_lhip_dot q_lknee_dot q_rhip_dot q_rknee_dot]'
q_ddot = [θ_body_ddot x_body_ddot y_body_ddot q_lhip_ddot q_lknee_ddot q_rhip_ddot q_rknee_ddot]'
ddt(x) = (Symbolics.jacobian(x, q) * q_dot + Symbolics.jacobian(x, q_dot) * q_ddot)

r_body = [x_body, y_body]
θ_lthigh = θ_body + q_lhip
θ_rthigh = θ_body + q_rhip

u_hip_lknee = rotmat(θ_lthigh) * [0; -1]
u_hip_rknee = rotmat(θ_rthigh) * [0; -1]

θ_lknee = θ_lthigh + q_lknee
θ_rknee = θ_rthigh + q_rknee

u_lknee_lfoot = rotmat(θ_lknee) * [0; -1]
u_rknee_rfoot = rotmat(θ_rknee) * [0; -1]

u_body_com_head = rotmat(θ_body) * [0; 1]

r_hip = r_body - l_body * u_body_com_head
r_lknee = r_hip + u_hip_lknee * l_thigh
r_rknee = r_hip + u_hip_rknee * l_thigh
r_lfoot = r_lknee + u_lknee_lfoot * l_shank
r_rfoot = r_rknee + u_rknee_rfoot * l_shank

r_lthigh_com = r_hip + u_hip_lknee * l_thigh / 2
r_lshank_com = r_lknee + u_lknee_lfoot * l_shank / 2

r_rthigh_com = r_hip + u_hip_rknee * l_thigh / 2
r_rshank_com = r_rknee + u_rknee_rfoot * l_shank / 2

T_body = -m_body * g * y_body
T_legs = -g * (m_thigh * (r_rthigh_com[2] + r_lthigh_com[2]) + m_shank * (r_rshank_com[2] + r_lshank_com[2]))
T = T_body + T_legs

V_body_angular = 1 / 2 * I_body * (ddt([θ_body])[1])^2
V_body_linear = 1 / 2 * m_body * dot_self(ddt(r_body))
V_body = V_body_angular + V_body_linear

V_leg_linear = 1 / 2 * (m_thigh * (dot_self(ddt(r_lthigh_com)) + dot_self(ddt(r_rthigh_com))) + m_shank * (dot_self(ddt(r_lshank_com)) + dot_self(ddt(r_rshank_com))))
V_leg_angular = 1 / 2 * (I_thigh * ((ddt([θ_lthigh])[1])^2 + (ddt([θ_rthigh])[1])^2) + I_shank * ((ddt([θ_lknee])[1])^2 + (ddt([θ_rknee])[1])^2))
V_leg = V_leg_linear + V_leg_angular

V = V_body + V_leg

L = T - V
L = simplify(L)

dL_dq = Symbolics.jacobian([L], q)'
dL_dqdot = Symbolics.jacobian([L], q_dot)'
ddL_dqdot_dt = ddt(dL_dqdot)

f = ddL_dqdot_dt - dL_dq
f = simplify(f)

mass_matrix = Symbolics.jacobian(f, q_ddot)
mass_matrix = simplify(mass_matrix)

acceleration_invariant_forces = f - (mass_matrix * q_ddot)
acceleration_invariant_forces = simplify(acceleration_invariant_forces)

velocity_effects = substitute(acceleration_invariant_forces, Dict([g => 0.0]))
velocity_effects = simplify(velocity_effects)

gravity_effects = acceleration_invariant_forces - velocity_effects
gravity_effects = simplify(gravity_effects)

mass_syms = [m_body m_thigh m_shank I_body I_thigh I_shank]
shape_syms = [l_body, l_thigh, l_shank]

mass_matrix_cfunc = build_function(mass_matrix, q, q_dot, q_ddot, mass_syms, shape_syms, g; target=Symbolics.CTarget())

open("mass_matrix.c", "w") do file
    write(file, mass_matrix_cfunc)
end

other_forces_cfunc = build_function(acceleration_invariant_forces, q, q_dot, q_ddot, mass_syms, shape_syms, g; target=Symbolics.CTarget())

open("other_forces.c", "w") do file
    write(file, other_forces_cfunc)
end

velocity_effects_cfunc = build_function(velocity_effects, q, q_dot, q_ddot, mass_syms, shape_syms, g; target=Symbolics.CTarget())

open("velocity_effects.c", "w") do file
    write(file, velocity_effects_cfunc)
end

gravity_effects_cfunc = build_function(gravity_effects, q, q_dot, q_ddot, mass_syms, shape_syms, g; target=Symbolics.CTarget())

open("gravity_effects.c", "w") do file
    write(file, gravity_effects_cfunc)
end