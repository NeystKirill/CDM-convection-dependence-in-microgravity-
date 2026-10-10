import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from emulator.common.history import Recorder

st.set_page_config(page_title="Flame in Freefall", layout="wide")
st.title("Flame in Freefall — эмулятор пламени в микрогравитации")

LEVELS = {
    "1D — сферическое пламя": {"glob": "sflame_*.yaml", "module": "emulator.reference"},
    "2D — свеча в потоке (осесимм.)": {"glob": "candle_l2_*.yaml", "module": "level2"},
    "3D — свеча в потоке": {"glob": "candle_l3_*.yaml", "module": "level3"},
}

config_dir = Path(__file__).resolve().parent.parent / "configs"

with st.sidebar:
    st.header("Уровень")
    level_name = st.radio("Размерность модели", list(LEVELS.keys()))
    level = LEVELS[level_name]

    st.header("Режим")
    config_files = sorted(config_dir.glob(level["glob"]))
    config_names = [f.name for f in config_files]
    selected = st.selectbox("Конфиг (configs/*.yaml)", config_names)
    run_button = st.button("Запустить")
    save_history = st.checkbox("Сохранить историю прогона", value=True)
    st.caption(
        "Все физические и численные параметры задаются в YAML-файле конфига, "
        "не в интерфейсе. Чтобы изменить режим — отредактируйте или добавьте файл в configs/."
    )

module = __import__(level["module"], fromlist=["Params", "Solver"])
Params, Solver = module.Params, module.Solver

config_path = config_dir / selected
params = Params.from_yaml(str(config_path))

with st.expander("Параметры выбранного конфига", expanded=True):
    cols = st.columns(4)
    cols[0].write(f"**Топливо:** {params.fuel}")
    cols[0].write(f"**T_inf:** {params.T_inf:.0f} K")
    cols[1].write(f"**X_O2,∞:** {params.XO2_inf:.3f}")
    cols[1].write(f"**p:** {params.p/101325:.2f} атм")
    cols[2].write(f"**B / Ta:** {params.B:.2e} / {params.Ta:.0f}")
    cols[2].write(f"**Qcp / s:** {params.Qcp:.0f} / {params.s:.2f}")
    cols[3].write(f"**t_end:** {params.t_end:.2f} с")
    if hasattr(params, "g_frac"):
        cols[3].write(f"**g_frac / U_ext:** {params.g_frac:.2f} / {params.U_ext:.3f} м/с")


def draw_level1(solver):
    state = solver.state()
    q = solver.qoi()

    fig, (ax_ball, ax_profile) = plt.subplots(1, 2, figsize=(10, 4))

    r_f_mm = q["r_f"] * 1000.0
    r_out_mm = solver.grid.r_face[-1] * 1000.0
    color = plt.cm.inferno(min(q["T_max"] / 2500.0, 1.0))
    ax_ball.set_xlim(-r_out_mm * 0.3, r_out_mm * 0.3)
    ax_ball.set_ylim(-r_out_mm * 0.3, r_out_mm * 0.3)
    ax_ball.add_patch(plt.Circle((0, 0), r_f_mm, color=color))
    ax_ball.add_patch(plt.Circle((0, 0), solver.params.Rb * 1000.0, color="gray"))
    ax_ball.set_aspect("equal")
    ax_ball.set_title(f"t = {solver.t:.3f} с, r_f = {r_f_mm:.1f} мм")

    ax_profile.plot(state["r"] * 1000.0, state["T"])
    ax_profile.set_xlabel("r, мм")
    ax_profile.set_ylabel("T, К")
    ax_profile.set_title("Температурный профиль")

    fig.tight_layout()
    return fig, q


def draw_level2(solver):
    state = solver.state()
    q = solver.qoi()

    fig, ax = plt.subplots(figsize=(5, 8))
    cs = ax.contourf(state["r"] * 1000.0, state["z"] * 1000.0, state["T"], levels=30, cmap="inferno")
    fig.colorbar(cs, ax=ax, label="T, К")
    ax.set_xlabel("r, мм")
    ax.set_ylabel("z, мм")
    ax.set_title(f"t = {solver.t:.3f} с, T_max = {q['T_max']:.0f} К, Da = {q['da']:.2f}")
    fig.tight_layout()
    return fig, q


def draw_level3(solver):
    state = solver.state()
    q = solver.qoi()

    j_mid = params.ny // 2
    fig, ax = plt.subplots(figsize=(5, 8))
    cs = ax.contourf(state["x"] * 1000.0, state["z"] * 1000.0, state["T"][:, j_mid, :], levels=30, cmap="inferno")
    fig.colorbar(cs, ax=ax, label="T, К")
    ax.set_xlabel("x, мм")
    ax.set_ylabel("z, мм")
    ax.set_title(f"t = {solver.t:.3f} с, срез y = {state['y'][j_mid]*1000:.1f} мм, T_max = {q['T_max']:.0f} К")
    fig.tight_layout()
    return fig, q


DRAWERS = {
    "1D — сферическое пламя": draw_level1,
    "2D — свеча в потоке (осесимм.)": draw_level2,
    "3D — свеча в потоке": draw_level3,
}

if run_button:
    solver = Solver(params)
    recorder = Recorder(solver)
    draw_frame = DRAWERS[level_name]

    plot_slot = st.empty()
    metrics_slot = st.empty()

    steps_per_frame = 20
    while solver.t < params.t_end:
        for _ in range(steps_per_frame):
            if solver.t >= params.t_end:
                break
            solver.step()
            if save_history:
                recorder.capture()

        fig, q = draw_frame(solver)
        plot_slot.pyplot(fig)
        plt.close(fig)

        cols = metrics_slot.columns(5)
        cols[0].metric("T_max, К", f"{q['T_max']:.0f}")
        cols[1].metric("χ_rad", f"{q['chi_rad']:.2f}")
        cols[2].metric("η", f"{q['eta']:.2f}")
        cols[3].metric("Режим", q["regime"])
        if "da" in q:
            cols[4].metric("Da", f"{q['da']:.2f}")

    st.success(f"Расчёт завершён: {solver.n_steps} шагов, t = {solver.t:.3f} с")

    if save_history:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_{selected.replace('.yaml', '')}_{int(time.time())}.h5"
        recorder.to_hdf5(out_path)
        st.info(f"История сохранена: {out_path}")
else:
    st.write("Выберите уровень и конфиг слева, затем нажмите «Запустить».")
