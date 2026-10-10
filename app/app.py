import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import streamlit as st

from emulator.reference import Params, Solver
from emulator.reference.history import Recorder

st.set_page_config(page_title="Flame in Freefall", layout="wide")
st.title("Flame in Freefall — эмулятор сферического пламени в микрогравитации")

config_dir = Path(__file__).resolve().parent.parent / "configs"
config_files = sorted(config_dir.glob("sflame_*.yaml"))
config_names = [f.name for f in config_files]

with st.sidebar:
    st.header("Режим")
    selected = st.selectbox("Конфиг (configs/*.yaml)", config_names)
    run_button = st.button("Запустить")
    save_history = st.checkbox("Сохранить историю прогона", value=True)
    st.caption(
        "Все физические и численные параметры задаются в YAML-файле конфига, "
        "не в интерфейсе. Чтобы изменить режим — отредактируйте или добавьте файл в configs/."
    )

config_path = config_dir / selected
params = Params.from_yaml(str(config_path))

with st.expander("Параметры выбранного конфига", expanded=True):
    cols = st.columns(4)
    cols[0].write(f"**Топливо:** {params.fuel}")
    cols[0].write(f"**ṁ:** {params.mdot*1e6:.2f} мг/с")
    cols[0].write(f"**Y_F,inj:** {params.YF_inj:.2f}")
    cols[1].write(f"**X_O2,∞:** {params.XO2_inf:.3f}")
    cols[1].write(f"**p:** {params.p/101325:.2f} атм")
    cols[1].write(f"**T_inf / T_b:** {params.T_inf:.0f} / {params.Tb:.0f} K")
    cols[2].write(f"**R_b / R_out:** {params.Rb*1000:.2f} / {params.Rout*1000:.1f} мм")
    cols[2].write(f"**n ячеек:** {params.n}")
    cols[2].write(f"**Излучение:** {'вкл' if params.rad_enabled else 'выкл'} × {params.rad_mult}")
    cols[3].write(f"**B / Ta:** {params.B:.2e} / {params.Ta:.0f}")
    cols[3].write(f"**Qcp / s:** {params.Qcp:.0f} / {params.s:.2f}")
    cols[3].write(f"**t_end:** {params.t_end:.2f} с")


def draw_frame(solver):
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


if run_button:
    solver = Solver(params)
    recorder = Recorder(solver)

    plot_slot = st.empty()
    metrics_slot = st.empty()

    steps_per_frame = 40
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
        cols[1].metric("r_f, мм", f"{q['r_f']*1000:.1f}")
        cols[2].metric("χ_rad", f"{q['chi_rad']:.2f}")
        cols[3].metric("η", f"{q['eta']:.2f}")
        cols[4].metric("Режим", q["regime"])

    st.success(f"Расчёт завершён: {solver.n_steps} шагов, t = {solver.t:.3f} с")

    if save_history:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_{selected.replace('.yaml', '')}_{int(time.time())}.h5"
        recorder.to_hdf5(out_path)
        st.info(f"История сохранена: {out_path}")
else:
    st.write("Выберите конфиг слева и нажмите «Запустить».")
