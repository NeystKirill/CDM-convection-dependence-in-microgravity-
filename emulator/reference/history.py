import numpy as np


class Recorder:
    def __init__(self, solver, field_every=50):
        self.solver = solver
        self.field_every = field_every
        self.t = []
        self.qoi_rows = []
        self.snapshot_t = []
        self.snapshot_T = []
        self.snapshot_Yf = []
        self.snapshot_Yo = []

    def capture(self):
        s = self.solver
        q = s.qoi()
        self.t.append(q["t"])
        self.qoi_rows.append(q)

        if s.n_steps % self.field_every == 0:
            state = s.state()
            self.snapshot_t.append(s.t)
            self.snapshot_T.append(state["T"])
            self.snapshot_Yf.append(state["Yf"])
            self.snapshot_Yo.append(state["Yo"])

    def qoi_series(self, key):
        return np.array([row[key] if row[key] is not None else np.nan for row in self.qoi_rows])

    def to_hdf5(self, path):
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("r", data=self.solver.grid.r_c)
            f.create_dataset("t", data=np.array(self.t))
            for key in ("r_f", "T_max", "chi_rad", "eta"):
                f.create_dataset(f"qoi/{key}", data=self.qoi_series(key))
            f.create_dataset("qoi/regime", data=np.array(
                [row["regime"] for row in self.qoi_rows], dtype="S32"
            ))
            f.create_dataset("fields/t", data=np.array(self.snapshot_t))
            f.create_dataset("fields/T", data=np.array(self.snapshot_T))
            f.create_dataset("fields/Yf", data=np.array(self.snapshot_Yf))
            f.create_dataset("fields/Yo", data=np.array(self.snapshot_Yo))
