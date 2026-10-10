import numpy as np


class Recorder:
    def __init__(self, solver, field_every=50):
        self.solver = solver
        self.field_every = field_every
        self.t = []
        self.qoi_rows = []
        self.snapshot_t = []
        self.snapshot_fields = {}

    def capture(self):
        s = self.solver
        q = s.qoi()
        self.t.append(q["t"])
        self.qoi_rows.append(q)

        if s.n_steps % self.field_every == 0:
            state = s.state()
            self.snapshot_t.append(s.t)
            for key, value in state.items():
                self.snapshot_fields.setdefault(key, []).append(value)

    def qoi_series(self, key):
        return np.array([row[key] if row[key] is not None else np.nan for row in self.qoi_rows])

    def to_hdf5(self, path):
        import h5py

        qoi_keys = [k for k, v in self.qoi_rows[0].items() if k not in ("t", "regime")]

        with h5py.File(path, "w") as f:
            f.create_dataset("t", data=np.array(self.t))
            for key in qoi_keys:
                f.create_dataset(f"qoi/{key}", data=self.qoi_series(key))
            f.create_dataset("qoi/regime", data=np.array(
                [row["regime"] for row in self.qoi_rows], dtype="S32"
            ))
            f.create_dataset("fields/t", data=np.array(self.snapshot_t))
            for key, values in self.snapshot_fields.items():
                f.create_dataset(f"fields/{key}", data=np.array(values))
