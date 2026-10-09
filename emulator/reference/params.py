from dataclasses import dataclass, field

import yaml

from formulas import thermal_diffusivity, mass_diffusivity


@dataclass
class Params:
    nx: int
    ny: int
    Lx: float
    Ly: float
    rho: float
    cp: float
    lam: float
    U: float
    T_inf: float
    Yo_inf: float
    fuel_x0: float
    fuel_x1: float
    Ts: float
    Yf_s: float
    B: float
    Ta: float
    Qcp: float
    s: float
    a_p: float
    ign_T: float
    ign_t: float
    ign_x0: float
    ign_x1: float
    ign_y1: float
    t_end: float
    cfl: float
    fo: float

    dx: float = field(init=False)
    dy: float = field(init=False)
    alpha: float = field(init=False)
    D: float = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.alpha = thermal_diffusivity(self.lam, self.rho, self.cp)
        self.D = mass_diffusivity(self.alpha, Le=1.0)

        dt_diff = self.fo * 0.5 / (self.alpha * (1.0 / self.dx**2 + 1.0 / self.dy**2))
        if self.U > 0:
            dt_adv = self.cfl * self.dx / self.U
            self.dt = min(dt_adv, dt_diff)
        else:
            self.dt = dt_diff

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls(
            nx=int(cfg["grid"]["nx"]), ny=int(cfg["grid"]["ny"]),
            Lx=float(cfg["grid"]["Lx"]), Ly=float(cfg["grid"]["Ly"]),
            rho=float(cfg["gas"]["rho"]), cp=float(cfg["gas"]["cp"]), lam=float(cfg["gas"]["lam"]),
            U=float(cfg["flow"]["U"]), T_inf=float(cfg["flow"]["T_inf"]), Yo_inf=float(cfg["flow"]["Yo_inf"]),
            fuel_x0=float(cfg["fuel"]["x0"]), fuel_x1=float(cfg["fuel"]["x1"]),
            Ts=float(cfg["fuel"]["Ts"]), Yf_s=float(cfg["fuel"]["Yf_s"]),
            B=float(cfg["react"]["B"]), Ta=float(cfg["react"]["Ta"]),
            Qcp=float(cfg["react"]["Qcp"]), s=float(cfg["react"]["s"]),
            a_p=float(cfg["rad"]["a_p"]),
            ign_T=float(cfg["ignite"]["T"]), ign_t=float(cfg["ignite"]["t"]),
            ign_x0=float(cfg["ignite"]["x0"]), ign_x1=float(cfg["ignite"]["x1"]), ign_y1=float(cfg["ignite"]["y1"]),
            t_end=float(cfg["time"]["t_end"]), cfl=float(cfg["time"]["cfl"]), fo=float(cfg["time"]["fo"]),
        )
