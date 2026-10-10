from dataclasses import dataclass

import yaml

from emulator.reference.fuel import FUELS, M_O2, M_N2, M_PRODUCT, LE_FUEL, LE_O2, LE_PRODUCT
from emulator.reference.gas import mole_to_mass_fraction

G0 = 9.81


@dataclass
class Params:
    fuel: str
    Lx: float
    Ly: float
    Lz: float
    nx: int
    ny: int
    nz: int
    wick_r: float
    wick_x0: float
    wick_y0: float
    p: float
    XO2_inf: float
    T_inf: float
    Ts: float
    Yf_s: float
    g_frac: float
    U_ext: float
    Tign: float
    tign: float
    ignition_kernel_r: float
    ignition_kernel_z: float
    rad_enabled: bool
    rad_mult: float
    a_p_ref: float
    lam_ref: float
    lam_exponent: float
    cp: float
    Pr: float
    Le_fuel: float
    Le_o2: float
    Le_product: float
    Mo: float
    Mn2: float
    Mp: float
    B: float
    Ta: float
    Qcp: float
    s: float
    Mf: float
    cfl: float
    t_end: float
    n_iter: int

    def __post_init__(self):
        self.Yo_inf = mole_to_mass_fraction(self.XO2_inf, self.Mo, self.Mn2)

    @classmethod
    def from_yaml(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        fuel_type = cfg["fuel"]["type"]
        fuel_defaults = FUELS[fuel_type]
        kin = cfg["fuel"].get("kinetics", {})

        geom = cfg["geometry"]
        amb = cfg["ambient"]
        tr = cfg.get("transport", {})
        flow = cfg.get("flow", {})
        rad = cfg.get("radiation", {})
        ign = cfg["ignite"]
        num = cfg.get("numerics", {})

        Lx = float(geom["Lx"])
        Ly = float(geom["Ly"])
        wick_r = float(geom.get("wick_r", 0.004))

        return cls(
            fuel=fuel_type,
            Lx=Lx,
            Ly=Ly,
            Lz=float(geom["Lz"]),
            nx=int(geom.get("nx", 20)),
            ny=int(geom.get("ny", 20)),
            nz=int(geom.get("nz", 40)),
            wick_r=wick_r,
            wick_x0=float(geom.get("wick_x0", Lx / 2.0)),
            wick_y0=float(geom.get("wick_y0", Ly / 2.0)),
            p=float(amb["p_atm"]) * 101325.0,
            XO2_inf=float(amb["XO2"]),
            T_inf=float(amb.get("T_inf", 300.0)),
            Ts=float(cfg["fuel"].get("Ts", 600.0)),
            Yf_s=float(cfg["fuel"].get("Yf_s", 1.0)),
            g_frac=float(flow.get("g_frac", 1.0)),
            U_ext=float(flow.get("U_ext", 0.0)),
            Tign=float(ign["T"]),
            tign=float(ign["t"]),
            ignition_kernel_r=float(ign.get("kernel_r", 2.0 * wick_r)),
            ignition_kernel_z=float(ign.get("kernel_z", 2.0 * wick_r)),
            rad_enabled=bool(rad.get("enabled", True)),
            rad_mult=float(rad.get("mult", 1.0)),
            a_p_ref=float(rad.get("a_p_ref", 0.6)),
            lam_ref=float(tr.get("lam_ref", 0.07)),
            lam_exponent=float(tr.get("lam_exponent", 0.7)),
            cp=float(tr.get("cp", 1167.0)),
            Pr=float(tr.get("Pr", 0.7)),
            Le_fuel=float(tr.get("Le_fuel", LE_FUEL)),
            Le_o2=float(tr.get("Le_o2", LE_O2)),
            Le_product=float(tr.get("Le_product", LE_PRODUCT)),
            Mo=float(tr.get("M_o2", M_O2)),
            Mn2=float(tr.get("M_n2", M_N2)),
            Mp=float(tr.get("M_product", M_PRODUCT)),
            B=float(kin.get("B", fuel_defaults["B"])),
            Ta=float(kin.get("Ta", fuel_defaults["Ta"])),
            Qcp=float(kin.get("Qcp", fuel_defaults["Qcp"])),
            s=float(kin.get("s", fuel_defaults["s"])),
            Mf=float(kin.get("Mf", fuel_defaults["M"])),
            cfl=float(num.get("cfl", 0.4)),
            t_end=float(num["t_end"]),
            n_iter=int(num.get("reaction_iterations", 12)),
        )
