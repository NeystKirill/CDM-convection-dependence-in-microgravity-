from dataclasses import dataclass

import yaml

from .fuel import FUELS, M_O2, M_N2, M_PRODUCT, LE_FUEL, LE_O2, LE_PRODUCT, RB_SFLAME
from .gas import mole_to_mass_fraction


@dataclass
class Params:
    fuel: str
    Rb: float
    Rout: float
    n: int
    stretch: float
    p: float
    mdot: float
    YF_inj: float
    XO2_inf: float
    T_inf: float
    Tb: float
    Tign: float
    tign: float
    ignition_kernel_Rb: float
    rad_enabled: bool
    rad_mult: float
    a_p_ref: float
    lam_ref: float
    lam_exponent: float
    cp: float
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
    fo: float
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
        rad = cfg.get("radiation", {})
        ign = cfg["ignite"]
        num = cfg.get("numerics", cfg.get("time", {}))

        return cls(
            fuel=fuel_type,
            Rb=float(geom.get("Rb", RB_SFLAME)),
            Rout=float(geom.get("Rout", 30.0 * geom.get("Rb", RB_SFLAME))),
            n=int(geom.get("n", 150)),
            stretch=float(geom.get("stretch", 1.0)),
            p=float(amb["p_atm"]) * 101325.0,
            mdot=float(cfg["fuel"]["mdot_mg_s"]) * 1.0e-6,
            YF_inj=float(cfg["fuel"].get("YF_inj", 1.0)),
            XO2_inf=float(amb["XO2"]),
            T_inf=float(amb.get("T_inf", 300.0)),
            Tb=float(cfg["fuel"].get("Tb", 400.0)),
            Tign=float(ign["T"]),
            tign=float(ign["t"]),
            ignition_kernel_Rb=float(ign.get("kernel_Rb_multiples", 2.0)),
            rad_enabled=bool(rad.get("enabled", True)),
            rad_mult=float(rad.get("mult", 1.0)),
            a_p_ref=float(rad.get("a_p_ref", 0.6)),
            lam_ref=float(tr.get("lam_ref", 0.07)),
            lam_exponent=float(tr.get("lam_exponent", 0.7)),
            cp=float(tr.get("cp", 1167.0)),
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
            fo=float(num.get("fo", 0.4)),
            t_end=float(num["t_end"]),
            n_iter=int(num.get("reaction_iterations", 12)),
        )
