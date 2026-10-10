#pragma once

namespace ff1d {

struct Params {
    int n;
    double Rb, Rout, stretch;
    double p, mdot, YF_inj;
    double XO2_inf, T_inf, Tb;
    double Tign, tign, ignition_kernel_Rb;
    bool rad_enabled;
    double rad_mult, a_p_ref;
    double lam_ref, lam_exponent, cp;
    double Le_fuel, Le_o2, Le_product;
    double Mo, Mn2, Mp;
    double B, Ta, Qcp, s, Mf;
    double cfl, t_end;
    int n_iter;

    double Yo_inf;

    Params(int n_, double Rb_, double Rout_, double stretch_,
           double p_, double mdot_, double YF_inj_,
           double XO2_inf_, double T_inf_, double Tb_,
           double Tign_, double tign_, double ignition_kernel_Rb_,
           bool rad_enabled_, double rad_mult_, double a_p_ref_,
           double lam_ref_, double lam_exponent_, double cp_,
           double Le_fuel_, double Le_o2_, double Le_product_,
           double Mo_, double Mn2_, double Mp_,
           double B_, double Ta_, double Qcp_, double s_, double Mf_,
           double cfl_, double t_end_, int n_iter_)
        : n(n_), Rb(Rb_), Rout(Rout_), stretch(stretch_),
          p(p_), mdot(mdot_), YF_inj(YF_inj_),
          XO2_inf(XO2_inf_), T_inf(T_inf_), Tb(Tb_),
          Tign(Tign_), tign(tign_), ignition_kernel_Rb(ignition_kernel_Rb_),
          rad_enabled(rad_enabled_), rad_mult(rad_mult_), a_p_ref(a_p_ref_),
          lam_ref(lam_ref_), lam_exponent(lam_exponent_), cp(cp_),
          Le_fuel(Le_fuel_), Le_o2(Le_o2_), Le_product(Le_product_),
          Mo(Mo_), Mn2(Mn2_), Mp(Mp_),
          B(B_), Ta(Ta_), Qcp(Qcp_), s(s_), Mf(Mf_),
          cfl(cfl_), t_end(t_end_), n_iter(n_iter_)
    {
        Yo_inf = (XO2_inf * Mo) / (XO2_inf * Mo + (1.0 - XO2_inf) * Mn2);
    }
};

}
