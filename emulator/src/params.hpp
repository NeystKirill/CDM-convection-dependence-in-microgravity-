#pragma once
#include <algorithm>

namespace blueball {

struct Params {
    int nx, ny;
    double Lx, Ly;
    double rho, cp, lam;
    double U, T_inf, Yo_inf;
    double fuel_x0, fuel_x1, Ts, Yf_s;
    double B, Ta, Qcp, s;
    double a_p;
    double ign_T, ign_t, ign_x0, ign_x1, ign_y1;
    double t_end, cfl, fo;

    double dx, dy, alpha, D, dt;

    Params(int nx_, int ny_, double Lx_, double Ly_,
           double rho_, double cp_, double lam_,
           double U_, double T_inf_, double Yo_inf_,
           double fuel_x0_, double fuel_x1_, double Ts_, double Yf_s_,
           double B_, double Ta_, double Qcp_, double s_,
           double a_p_,
           double ign_T_, double ign_t_, double ign_x0_, double ign_x1_, double ign_y1_,
           double t_end_, double cfl_, double fo_)
        : nx(nx_), ny(ny_), Lx(Lx_), Ly(Ly_),
          rho(rho_), cp(cp_), lam(lam_),
          U(U_), T_inf(T_inf_), Yo_inf(Yo_inf_),
          fuel_x0(fuel_x0_), fuel_x1(fuel_x1_), Ts(Ts_), Yf_s(Yf_s_),
          B(B_), Ta(Ta_), Qcp(Qcp_), s(s_),
          a_p(a_p_),
          ign_T(ign_T_), ign_t(ign_t_), ign_x0(ign_x0_), ign_x1(ign_x1_), ign_y1(ign_y1_),
          t_end(t_end_), cfl(cfl_), fo(fo_)
    {
        dx = Lx / nx;
        dy = Ly / ny;
        alpha = lam / (rho * cp);
        D = alpha;

        double dt_diff = fo * 0.5 / (alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        if (U > 0.0) {
            double dt_adv = cfl * dx / U;
            dt = std::min(dt_adv, dt_diff);
        } else {
            dt = dt_diff;
        }
    }
};

}
