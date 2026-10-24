import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.optimize import brentq
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

ATM_MMHG = 760.0


# ====================== THERMODYNAMIQUE ======================
def psat(T):
    """Antoine (mmHg, degC). 1 = methanol, 2 = eau"""
    p1 = 10 ** (8.08097 - 1582.271 / (T + 239.726))
    p2 = 10 ** (8.07131 - 1730.63 / (T + 233.426))
    return p1, p2


def gamma(x1, A12, A21):
    """Coefficients d'activite (Van Laar)"""
    x2 = 1 - x1
    d = A12 * x1 + A21 * x2
    if d == 0:
        return 1.0, 1.0
    return (np.exp(A12 * (A21 * x2 / d) ** 2),
            np.exp(A21 * (A12 * x1 / d) ** 2))


def K_values(x1, T, P, A12, A21):
    """Ki = gamma_i * Psat_i / P"""
    g1, g2 = gamma(x1, A12, A21)
    p1, p2 = psat(T)
    return g1 * p1 / P, g2 * p2 / P


def rachford_rice(V, z1, K1, K2):
    """f(V) = sum zi (Ki - 1) / (1 + V (Ki - 1))"""
    return (z1 * (K1 - 1) / (1 + V * (K1 - 1))
            + (1 - z1) * (K2 - 1) / (1 + V * (K2 - 1)))


def flash(z1, T, P, A12, A21, tol=1e-10, nmax=1000):
    """Flash isotherme par Rachford-Rice. Retourne un dict de resultats."""
    x1 = z1
    V = 0.5
    for _ in range(nmax):
        K1, K2 = K_values(x1, T, P, A12, A21)
        f0 = rachford_rice(0.0, z1, K1, K2)
        f1 = rachford_rice(1.0, z1, K1, K2)
        if f0 < 0:
            return {"phase": "Liquide sous-refroidi (T < T bulle) : une seule phase",
                    "K1": K1, "K2": K2}
        if f1 > 0:
            return {"phase": "Vapeur surchauffee (T > T rosee) : une seule phase",
                    "K1": K1, "K2": K2}
        V = brentq(rachford_rice, 0.0, 1.0, args=(z1, K1, K2))
        x_new = z1 / (1 + V * (K1 - 1))
        if abs(x_new - x1) < tol:
            x1 = x_new
            break
        x1 = 0.5 * x1 + 0.5 * x_new
    K1, K2 = K_values(x1, T, P, A12, A21)
    V = brentq(rachford_rice, 0.0, 1.0, args=(z1, K1, K2))
    x1 = z1 / (1 + V * (K1 - 1))
    y1 = K1 * x1
    return {"phase": "Equilibre liquide-vapeur (deux phases)",
            "V": V, "L": 1 - V, "x1": x1, "x2": 1 - x1,
            "y1": y1, "y2": 1 - y1, "K1": K1, "K2": K2}


def courbes_Txy(P, A12, A21, n=101):
    """Courbe de bulle (x, T) et de rosee (y, T) a P donnee"""
    xs = np.linspace(0, 1, n)
    Tb, ys = [], []
    for x in xs:
        def f(t):
            g1, g2 = gamma(x, A12, A21)
            p1, p2 = psat(t)
            return x * g1 * p1 / P + (1 - x) * g2 * p2 / P - 1
        t = brentq(f, 0.0, 250.0)
        g1, _ = gamma(x, A12, A21)
        Tb.append(t)
        ys.append(x * g1 * psat(t)[0] / P)
    return xs, np.array(Tb), np.array(ys)


# ====================== INTERFACE ======================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flash eau - methanol (Rachford-Rice)")
        self.geometry("1300x760")

        gauche = ttk.Frame(self, padding=10)
        gauche.pack(side="left", fill="y")

        # ---- Inputs
        inp = ttk.LabelFrame(gauche, text="INPUTS", padding=10)
        inp.pack(fill="x")
        self.vars = {}
        champs = [("Débit F (mol/h)", "F", "50"),
                  ("T (°C)", "T", "80"),
                  ("P (atm)", "P", "1"),
                  ("z méthanol (fraction molaire)", "z", "0.4"),
                  ("Van Laar A12", "A12", "0.58"),
                  ("Van Laar A21", "A21", "0.46")]
        for i, (txt, cle, val) in enumerate(champs):
            ttk.Label(inp, text=txt).grid(row=i, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=val)
            ttk.Entry(inp, textvariable=v, width=10).grid(row=i, column=1, padx=8)
            self.vars[cle] = v

        ttk.Button(gauche, text="DESSINER", command=self.dessiner).pack(
            fill="x", pady=12, ipady=6)

        # ---- Outputs
        out = ttk.LabelFrame(gauche, text="OUTPUTS", padding=10)
        out.pack(fill="x")
        self.out = {}
        lignes = [("Etat", "phase"), ("V/F", "V"), ("L/F", "L"),
                  ("Débit vapeur V (mol/h)", "Vmol"),
                  ("Débit liquide L (mol/h)", "Lmol"),
                  ("K méthanol", "K1"), ("K eau", "K2"),
                  ("Liquide : x méthanol", "x1"), ("Liquide : x eau", "x2"),
                  ("Vapeur : y méthanol", "y1"), ("Vapeur : y eau", "y2"),
                  ("Liquide : méthanol (mol/h)", "Lm"),
                  ("Liquide : eau (mol/h)", "Le"),
                  ("Vapeur : méthanol (mol/h)", "Vm"),
                  ("Vapeur : eau (mol/h)", "Ve")]
        for i, (txt, cle) in enumerate(lignes):
            ttk.Label(out, text=txt).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.StringVar(value="-")
            ttk.Label(out, textvariable=v, font=("Segoe UI", 10, "bold"),
                      wraplength=220, justify="left").grid(
                row=i, column=1, sticky="w", padx=8)
            self.out[cle] = v

        # ---- Graphe
        droite = ttk.Frame(self)
        droite.pack(side="right", fill="both", expand=True)
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=droite)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.dessiner()

    def lire(self):
        try:
            F = float(self.vars["F"].get().replace(",", "."))
            T = float(self.vars["T"].get().replace(",", "."))
            P = float(self.vars["P"].get().replace(",", ".")) * ATM_MMHG
            z = float(self.vars["z"].get().replace(",", "."))
            A12 = float(self.vars["A12"].get().replace(",", "."))
            A21 = float(self.vars["A21"].get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Erreur", "Entrez des nombres valides.")
            return None
        if not 0 < z < 1:
            messagebox.showerror("Erreur", "z doit être entre 0 et 1 (exclus).")
            return None
        if P <= 0 or F <= 0:
            messagebox.showerror("Erreur", "F et P doivent être positifs.")
            return None
        return F, T, P, z, A12, A21

    def dessiner(self):
        donnees = self.lire()
        if donnees is None:
            return
        F, T, P, z, A12, A21 = donnees

        # --- Calcul flash (Rachford-Rice)
        r = flash(z, T, P, A12, A21)
        if "V" in r:
            r["Vmol"] = r["V"] * F            # debit vapeur (mol/h)
            r["Lmol"] = r["L"] * F            # debit liquide (mol/h)
            r["Lm"] = r["Lmol"] * r["x1"]     # methanol dans le liquide
            r["Le"] = r["Lmol"] * r["x2"]     # eau dans le liquide
            r["Vm"] = r["Vmol"] * r["y1"]     # methanol dans la vapeur
            r["Ve"] = r["Vmol"] * r["y2"]     # eau dans la vapeur
        for cle, var in self.out.items():
            if cle == "phase":
                var.set(r["phase"])
            elif cle in r:
                var.set(f"{r[cle]:.4f}")
            else:
                var.set("-")

        # --- Diagrammes
        try:
            xs, Tb, ys = courbes_Txy(P, A12, A21)
        except ValueError:
            messagebox.showerror("Erreur", "Impossible de tracer le diagramme "
                                           "pour ces conditions.")
            return
        P_atm = P / ATM_MMHG
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(xs, Tb, "b", label="Liquide (courbe de bulle)")
        self.ax1.plot(ys, Tb, "r", label="Vapeur (courbe de rosée)")
        self.ax1.axvline(z, color="gray", ls=":", label=f"z = {z}")
        if "x1" in r:
            self.ax1.plot([r["x1"], r["y1"]], [T, T], "ko-", label=f"T = {T} °C")
            self.ax1.annotate(f"x={r['x1']:.3f}", (r["x1"], T),
                              textcoords="offset points", xytext=(-5, -15))
            self.ax1.annotate(f"y={r['y1']:.3f}", (r["y1"], T),
                              textcoords="offset points", xytext=(-5, 8))
        else:
            self.ax1.plot([z], [T], "ks", label=f"Alimentation ({T} °C)")
        self.ax1.set_xlabel("x, y (méthanol)")
        self.ax1.set_ylabel("T (°C)")
        self.ax1.set_title(f"T-xy eau-méthanol, P = {P_atm:g} atm")
        self.ax1.legend(fontsize=8)
        self.ax1.grid(True)

        self.ax2.plot(xs, ys, "r", label="y = f(x)")
        self.ax2.plot([0, 1], [0, 1], "k--")
        if "x1" in r:
            self.ax2.plot(r["x1"], r["y1"], "ko")
        self.ax2.set_xlabel("x méthanol")
        self.ax2.set_ylabel("y méthanol")
        self.ax2.set_title("Diagramme y-x")
        self.ax2.legend(fontsize=8)
        self.ax2.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()