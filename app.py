from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from flask import Flask, request, render_template
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Point = Optional[Tuple[int, int]]

app = Flask(__name__)
app.secret_key = "ecc-demo-secret"

scale=10

@dataclass
class CurveResult:
    a: int
    b: int
    p: int
    G: Point
    order_G: Optional[int]
    points: List[Tuple[int, int]]
    plot_uri: str
    error: Optional[str] = None


@dataclass
class HomoResult:
    G: Point
    K: Point
    k: int
    m1: int
    m2: int
    m1_float: float
    m2_float: float
    r1: int
    r2: int
    enc_m1: Point
    enc_m2: Point
    C1_1: Point
    C2_1: Point
    C1_2: Point
    C2_2: Point
    C1_sum: Point
    C2_sum: Point
    ecc_sum_encrypted: Point
    ecc_decoded_sum: Optional[int]
    report_steps: List[Dict[str, Any]]
    error: Optional[str] = None


@dataclass
class ScalarResult:
    G: Point
    K: Point
    k: int
    m1_float: float
    m1: int
    s_float: float
    s: int
    r1: int
    P_m1: Point
    C1: Point
    C2: Point
    C1_scaled: Point
    C2_scaled: Point
    decrypted_point: Point
    result: Optional[float]
    report_steps: List[Dict[str, Any]]
    error: Optional[str] = None

class EllipticCurve:
    def __init__(self, a: int, b: int, p: int):
        self.a = a
        self.b = b
        self.p = p
        if p <= 2:
            raise ValueError("p must be a prime greater than 2.")
        if (4 * a**3 + 27 * b**2) % p == 0:
            raise ValueError("Invalid curve: discriminant is zero modulo p.")

    def is_on_curve(self, P: Point) -> bool:
        if P is None:
            return True
        x, y = P
        return (y * y - (x**3 + self.a * x + self.b)) % self.p == 0

    def inv(self, x: int) -> int:
        return pow(x % self.p, -1, self.p)

    def add(self, P: Point, Q: Point) -> Point:
        if P is None:
            return Q
        if Q is None:
            return P

        x1, y1 = P
        x2, y2 = Q

        if x1 == x2 and (y1 + y2) % self.p == 0:
            return None

        if P != Q:
            m = ((y2 - y1) * self.inv(x2 - x1)) % self.p
        else:
            if y1 == 0:
                return None
            m = ((3 * x1**2 + self.a) * self.inv(2 * y1)) % self.p

        x3 = (m * m - x1 - x2) % self.p
        y3 = (m * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def negate(self, P: Point) -> Point:
        if P is None:
            return None
        x, y = P
        return (x, (-y) % self.p)

    def multiply(self, k: int, P: Point) -> Point:
        if P is None:
            return None
        if k < 0:
            return self.multiply(-k, self.negate(P))

        result = None
        temp = P
        while k > 0:
            if k & 1:
                result = self.add(result, temp)
            temp = self.add(temp, temp)
            k >>= 1
        return result

    def get_points(self) -> List[Tuple[int, int]]:
        pts = []
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            for y in range(self.p):
                if (y * y) % self.p == rhs:
                    pts.append((x, y))
        return pts

    def get_order_of_point(self, P: Point, max_search: Optional[int] = None) -> Optional[int]:
        if P is None:
            return 1
        current = None
        limit = self.p * 2 + 10 if max_search is None else max_search
        for i in range(1, limit + 1):
            current = self.add(current, P)
            if current is None:
                return i
        return None

    def decode_scalar_mult_to_int(self, P: Point, G: Point, max_search: Optional[int] = None) -> Optional[int]:
        if P is None:
            return 0
        current = None
        limit = self.p * 2 + 10 if max_search is None else max_search
        for i in range(1, limit + 1):
            current = self.add(current, G)
            if current == P:
                return i
        return None


def point_str(P: Point) -> str:
    return "∞" if P is None else f"({P[0]}, {P[1]})"


def plot_curve_png(
    curve: EllipticCurve,
    G: Point = None,
    K: Point = None,
    P1: Point = None,
    P2: Point = None
) -> str:
    pts = curve.get_points()
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xs, ys, s=35, label="Curve points", alpha=0.75)

    for label, P in [("G", G), ("K", K), ("P1", P1), ("P2", P2)]:
        if P is not None:
            ax.scatter([P[0]], [P[1]], s=120, zorder=5)
            ax.annotate(label, (P[0], P[1]), xytext=(6, 6), textcoords="offset points", fontsize=10)

    ax.set_title(f"Elliptic Curve over F_{curve.p}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1, curve.p)
    ax.set_ylim(-1, curve.p)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", frameon=False)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def compute_curve_result(form: Dict[str, str]) -> CurveResult:
    a = int(form.get("a", 1))
    b = int(form.get("b", 1))
    p = int(form.get("p", 23))

    curve = EllipticCurve(a, b, p)
    points = curve.get_points()
    if not points:
        raise ValueError("No points found on this curve.")

    G = points[0]
    order_G = curve.get_order_of_point(G)
    plot_uri = plot_curve_png(curve, G=G)

    return CurveResult(
        a=a,
        b=b,
        p=p,
        G=G,
        order_G=order_G,
        points=points,
        plot_uri=plot_uri,
    )


def compute_homo_result(curve_result: CurveResult, form: Dict[str, str]) -> HomoResult:
    p = curve_result.p
    order_G = curve_result.order_G
    max_m = ((order_G // 2 - 1)/scale)

    m1_float = float(form.get("m1", 0))
    m2_float = float(form.get("m2", 0))

    m1 = int(m1_float * scale)
    m2 = int(m2_float * scale)
    k = int(form.get("k", 2))
    r1 = int(form.get("r1", 3))
    r2 = int(form.get("r2", 7))

    if not (0 <= m1 <= max_m * scale and 0 <= m2 <= max_m * scale):
        raise ValueError(f"Messages must be between 0 and {max_m}.")
    if not (0 <= r1 < p and 0 <= r2 < p):
        raise ValueError(f"Random values must be between 0 and {p - 1}.")

    curve = EllipticCurve(curve_result.a, curve_result.b, p)
    G = curve_result.G

    K = curve.multiply(k, G)

    enc_m1 = curve.multiply(m1, G)
    enc_m2 = curve.multiply(m2, G)

    # ECC-ElGamal style encryption
    C1_1 = curve.multiply(r1, G)
    C2_1 = curve.add(enc_m1, curve.multiply(r1, K))

    C1_2 = curve.multiply(r2, G)
    C2_2 = curve.add(enc_m2, curve.multiply(r2, K))

    # Homomorphic addition of ciphertexts
    C1_sum = curve.add(C1_1, C1_2)
    C2_sum = curve.add(C2_1, C2_2)

    # Decryption
    decrypted_point = curve.add(
        C2_sum,
        curve.negate(curve.multiply(k, C1_sum))
)
    decoded_int = curve.decode_scalar_mult_to_int(decrypted_point, G)

    if decoded_int is not None:
        ecc_decoded_sum = decoded_int / scale
    else:
        ecc_decoded_sum = None

    report_steps = [
        {"kind": "normal", "label": "Curve equation", "value": f"y² = x³ + {curve_result.a}x + {curve_result.b} mod {p}"},
        {"kind": "normal", "label": "Generator point (G)", "value": point_str(G)},
        {"kind": "normal", "label": "Private key (k)", "value": k},
        {"kind": "normal", "label": "Public key (K = kG)", "value": point_str(K)},
        {"kind": "normal", "label": "Message m1 (float)", "value": m1_float},
        {"kind": "normal", "label": "Message m2 (float)", "value": m2_float},
        {"kind": "normal", "label": "Scaled m1", "value": m1},
        {"kind": "normal", "label": "Scaled m2", "value": m2},
        {"kind": "normal", "label": "Random r1", "value": r1},
        {"kind": "normal", "label": "Random r2", "value": r2},
        {"kind": "normal", "label": "Encode m1 → m1·G", "value": f"{m1} × G = {point_str(enc_m1)}"},
        {"kind": "normal", "label": "Encode m2 → m2·G", "value": f"{m2} × G = {point_str(enc_m2)}"},
        {"kind": "normal", "label": "Ciphertext 1: C1_1 = r1·G", "value": point_str(C1_1)},
        {"kind": "normal", "label": "Ciphertext 1: C2_1 = m1·G + r1·K", "value": point_str(C2_1)},
        {"kind": "normal", "label": "Ciphertext 2: C1_2 = r2·G", "value": point_str(C1_2)},
        {"kind": "normal", "label": "Ciphertext 2: C2_2 = m2·G + r2·K", "value": point_str(C2_2)},
        {"label": "Homomorphic add: C1_sum = C1_1 + C1_2", "value": point_str(C1_sum)},
        {"label": "Homomorphic add: C2_sum = C2_1 + C2_2", "value": point_str(C2_sum)},
        {"label": "Decryption: C2_sum - k·C1_sum", "value": point_str(decrypted_point)},
        {"kind": "success", "label": "Recovered sum point", "value": point_str(decrypted_point)},
        {"kind": "success", "label": "Decoded final answer", "value": ecc_decoded_sum if ecc_decoded_sum is not None else "N/A"},
    ]

    return HomoResult(
        G=G,
        K=K,
        k=k,
        m1=m1,
        m2=m2,
        m1_float=m1_float, 
        m2_float=m2_float,
        r1=r1,
        r2=r2,
        enc_m1=enc_m1,
        enc_m2=enc_m2,
        C1_1=C1_1,
        C2_1=C2_1,
        C1_2=C1_2,
        C2_2=C2_2,
        C1_sum=C1_sum,
        C2_sum=C2_sum,
        ecc_sum_encrypted=decrypted_point,
        ecc_decoded_sum=ecc_decoded_sum,
        report_steps=report_steps,
    )


def compute_scalar_result(curve_result: CurveResult, form: Dict[str, str]) -> ScalarResult:
    scale = 10

    curve = EllipticCurve(curve_result.a, curve_result.b, curve_result.p)
    G = curve_result.G

    # --- Inputs ---
    m1_float = float(form.get("m1", 0))
    s_float = float(form.get("s", 0))

    m1 = int(m1_float * scale)
    s = int(s_float * scale)

    k = int(form.get("k", 2))
    r1 = int(form.get("r1", 3))

    # --- Keys ---
    K = curve.multiply(k, G)

    # --- Encode ---
    P_m1 = curve.multiply(m1, G)

    # --- Encrypt ---
    C1 = curve.multiply(r1, G)
    C2 = curve.add(P_m1, curve.multiply(r1, K))

    # --- Scalar multiplication on ciphertext ---
    C1_scaled = curve.multiply(s, C1)
    C2_scaled = curve.multiply(s, C2)

    # --- Decrypt ---
    decrypted_point = curve.add(
        C2_scaled,
        curve.negate(curve.multiply(k, C1_scaled))
    )

    # --- Decode ---
    decoded_int = curve.decode_scalar_mult_to_int(decrypted_point, G)

    if decoded_int is not None:
        decoded_float = decoded_int / (scale * scale)   # VERY IMPORTANT
    else:
        decoded_float = None
    report_steps = [
        {"kind": "normal", "label": "Curve equation", "value": f"y² = x³ + {curve_result.a}x + {curve_result.b} mod {curve_result.p}"},
        {"kind": "normal", "label": "Generator point (G)", "value": point_str(G)},
        {"kind": "normal", "label": "Private key (k)", "value": k},
        {"kind": "normal", "label": "Public key (K = kG)", "value": point_str(K)},
        {"kind": "normal", "label": "Message m1 (float)", "value": m1_float},
        {"kind": "normal", "label": "Scaled m1", "value": m1},
        {"kind": "normal", "label": "Scalar s (float)", "value": s_float},
        {"kind": "normal", "label": "Scaled s", "value": s},
        {"kind": "normal", "label": "Random r1", "value": r1},
        {"kind": "normal", "label": "Encode m1 → m1·G", "value": point_str(P_m1)},
        {"kind": "normal", "label": "Ciphertext: C1 = r1·G", "value": point_str(C1)},
        {"kind": "normal", "label": "Ciphertext: C2 = P_m1 + r1·K", "value": point_str(C2)},
        {"kind": "normal", "label": "Scaled ciphertext: C1' = s·C1", "value": point_str(C1_scaled)},
        {"kind": "normal", "label": "Scaled ciphertext: C2' = s·C2", "value": point_str(C2_scaled)},
        {"kind": "normal", "label": "Decryption: C2' - k·C1'", "value": point_str(decrypted_point)},
        {"kind": "success", "label": "Recovered result point", "value": point_str(decrypted_point)},
        {"kind": "success", "label": "Decoded final answer", "value": decoded_float if decoded_float is not None else "N/A"},
    ]
    return ScalarResult(
        G=G,
        K=K,
        k=k,
        m1_float=m1_float,
        m1=m1,
        s_float=s_float,
        s=s,
        r1=r1,
        P_m1=P_m1,
        C1=C1,
        C2=C2,
        C1_scaled=C1_scaled,
        C2_scaled=C2_scaled,
        decrypted_point=decrypted_point,
        result=decoded_float,
        report_steps=report_steps,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    curve_result = None
    homo_result = None
    scalar_result = None
    error = None
    step = request.form.get("step", "1") if request.method == "POST" else "1"

    if step == "1":
        if request.method == "POST":
            try:
                curve_result = compute_curve_result(request.form)
            except Exception as e:
                error = str(e)

    elif step == "2":
        try:
            curve_form = {
                "a": request.form.get("curve_a"),
                "b": request.form.get("curve_b"),
                "p": request.form.get("curve_p"),
            }
            curve_result = compute_curve_result(curve_form)
            homo_result = compute_homo_result(curve_result, request.form)
        except Exception as e:
            error = str(e)

    elif step == "3":
        try:
            curve_form = {
                "a": request.form.get("curve_a"),
                "b": request.form.get("curve_b"),
                "p": request.form.get("curve_p"),
            }
            curve_result = compute_curve_result(curve_form)
            scalar_result = compute_scalar_result(curve_result, request.form)
        except Exception as e:
            error = str(e)

    max_m = (curve_result.order_G // 2 - 1)/10 if curve_result else 0
    max_m1 = (int(curve_result.order_G**(1/2)))/10 if curve_result else 0
    max_s = (int(curve_result.order_G**(1/2)))/10 if curve_result else 0

    return render_template(
    "index.html",
    curve_result=curve_result,
    homo_result=homo_result,
    scalar_result=scalar_result, 
    error=error,
    step=step,
    max_m=max_m,
    max_m1=max_m1,
    max_s=max_s,
)


import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))