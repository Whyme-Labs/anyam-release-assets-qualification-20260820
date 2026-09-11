from manim import *
import numpy as np

config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30

PURPLE = "#6630F8"
PURPLE_LIGHT = "#A175F1"
ORANGE = "#F05F41"
GREEN = "#8FF740"
GREY = "#A0A0A1"
GREY_DARK = "#34343D"
WHITE = "#FFFFFF"
BLACK = "#000000"


def ui_text(text, size=28, color=WHITE, weight="NORMAL"):
    return Text(text, font="DejaVu Sans", font_size=size, color=color, weight=weight)


class MathShot(Scene):
    duration = 8.0
    kicker = ""

    def panel(self):
        card = RoundedRectangle(
            width=12.45,
            height=5.45,
            corner_radius=0.20,
            fill_color=BLACK,
            fill_opacity=0.965,
            stroke_color=GREY_DARK,
            stroke_width=1.4,
        ).shift(DOWN * 0.08)
        kicker = ui_text(self.kicker, 23, GREY, "BOLD").move_to(UP * 2.36)
        rule = Line(LEFT * 5.65, RIGHT * 5.65, color=GREY_DARK, stroke_width=1.2).move_to(UP * 2.05)
        self.play(FadeIn(card), FadeIn(kicker, shift=UP * 0.06), Create(rule), run_time=0.35)
        return card

    def finish(self):
        remaining = self.duration - self.time - 0.28
        if remaining > 0:
            self.wait(remaining)
        self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.28)


class AEObjective(MathShot):
    duration = 12.756
    kicker = "AUTOENCODER  /  MAP + RECONSTRUCTION OBJECTIVE"

    def construct(self):
        self.panel()
        x = MathTex(r"x", color=WHITE).scale(1.35)
        z = MathTex(r"z=f_{\theta}(x)", color=PURPLE_LIGHT).scale(1.05)
        xh = MathTex(r"\hat{x}=g_{\psi}(z)", color=WHITE).scale(1.05)
        x.move_to(LEFT * 4.7 + UP * 0.55)
        z.move_to(ORIGIN + UP * 0.55)
        xh.move_to(RIGHT * 4.2 + UP * 0.55)
        a1 = Arrow(x.get_right(), z.get_left(), buff=0.22, color=GREY, stroke_width=3, max_tip_length_to_length_ratio=0.08)
        a2 = Arrow(z.get_right(), xh.get_left(), buff=0.22, color=GREY, stroke_width=3, max_tip_length_to_length_ratio=0.08)
        enc = ui_text("encode", 22, GREY).next_to(a1, UP, buff=0.12)
        dec = ui_text("decode", 22, GREY).next_to(a2, UP, buff=0.12)
        self.play(Write(x), run_time=0.45)
        self.play(GrowArrow(a1), FadeIn(enc), Write(z), run_time=0.9)
        self.play(GrowArrow(a2), FadeIn(dec), Write(xh), run_time=0.9)

        loss = MathTex(r"\mathcal{L}_{\mathrm{rec}} = d\!\left(x,\hat{x}\right)", color=WHITE).scale(1.25)
        loss.set_color_by_tex(r"\mathcal{L}_{\mathrm{rec}}", PURPLE_LIGHT)
        loss.move_to(DOWN * 1.25)
        brace = Brace(loss, DOWN, color=PURPLE)
        note = ui_text("the bottleneck is useful only if the decoder can reconstruct what matters", 23, GREY).next_to(brace, DOWN, buff=0.16)
        self.play(Write(loss), run_time=0.9)
        self.play(GrowFromCenter(brace), FadeIn(note, shift=UP * 0.08), run_time=0.6)
        self.finish()


class VAEReparameterization(MathShot):
    duration = 10.616
    kicker = "VAE  /  REPARAMETERIZATION"

    def construct(self):
        self.panel()
        rng = np.random.default_rng(7)
        raw = rng.normal(size=(28, 2))
        raw[:, 0] *= 0.72
        raw[:, 1] *= 0.72
        cloud = VGroup(*[
            Dot(np.array([-3.65 + p[0], -0.15 + p[1], 0]), radius=0.035, color=WHITE, fill_opacity=0.82)
            for p in raw
        ])
        eps_label = MathTex(r"\epsilon\sim\mathcal{N}(0,I)", color=WHITE).scale(0.9).move_to(LEFT * 3.65 + DOWN * 1.55)
        target = VGroup(*[
            Dot(np.array([3.55 + 1.15 * p[0], -0.15 + 0.62 * p[1], 0]), radius=0.035, color=PURPLE_LIGHT, fill_opacity=0.9)
            for p in raw
        ])
        z_label = MathTex(r"z", color=PURPLE_LIGHT).scale(1.15).move_to(RIGHT * 3.55 + DOWN * 1.55)
        self.play(LaggedStart(*[FadeIn(d) for d in cloud], lag_ratio=0.025), Write(eps_label), run_time=0.8)

        formula = MathTex(r"z", "=", r"\mu", "+", r"\sigma\odot\epsilon").scale(1.32).move_to(UP * 1.0)
        formula[0].set_color(PURPLE_LIGHT)
        formula[2].set_color(PURPLE_LIGHT)
        formula[4].set_color(ORANGE)
        self.play(Write(formula), run_time=0.85)

        arrow = Arrow(LEFT * 1.75 + DOWN * 0.2, RIGHT * 1.55 + DOWN * 0.2, color=GREY, stroke_width=3, max_tip_length_to_length_ratio=0.07)
        scale_txt = ui_text("scale by σ, then shift by μ", 23, GREY).next_to(arrow, UP, buff=0.13)
        self.play(GrowArrow(arrow), FadeIn(scale_txt), run_time=0.45)
        self.play(Transform(cloud, target), Transform(eps_label, z_label), run_time=1.25, rate_func=smooth)
        note = ui_text("randomness comes from ε; μ and σ remain differentiable", 23, GREY).move_to(DOWN * 2.02)
        self.play(FadeIn(note, shift=UP * 0.08), run_time=0.45)
        self.finish()


class ELBO(MathShot):
    duration = 7.912
    kicker = "VAE  /  EVIDENCE LOWER BOUND"

    def construct(self):
        self.panel()
        lhs = MathTex(r"\mathcal{L}_{\mathrm{ELBO}}", "=", color=WHITE).scale(1.05)
        rec = MathTex(r"\mathbb{E}_{q_{\phi}(z\mid x)}[\log p_{\theta}(x\mid z)]", color=PURPLE_LIGHT).scale(0.95)
        minus = MathTex("-", color=WHITE).scale(1.1)
        kl = MathTex(r"D_{\mathrm{KL}}\!\left(q_{\phi}(z\mid x)\,\|\,p(z)\right)", color=ORANGE).scale(0.95)
        row = VGroup(lhs, rec, minus, kl).arrange(RIGHT, buff=0.28).move_to(UP * 0.55)
        self.play(Write(lhs), run_time=0.4)
        self.play(Write(rec), run_time=0.8)
        self.play(Write(minus), Write(kl), run_time=0.8)

        b1 = Brace(rec, DOWN, color=PURPLE)
        t1 = ui_text("reconstruct the observation", 23, PURPLE_LIGHT, "BOLD").next_to(b1, DOWN, buff=0.14)
        b2 = Brace(kl, DOWN, color=ORANGE)
        t2 = ui_text("keep q(z|x) close to the prior", 23, ORANGE, "BOLD").next_to(b2, DOWN, buff=0.14)
        self.play(GrowFromCenter(b1), FadeIn(t1), run_time=0.5)
        self.play(GrowFromCenter(b2), FadeIn(t2), run_time=0.5)

        balance = ui_text("maximize both goals together", 25, WHITE, "BOLD").move_to(DOWN * 2.0)
        self.play(FadeIn(balance, shift=UP * 0.08), run_time=0.45)
        self.finish()


class RateDistortion(MathShot):
    duration = 11.433
    kicker = "VAE  /  RATE–DISTORTION VIEW"

    def construct(self):
        self.panel()
        dbox = RoundedRectangle(width=4.7, height=2.25, corner_radius=0.18, stroke_color=PURPLE, fill_color=BLACK, fill_opacity=0.55).shift(LEFT * 2.75 + UP * 0.2)
        rbox = RoundedRectangle(width=4.7, height=2.25, corner_radius=0.18, stroke_color=ORANGE, fill_color=BLACK, fill_opacity=0.55).shift(RIGHT * 2.75 + UP * 0.2)
        D = MathTex(r"D=-\mathbb{E}_{q}[\log p_{\theta}(x\mid z)]", color=PURPLE_LIGHT).scale(0.82).move_to(dbox.get_center() + UP * 0.25)
        R = MathTex(r"R=D_{\mathrm{KL}}(q_{\phi}(z\mid x)\,\|\,p(z))", color=ORANGE).scale(0.82).move_to(rbox.get_center() + UP * 0.25)
        dl = ui_text("distortion: reconstruction cost", 22, GREY).move_to(dbox.get_center() + DOWN * 0.62)
        rl = ui_text("rate: information paid for in z", 22, GREY).move_to(rbox.get_center() + DOWN * 0.62)
        self.play(Create(dbox), Write(D), FadeIn(dl), run_time=0.85)
        self.play(Create(rbox), Write(R), FadeIn(rl), run_time=0.85)

        objective = MathTex(r"\mathcal{J}_{\beta}=D+\beta R", color=WHITE).scale(1.25).move_to(DOWN * 1.65)
        objective.set_color_by_tex("D", PURPLE_LIGHT)
        objective.set_color_by_tex("R", ORANGE)
        self.play(Write(objective), run_time=0.75)
        beta = MathTex(r"\beta\uparrow", color=ORANGE).scale(0.92).next_to(objective, RIGHT, buff=0.55)
        note = ui_text("more pressure toward the prior", 22, GREY).next_to(beta, DOWN, buff=0.1)
        self.play(FadeIn(beta), FadeIn(note), run_time=0.5)
        self.play(Indicate(rbox, color=ORANGE, scale_factor=1.03), run_time=0.7)
        self.finish()


class VQNearest(MathShot):
    duration = 14.961
    kicker = "VQ-VAE  /  NEAREST CODEBOOK ENTRY"

    def construct(self):
        self.panel()
        axes = Axes(x_range=[-2.5, 2.6, 1], y_range=[-1.8, 1.9, 1], x_length=5.0, y_length=3.35, tips=False,
                    axis_config={"stroke_color": GREY_DARK, "stroke_width": 1.3}).shift(LEFT * 3.0 + DOWN * 0.15)
        points = [(-1.8, 1.1), (-1.1, -0.9), (0.1, 1.35), (0.55, -0.5), (1.65, 0.75), (2.0, -1.0)]
        code_dots = VGroup(*[Dot(axes.c2p(x, y), radius=0.075, color=GREY) for x, y in points])
        query_xy = (0.92, -0.1)
        query = Dot(axes.c2p(*query_xy), radius=0.1, color=ORANGE)
        qlab = MathTex(r"z_e(x)", color=ORANGE).scale(0.7).next_to(query, UP, buff=0.12)
        nearest_idx = 3
        nearest = code_dots[nearest_idx]
        nearest_lab = MathTex(r"e_{k^*}", color=PURPLE_LIGHT).scale(0.75).next_to(nearest, DOWN, buff=0.12)
        self.play(Create(axes), LaggedStart(*[FadeIn(d) for d in code_dots], lag_ratio=0.08), run_time=0.9)
        self.play(FadeIn(query, scale=0.5), Write(qlab), run_time=0.55)

        formula = MathTex(r"k^*=\arg\min_k\left\|z_e(x)-e_k\right\|_2^2", color=WHITE).scale(1.0).move_to(RIGHT * 3.05 + UP * 0.65)
        formula.set_color_by_tex(r"z_e(x)", ORANGE)
        formula.set_color_by_tex(r"e_k", PURPLE_LIGHT)
        self.play(Write(formula), run_time=1.0)

        line = Line(query.get_center(), nearest.get_center(), color=PURPLE_LIGHT, stroke_width=4)
        self.play(Create(line), nearest.animate.set_color(PURPLE_LIGHT).scale(1.35), Write(nearest_lab), run_time=0.8)
        quant = MathTex(r"z_q=e_{k^*}", color=PURPLE_LIGHT).scale(1.15).move_to(RIGHT * 3.05 + DOWN * 0.65)
        self.play(query.animate.move_to(nearest.get_center()).set_color(PURPLE_LIGHT), FadeOut(qlab), Write(quant), run_time=1.0)
        note = ui_text("quantization replaces the encoder vector with one learned code", 22, GREY).move_to(DOWN * 2.1)
        self.play(FadeIn(note, shift=UP * 0.08), run_time=0.45)
        self.finish()


class StraightThrough(MathShot):
    duration = 14.750
    kicker = "VQ-VAE  /  STRAIGHT-THROUGH ESTIMATOR"

    def construct(self):
        self.panel()
        st = MathTex(r"\tilde z_q=z_e+\operatorname{sg}(z_q-z_e)", color=WHITE).scale(1.2).move_to(UP * 1.15)
        st.set_color_by_tex(r"z_e", ORANGE)
        st.set_color_by_tex(r"z_q", PURPLE_LIGHT)
        self.play(Write(st), run_time=1.0)

        forward_box = RoundedRectangle(width=5.1, height=1.45, corner_radius=0.16, stroke_color=PURPLE, fill_color=BLACK, fill_opacity=0.45).shift(LEFT * 2.75 + DOWN * 0.35)
        backward_box = RoundedRectangle(width=5.1, height=1.45, corner_radius=0.16, stroke_color=ORANGE, fill_color=BLACK, fill_opacity=0.45).shift(RIGHT * 2.75 + DOWN * 0.35)
        ftitle = ui_text("FORWARD", 23, PURPLE_LIGHT, "BOLD").move_to(forward_box.get_center() + UP * 0.42)
        fmath = MathTex(r"\tilde z_q=z_q", color=PURPLE_LIGHT).scale(0.95).move_to(forward_box.get_center() + DOWN * 0.22)
        btitle = ui_text("BACKWARD", 23, ORANGE, "BOLD").move_to(backward_box.get_center() + UP * 0.42)
        bmath = MathTex(r"\nabla_{z_e}\mathcal{L}\approx\nabla_{z_q}\mathcal{L}", color=ORANGE).scale(0.9).move_to(backward_box.get_center() + DOWN * 0.22)
        self.play(Create(forward_box), FadeIn(ftitle), Write(fmath), run_time=0.75)
        self.play(Create(backward_box), FadeIn(btitle), Write(bmath), run_time=0.75)

        sg = ui_text("sg(·) has zero gradient", 22, GREY).move_to(DOWN * 1.55)
        wire = Arrow(LEFT * 1.15 + DOWN * 1.55, RIGHT * 1.15 + DOWN * 1.55, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.08)
        self.play(FadeIn(sg), GrowArrow(wire), run_time=0.6)
        note = ui_text("discrete forward pass, approximate identity gradient backward", 23, WHITE, "BOLD").move_to(DOWN * 2.12)
        self.play(FadeIn(note, shift=UP * 0.08), run_time=0.45)
        self.finish()


class RQResidual(MathShot):
    duration = 9.079
    kicker = "RQ-VAE  /  RESIDUAL QUANTIZATION"

    def construct(self):
        self.panel()
        r0 = MathTex(r"r_0=z_e", color=ORANGE).scale(0.9)
        r1 = MathTex(r"r_1=r_0-e_{k_1}", color=WHITE).scale(0.9)
        r2 = MathTex(r"r_2=r_1-e_{k_2}", color=WHITE).scale(0.9)
        rd = MathTex(r"r_D=r_{D-1}-e_{k_D}", color=WHITE).scale(0.9)
        row = VGroup(r0, r1, r2, rd).arrange(RIGHT, buff=0.45).move_to(UP * 0.9)
        arrows = VGroup(*[
            Arrow(row[i].get_right(), row[i+1].get_left(), buff=0.12, color=GREY, stroke_width=2.4, max_tip_length_to_length_ratio=0.09)
            for i in range(3)
        ])
        self.play(Write(r0), run_time=0.4)
        for i in range(3):
            self.play(GrowArrow(arrows[i]), Write(row[i+1]), run_time=0.55)

        nearest = MathTex(r"k_d=\arg\min_k\|r_{d-1}-e_k\|_2^2", color=PURPLE_LIGHT).scale(0.95).move_to(DOWN * 0.3)
        total = MathTex(r"\hat z=\sum_{d=1}^{D}e_{k_d}", color=WHITE).scale(1.25).move_to(DOWN * 1.45)
        total.set_color_by_tex(r"e_{k_d}", PURPLE_LIGHT)
        self.play(Write(nearest), run_time=0.7)
        self.play(Write(total), run_time=0.75)
        note = ui_text("each code explains what the previous codes left behind", 23, GREY).move_to(DOWN * 2.1)
        self.play(FadeIn(note, shift=UP * 0.08), run_time=0.4)
        self.finish()


class SAESparsity(MathShot):
    duration = 6.372
    kicker = "SPARSE AUTOENCODER  /  TWO COMMON SPARSITY CONTROLS"

    def construct(self):
        self.panel()
        l1 = MathTex(r"\mathcal{L}=\|x-\hat x\|_2^2+\lambda\|a\|_1", color=WHITE).scale(1.12).move_to(UP * 0.85)
        l1.set_color_by_tex(r"\lambda\|a\|_1", PURPLE_LIGHT)
        topk = MathTex(r"a\leftarrow\operatorname{TopK}(a,k)", color=ORANGE).scale(1.15).move_to(DOWN * 0.15)
        self.play(Write(l1), run_time=0.8)
        self.play(Write(topk), run_time=0.65)

        dense = ui_text("many possible coordinates", 23, GREY).move_to(LEFT * 3.2 + DOWN * 1.25)
        sparse = ui_text("few active coordinates", 23, PURPLE_LIGHT, "BOLD").move_to(RIGHT * 3.2 + DOWN * 1.25)
        arrow = Arrow(LEFT * 1.15 + DOWN * 1.25, RIGHT * 1.15 + DOWN * 1.25, color=PURPLE_LIGHT, stroke_width=3, max_tip_length_to_length_ratio=0.08)
        self.play(FadeIn(dense), GrowArrow(arrow), FadeIn(sparse), run_time=0.6)
        warning = ui_text("sparse does not automatically mean monosemantic", 22, ORANGE).move_to(DOWN * 2.05)
        self.play(FadeIn(warning), run_time=0.4)
        self.finish()
