from manim import *
import numpy as np

config.pixel_width = 1280
config.pixel_height = 540
config.frame_rate = 30
config.frame_width = 16
config.frame_height = 6.75
config.background_color = "#050509"
config.disable_caching = True

PURPLE = "#8B5CF6"
PURPLE_LIGHT = "#B89CFF"
ORANGE = "#F97352"
CYAN = "#67E8F9"
GREEN = "#8FF740"
GREY = "#8A8A98"
GREY_DARK = "#292936"
WHITE_SOFT = "#F4F1FF"

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath,amssymb,bm}")


def mathtex(*parts, font_size=48, color=WHITE_SOFT):
    return MathTex(*parts, font_size=font_size, color=color, tex_template=TEX)


def label(text, size=26, color=GREY):
    return Text(text, font_size=size, color=color, weight=MEDIUM)


def pill(text, width=2.4, height=0.78, color=PURPLE, subtitle=None):
    box = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.16,
        stroke_color=WHITE_SOFT,
        stroke_width=2,
        fill_color="#090910",
        fill_opacity=1,
    )
    main = Text(text, font_size=29, color=color, weight=BOLD)
    if subtitle:
        sub = Text(subtitle, font_size=17, color=GREY)
        words = VGroup(main, sub).arrange(DOWN, buff=0.08)
    else:
        words = main
    words.move_to(box)
    return VGroup(box, words)


class FormulaScene(Scene):
    DURATION = 12.0
    TITLE = ""

    def setup(self):
        self.camera.background_color = "#050509"
        self.add(self.dot_field())

    def dot_field(self):
        dots = VGroup()
        xs = np.linspace(-7.7, 7.7, 33)
        ys = np.linspace(-3.12, 3.12, 14)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                phase = (xi * 7 + yi * 11) % 13
                opacity = 0.055 + 0.02 * (phase / 12)
                dots.add(Dot([x, y, 0], radius=0.018, color="#CFE0FF", fill_opacity=opacity, stroke_opacity=0))
        return dots

    def heading(self, text=None):
        txt = Text(text or self.TITLE, font_size=34, color=WHITE_SOFT, weight=BOLD, letter_spacing=0.6)
        bar = Rectangle(width=0.11, height=0.55, fill_color=PURPLE, fill_opacity=1, stroke_width=0)
        group = VGroup(bar, txt).arrange(RIGHT, buff=0.22)
        group.to_edge(UP, buff=0.22).to_edge(LEFT, buff=0.42)
        return group

    def panel(self, width, height, stroke=GREY_DARK):
        return RoundedRectangle(
            width=width,
            height=height,
            corner_radius=0.18,
            stroke_color=stroke,
            stroke_width=2,
            fill_color="#090910",
            fill_opacity=0.94,
        )

    def hold_to_duration(self):
        remaining = self.DURATION - float(self.renderer.time)
        if remaining > 0:
            self.wait(remaining)


class AEObjective(FormulaScene):
    DURATION = 15.0
    TITLE = "AUTOENCODER: ONE DETERMINISTIC PATH"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head, shift=UP * 0.08), run_time=0.5)

        x = pill("x", 1.25, subtitle="input")
        enc = pill("fφ", 1.65, subtitle="encoder")
        z = pill("z", 1.25, color=PURPLE_LIGHT, subtitle="bottleneck")
        dec = pill("gθ", 1.65, subtitle="decoder")
        xh = pill("x̂", 1.25, color=ORANGE, subtitle="rebuild")
        chain = VGroup(x, enc, z, dec, xh).arrange(RIGHT, buff=0.68).shift(UP * 1.18)
        arrows = VGroup(*[
            Arrow(chain[i].get_right(), chain[i + 1].get_left(), buff=0.09, stroke_width=3, max_tip_length_to_length_ratio=0.18, color=WHITE_SOFT)
            for i in range(4)
        ])
        self.play(LaggedStart(*[FadeIn(m, scale=0.92) for m in chain], lag_ratio=0.12), run_time=1.3)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.13), run_time=1.2)

        eq1 = mathtex(r"z", "=", r"f_{\phi}(x)", font_size=49)
        eq2 = mathtex(r"\hat{x}", "=", r"g_{\theta}(z)", font_size=49)
        eqs = VGroup(eq1, eq2).arrange(RIGHT, buff=1.25).shift(DOWN * 0.2)
        eq1[0].set_color(PURPLE_LIGHT)
        eq2[0].set_color(ORANGE)
        self.play(Write(eq1), Write(eq2), run_time=1.5)

        objective = mathtex(
            r"\min_{\phi,\theta}",
            r"\mathbb{E}_{x\sim p_{\mathrm{data}}}",
            r"\left[\,\|x-\hat{x}\|_2^2\,\right]",
            font_size=47,
        ).shift(DOWN * 1.55)
        objective[0].set_color(PURPLE_LIGHT)
        objective[2].set_color(ORANGE)
        rec_label = label("reconstruction error", 24, ORANGE).next_to(objective[2], DOWN, buff=0.18)
        self.play(Write(objective), run_time=1.8)
        self.play(FadeIn(rec_label, shift=UP * 0.08), Circumscribe(objective[2], color=ORANGE, fade_out=True), run_time=1.2)
        self.play(Indicate(z, color=PURPLE_LIGHT), Indicate(objective[0], color=PURPLE_LIGHT), run_time=0.9)
        self.hold_to_duration()


class VAEReparameterization(FormulaScene):
    DURATION = 22.0
    TITLE = "VAE: DISTRIBUTION, THEN A DIFFERENTIABLE SAMPLE"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.5)

        q = mathtex(
            r"q_{\phi}(z\mid x)", "=",
            r"\mathcal{N}\!\left(\mu_{\phi}(x),\,\operatorname{diag}(\sigma_{\phi}^{2}(x))\right)",
            font_size=45,
        ).shift(UP * 1.92)
        q[0].set_color(PURPLE_LIGHT)
        self.play(Write(q), run_time=1.8)

        mu = pill("μ(x)", 2.1, color=PURPLE_LIGHT, subtitle="location")
        sig = pill("σ(x)", 2.1, color=CYAN, subtitle="scale")
        eps = pill("ε", 1.45, color=ORANGE, subtitle="noise")
        boxes = VGroup(mu, sig, eps).arrange(RIGHT, buff=1.0).shift(UP * 0.55)
        self.play(LaggedStart(*[FadeIn(b, scale=0.9) for b in boxes], lag_ratio=0.18), run_time=1.4)

        eps_eq = mathtex(r"\varepsilon", r"\sim", r"\mathcal N(0,I)", font_size=46).next_to(eps, DOWN, buff=0.35)
        eps_eq[0].set_color(ORANGE)
        self.play(Write(eps_eq), run_time=1.0)

        z_eq = mathtex(r"z", "=", r"\mu_{\phi}(x)", "+", r"\sigma_{\phi}(x)", r"\odot", r"\varepsilon", font_size=54)
        z_eq.shift(DOWN * 1.56)
        z_eq[0].set_color(PURPLE_LIGHT)
        z_eq[2].set_color(PURPLE_LIGHT)
        z_eq[4].set_color(CYAN)
        z_eq[6].set_color(ORANGE)
        self.play(Write(z_eq), run_time=1.8)

        arrows = VGroup(
            Arrow(mu.get_bottom(), z_eq[2].get_top(), buff=0.1, color=PURPLE_LIGHT, stroke_width=3),
            Arrow(sig.get_bottom(), z_eq[4].get_top(), buff=0.1, color=CYAN, stroke_width=3),
            Arrow(eps.get_bottom(), z_eq[6].get_top(), buff=0.1, color=ORANGE, stroke_width=3),
        )
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.16), run_time=1.4)

        brace = Brace(z_eq[2:], DOWN, color=WHITE_SOFT)
        note = label("randomness is isolated in ε; gradients still reach μ and σ", 24, WHITE_SOFT).next_to(brace, DOWN, buff=0.18)
        self.play(GrowFromCenter(brace), FadeIn(note, shift=UP * 0.08), run_time=1.2)
        self.play(Indicate(z_eq[2], color=PURPLE_LIGHT), Indicate(z_eq[4], color=CYAN), run_time=1.0)
        self.hold_to_duration()


class VAEELBO(FormulaScene):
    DURATION = 22.5
    TITLE = "THE ELBO: FIT THE DATA, REGULARIZE THE CODE"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.5)

        bound = mathtex(r"\log p_{\theta}(x)", r"\ge", r"\mathcal L_{\mathrm{ELBO}}(x)", font_size=51).shift(UP * 2.0)
        bound[2].set_color(PURPLE_LIGHT)
        self.play(Write(bound), run_time=1.4)

        rec_box = self.panel(6.5, 2.0, PURPLE).shift(LEFT * 3.55 + UP * 0.35)
        kl_box = self.panel(6.5, 2.0, ORANGE).shift(RIGHT * 3.55 + UP * 0.35)
        rec_title = label("RECONSTRUCTION", 26, PURPLE_LIGHT).next_to(rec_box.get_top(), DOWN, buff=0.22)
        kl_title = label("RATE / REGULARIZATION", 26, ORANGE).next_to(kl_box.get_top(), DOWN, buff=0.22)
        rec_eq = mathtex(r"\mathbb E_{q_{\phi}(z\mid x)}", r"[\log p_{\theta}(x\mid z)]", font_size=38).move_to(rec_box).shift(DOWN * 0.18)
        kl_eq = mathtex(r"D_{\mathrm{KL}}", r"\!\left(q_{\phi}(z\mid x)\,\|\,p(z)\right)", font_size=38).move_to(kl_box).shift(DOWN * 0.18)
        rec_eq[0].set_color(PURPLE_LIGHT)
        kl_eq[0].set_color(ORANGE)
        self.play(Create(rec_box), Create(kl_box), FadeIn(rec_title), FadeIn(kl_title), run_time=1.4)
        self.play(Write(rec_eq), Write(kl_eq), run_time=1.7)

        elbo = mathtex(
            r"\mathcal L_{\mathrm{ELBO}}",
            "=",
            r"\underbrace{\mathbb E_q[\log p_{\theta}(x\mid z)]}_{\text{preserve }x}",
            "-",
            r"\beta\,\underbrace{D_{\mathrm{KL}}(q_{\phi}(z\mid x)\|p(z))}_{\text{shape latent space}}",
            font_size=38,
        ).shift(DOWN * 1.65)
        elbo[0].set_color(PURPLE_LIGHT)
        elbo[2].set_color(PURPLE_LIGHT)
        elbo[4].set_color(ORANGE)
        self.play(Write(elbo), run_time=2.1)

        loss = mathtex(r"\min\;\mathcal J", "=", r"\mathcal L_{\mathrm{rec}}", "+", r"\beta\mathcal L_{\mathrm{KL}}", font_size=45).shift(DOWN * 2.62)
        loss[2].set_color(PURPLE_LIGHT)
        loss[4].set_color(ORANGE)
        self.play(TransformFromCopy(elbo, loss), run_time=1.6)
        self.play(Indicate(rec_box, color=PURPLE_LIGHT), Indicate(kl_box, color=ORANGE), run_time=1.2)
        self.hold_to_duration()


class RateDistortion(FormulaScene):
    DURATION = 13.5
    TITLE = "RATE–DISTORTION IS A CURVE, NOT A SINGLE SCORE"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.45)

        objective = mathtex(r"\min", r"D", "+", r"\beta R", font_size=56).to_edge(UP, buff=0.92)
        objective[1].set_color(PURPLE_LIGHT)
        objective[3].set_color(ORANGE)
        defs = VGroup(
            mathtex(r"D=\mathbb E[-\log p_{\theta}(x\mid z)]", font_size=31, color=PURPLE_LIGHT),
            mathtex(r"R=\mathbb E[D_{\mathrm{KL}}(q(z\mid x)\|p(z))]", font_size=31, color=ORANGE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_edge(RIGHT, buff=0.55).shift(UP * 0.65)
        self.play(Write(objective), FadeIn(defs), run_time=1.2)

        axes = Axes(
            x_range=[0, 6, 1], y_range=[0, 4.2, 1],
            x_length=7.1, y_length=4.1,
            axis_config={"color": GREY, "stroke_width": 2, "include_ticks": False},
            tips=True,
        ).shift(LEFT * 2.75 + DOWN * 0.62)
        xlab = mathtex("R", font_size=38, color=ORANGE).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        ylab = mathtex("D", font_size=38, color=PURPLE_LIGHT).next_to(axes.y_axis.get_end(), UP, buff=0.1)
        curve = axes.plot(lambda x: 3.6 / (x + 0.75) + 0.18, x_range=[0.35, 5.7], color=WHITE_SOFT, stroke_width=5)
        self.play(Create(axes), FadeIn(xlab), FadeIn(ylab), Create(curve), run_time=1.8)

        start = axes.c2p(0.85, 3.6 / 1.6 + 0.18)
        end = axes.c2p(4.6, 3.6 / 5.35 + 0.18)
        dot = Dot(start, radius=0.12, color=ORANGE)
        tag_hi = label("large β: little information", 21, ORANGE).next_to(dot, UP + RIGHT, buff=0.18)
        self.play(FadeIn(dot, scale=0.5), FadeIn(tag_hi), run_time=0.7)
        self.play(dot.animate.move_to(end), FadeOut(tag_hi), run_time=2.3, rate_func=smooth)
        tag_lo = label("small β: better fit, less regularity", 21, PURPLE_LIGHT).next_to(dot, DOWN + RIGHT, buff=0.18)
        self.play(FadeIn(tag_lo), run_time=0.6)
        self.hold_to_duration()


class VQQuantization(FormulaScene):
    DURATION = 18.0
    TITLE = "VQ-VAE: NEAREST CODE, THREE DISTINCT LOSSES"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.45)

        quant = mathtex(
            r"k^{*}", "=", r"\arg\min_j\|z_e-e_j\|_2^2",
            r",\qquad", r"z_q=e_{k^{*}}",
            font_size=44,
        ).shift(UP * 2.05)
        quant[0].set_color(PURPLE_LIGHT)
        quant[2].set_color(ORANGE)
        quant[4].set_color(PURPLE_LIGHT)
        self.play(Write(quant), run_time=1.5)

        plane = NumberPlane(
            x_range=[-3, 3, 1], y_range=[-2, 2, 1],
            x_length=5.4, y_length=3.7,
            background_line_style={"stroke_color": GREY_DARK, "stroke_width": 1, "stroke_opacity": 0.7},
            axis_config={"stroke_opacity": 0},
        ).shift(LEFT * 4.6 + DOWN * 0.25)
        code_coords = [(-2.1, -1.0), (-1.6, 1.1), (-0.2, -0.65), (0.3, 1.25), (1.6, 0.1), (2.1, -1.25)]
        codes = VGroup(*[Dot(plane.c2p(x, y), radius=0.105, color=PURPLE_LIGHT) for x, y in code_coords])
        ze = Dot(plane.c2p(0.85, 0.62), radius=0.13, color=ORANGE)
        ze_label = mathtex(r"z_e", font_size=30, color=ORANGE).next_to(ze, UP, buff=0.1)
        target = codes[4]
        snap = Arrow(ze.get_center(), target.get_center(), buff=0.14, color=ORANGE, stroke_width=4)
        self.play(FadeIn(plane), LaggedStart(*[FadeIn(c, scale=0.5) for c in codes], lag_ratio=0.08), FadeIn(ze), FadeIn(ze_label), run_time=1.4)
        self.play(GrowArrow(snap), ze.animate.move_to(target.get_center()), FadeOut(ze_label), run_time=1.1)

        rec = self.panel(7.5, 0.9, PURPLE).shift(RIGHT * 2.9 + UP * 0.75)
        cb = self.panel(7.5, 0.9, ORANGE).shift(RIGHT * 2.9 + DOWN * 0.35)
        com = self.panel(7.5, 0.9, CYAN).shift(RIGHT * 2.9 + DOWN * 1.45)
        rec_eq = mathtex(r"\mathcal L_{\mathrm{rec}}=\|x-\hat{x}\|_2^2", font_size=35).move_to(rec)
        cb_eq = mathtex(r"\mathcal L_{\mathrm{code}}=\|\operatorname{sg}[z_e]-e_{k^*}\|_2^2", font_size=34).move_to(cb)
        com_eq = mathtex(r"\mathcal L_{\mathrm{commit}}=\|z_e-\operatorname{sg}[e_{k^*}]\|_2^2", font_size=32).move_to(com)
        rec_eq.set_color(PURPLE_LIGHT)
        cb_eq.set_color(ORANGE)
        com_eq.set_color(CYAN)
        self.play(LaggedStart(Create(rec), Write(rec_eq), Create(cb), Write(cb_eq), Create(com), Write(com_eq), lag_ratio=0.1), run_time=2.6)

        total = mathtex(r"\mathcal L=\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{code}}+\beta\mathcal L_{\mathrm{commit}}", font_size=37).to_edge(DOWN, buff=0.2)
        total.set_color(WHITE_SOFT)
        self.play(Write(total), run_time=1.2)
        self.play(Indicate(rec, color=PURPLE_LIGHT), Indicate(cb, color=ORANGE), Indicate(com, color=CYAN), run_time=1.1)
        self.hold_to_duration()


class StraightThrough(FormulaScene):
    DURATION = 17.0
    TITLE = "STRAIGHT-THROUGH: DISCRETE FORWARD, IDENTITY BACKWARD"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.45)

        forward_box = self.panel(6.7, 2.0, PURPLE).shift(LEFT * 3.55 + UP * 0.55)
        backward_box = self.panel(6.7, 2.0, ORANGE).shift(RIGHT * 3.55 + UP * 0.55)
        ft = label("FORWARD PASS", 27, PURPLE_LIGHT).next_to(forward_box.get_top(), DOWN, buff=0.2)
        bt = label("BACKWARD PASS", 27, ORANGE).next_to(backward_box.get_top(), DOWN, buff=0.2)
        feq = mathtex(r"z_{\mathrm{st}}=z_q", font_size=49, color=PURPLE_LIGHT).move_to(forward_box).shift(DOWN * 0.15)
        beq = mathtex(r"\frac{\partial z_{\mathrm{st}}}{\partial z_e}=I", font_size=49, color=ORANGE).move_to(backward_box).shift(DOWN * 0.15)
        self.play(Create(forward_box), Create(backward_box), FadeIn(ft), FadeIn(bt), run_time=1.2)
        self.play(Write(feq), Write(beq), run_time=1.5)

        identity = mathtex(
            r"z_{\mathrm{st}}", "=", r"z_e", "+",
            r"\operatorname{sg}\!\left(z_q-z_e\right)",
            font_size=55,
        ).shift(DOWN * 1.45)
        identity[0].set_color(PURPLE_LIGHT)
        identity[2].set_color(ORANGE)
        identity[4].set_color(CYAN)
        self.play(Write(identity), run_time=1.8)

        brace = Brace(identity[4], DOWN, color=CYAN)
        frozen = label("stop-gradient: value passes, derivative does not", 24, CYAN).next_to(brace, DOWN, buff=0.15)
        self.play(GrowFromCenter(brace), FadeIn(frozen), run_time=1.0)

        f_arrow = Arrow(forward_box.get_bottom(), identity.get_top(), buff=0.15, color=PURPLE_LIGHT, stroke_width=4)
        b_arrow = CurvedArrow(identity.get_right() + DOWN * 0.15, backward_box.get_bottom() + DOWN * 0.05, angle=0.6, color=ORANGE, stroke_width=4)
        self.play(GrowArrow(f_arrow), Create(b_arrow), run_time=1.1)
        self.play(Indicate(identity[4], color=CYAN), Indicate(beq, color=ORANGE), run_time=1.1)
        self.hold_to_duration()


class RQResidual(FormulaScene):
    DURATION = 31.0
    TITLE = "RQ-VAE: EACH CODEBOOK EXPLAINS THE RESIDUAL"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.45)

        equations = VGroup(
            mathtex(r"r_0=z_e", font_size=39),
            mathtex(r"k_{\ell}=\arg\min_j\|r_{\ell-1}-e_j^{(\ell)}\|_2^2", font_size=36),
            mathtex(r"r_{\ell}=r_{\ell-1}-e_{k_{\ell}}^{(\ell)}", font_size=38),
            mathtex(r"z_q=\sum_{\ell=1}^{L}e_{k_{\ell}}^{(\ell)}", font_size=42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34).shift(RIGHT * 3.65 + DOWN * 0.15)
        equations[0].set_color(WHITE_SOFT)
        equations[1].set_color(PURPLE_LIGHT)
        equations[2].set_color(ORANGE)
        equations[3].set_color(CYAN)
        panel = self.panel(7.7, 4.9, PURPLE).move_to(equations).shift(UP * 0.03)
        self.play(Create(panel), run_time=0.7)
        self.play(LaggedStart(*[Write(e) for e in equations], lag_ratio=0.17), run_time=3.0)

        origin = np.array([-6.5, -1.75, 0])
        p1 = origin + np.array([2.6, 1.4, 0])
        p2 = p1 + np.array([1.45, 0.6, 0])
        p3 = p2 + np.array([0.72, 0.28, 0])
        target = origin + np.array([5.15, 2.45, 0])

        full = Arrow(origin, target, buff=0, color=WHITE_SOFT, stroke_width=7, max_tip_length_to_length_ratio=0.08)
        full_lab = mathtex(r"r_0=z_e", font_size=33).next_to(full, UP, buff=0.14)
        self.play(GrowArrow(full), FadeIn(full_lab), run_time=1.4)

        e1 = Arrow(origin, p1, buff=0, color=PURPLE_LIGHT, stroke_width=7, max_tip_length_to_length_ratio=0.12)
        r1 = Arrow(p1, target, buff=0, color=ORANGE, stroke_width=5, max_tip_length_to_length_ratio=0.1)
        l1 = mathtex(r"e_{k_1}^{(1)}", font_size=29, color=PURPLE_LIGHT).next_to(e1, DOWN, buff=0.1)
        lr1 = mathtex(r"r_1", font_size=29, color=ORANGE).next_to(r1, UP, buff=0.08)
        self.play(FadeOut(full), FadeOut(full_lab), GrowArrow(e1), GrowArrow(r1), FadeIn(l1), FadeIn(lr1), run_time=1.7)

        e2 = Arrow(p1, p2, buff=0, color=CYAN, stroke_width=7, max_tip_length_to_length_ratio=0.18)
        r2 = Arrow(p2, target, buff=0, color=ORANGE, stroke_width=4, max_tip_length_to_length_ratio=0.13)
        l2 = mathtex(r"e_{k_2}^{(2)}", font_size=27, color=CYAN).next_to(e2, DOWN, buff=0.08)
        lr2 = mathtex(r"r_2", font_size=27, color=ORANGE).next_to(r2, UP, buff=0.07)
        self.play(Transform(r1, r2), Transform(lr1, lr2), GrowArrow(e2), FadeIn(l2), run_time=1.7)

        e3 = Arrow(p2, p3, buff=0, color=GREEN, stroke_width=7, max_tip_length_to_length_ratio=0.25)
        r3 = Arrow(p3, target, buff=0, color=ORANGE, stroke_width=3, max_tip_length_to_length_ratio=0.17)
        l3 = mathtex(r"e_{k_3}^{(3)}", font_size=25, color=GREEN).next_to(e3, DOWN, buff=0.06)
        lr3 = mathtex(r"r_3", font_size=25, color=ORANGE).next_to(r3, UP, buff=0.05)
        self.play(Transform(r1, r3), Transform(lr1, lr3), GrowArrow(e3), FadeIn(l3), run_time=1.7)

        sum_arrow = VGroup(e1.copy(), e2.copy(), e3.copy())
        payoff = label("coarse code + correction + refinement", 25, WHITE_SOFT).next_to(sum_arrow, DOWN, buff=0.42)
        self.play(FadeIn(payoff, shift=UP * 0.08), Circumscribe(equations[3], color=CYAN, fade_out=True), run_time=1.2)
        self.play(Indicate(r1, color=ORANGE), Indicate(equations[2], color=ORANGE), run_time=1.0)
        self.hold_to_duration()


class SAESparsity(FormulaScene):
    DURATION = 31.0
    TITLE = "SAE: OVERCOMPLETE CODE, FEW ACTIVE COORDINATES"

    def construct(self):
        head = self.heading()
        self.play(FadeIn(head), run_time=0.45)

        encode = mathtex(r"h=\operatorname{ReLU}(W_e x+b_e)", font_size=42, color=PURPLE_LIGHT)
        decode = mathtex(r"\hat{x}=W_dh+b_d", font_size=42, color=CYAN)
        pipeline = VGroup(encode, decode).arrange(RIGHT, buff=1.0).shift(UP * 1.9)
        self.play(Write(encode), Write(decode), run_time=1.8)

        base_y = -1.45
        xs = np.linspace(-6.6, -0.6, 16)
        dense_vals = [1.1, 1.8, 0.9, 2.25, 1.35, 1.65, 0.75, 2.0, 1.15, 1.9, 0.85, 1.55, 1.0, 2.15, 1.45, 0.7]
        sparse_vals = [0, 0, 0, 2.25, 0, 0, 0, 0, 0, 1.9, 0, 0, 0, 2.15, 0, 0]
        dense = VGroup(*[
            Rectangle(width=0.24, height=v, stroke_width=0, fill_color=GREY, fill_opacity=0.72).move_to([x, base_y + v / 2, 0])
            for x, v in zip(xs, dense_vals)
        ])
        sparse = VGroup(*[
            Rectangle(width=0.24, height=max(v, 0.015), stroke_width=0, fill_color=(PURPLE_LIGHT if v > 0 else GREY_DARK), fill_opacity=(1 if v > 0 else 0.35)).move_to([x, base_y + max(v, 0.015) / 2, 0])
            for x, v in zip(xs, sparse_vals)
        ])
        axis = Line([-6.9, base_y, 0], [-0.3, base_y, 0], color=GREY, stroke_width=2)
        dense_lab = label("dense activations", 23, GREY).next_to(axis, DOWN, buff=0.2)
        self.play(Create(axis), LaggedStart(*[GrowFromEdge(b, DOWN) for b in dense], lag_ratio=0.035), FadeIn(dense_lab), run_time=1.9)
        sparse_lab = label("sparse feature code", 23, PURPLE_LIGHT).move_to(dense_lab)
        self.play(Transform(dense, sparse), Transform(dense_lab, sparse_lab), run_time=1.7)

        objective_panel = self.panel(7.3, 2.35, PURPLE).shift(RIGHT * 3.75 + DOWN * 0.38)
        objective = mathtex(
            r"\min_{W_e,W_d}",
            r"\|x-\hat{x}\|_2^2",
            "+",
            r"\lambda\|h\|_1",
            font_size=42,
        ).move_to(objective_panel).shift(UP * 0.34)
        objective[1].set_color(CYAN)
        objective[3].set_color(ORANGE)
        alt = mathtex(r"\text{or}\qquad h\leftarrow\operatorname{TopK}(h,k)", font_size=35).move_to(objective_panel).shift(DOWN * 0.62)
        alt.set_color(PURPLE_LIGHT)
        self.play(Create(objective_panel), Write(objective), run_time=1.6)
        self.play(Write(alt), run_time=1.0)

        over = mathtex(r"d_h>d_x", font_size=44, color=WHITE_SOFT).next_to(objective_panel, UP, buff=0.3)
        over_note = label("overcomplete dictionary", 22, GREY).next_to(over, RIGHT, buff=0.25)
        self.play(FadeIn(over, shift=UP * 0.08), FadeIn(over_note), run_time=0.8)
        self.play(Indicate(objective[3], color=ORANGE), Indicate(dense, color=PURPLE_LIGHT), run_time=1.2)
        self.hold_to_duration()
