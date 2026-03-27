\documentclass[11pt]{article}

\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{physics}
\usepackage{microtype}
\usepackage[dvipsnames]{xcolor}
\usepackage[most]{tcolorbox}
\usepackage{enumitem}
\usepackage{titlesec}

% ---------- Section formatting ----------
\titleformat{\section}{\Large\bfseries}{\thesection}{0.75em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.75em}{}
\titlespacing*{\section}{0pt}{1.2ex plus .2ex minus .2ex}{0.8ex}
\titlespacing*{\subsection}{0pt}{1.0ex plus .2ex minus .2ex}{0.5ex}

% ---------- Step box environment ----------
\newcounter{derivstep}

\newtcolorbox{stepbox}[1]{
    breakable,
    enhanced,
    colback=blue!2,
    colframe=blue!55!black,
    boxrule=0.7pt,
    arc=1.5mm,
    left=2mm,right=2mm,top=1mm,bottom=1mm,
    title={\bfseries Step \thederivstep\;--\;#1},
    coltitle=black,
    colbacktitle=blue!10,
    fonttitle=\bfseries
}

\newenvironment{derivationstep}[1]
{
    \refstepcounter{derivstep}
    \begin{stepbox}{#1}
}
{
    \end{stepbox}
}

% ---------- Shortcuts ----------
\newcommand{\R}{\mathbb{R}}
\newcommand{\Ac}{A_c}
\newcommand{\Bc}{B_c}
\newcommand{\Cc}{C_c}
\newcommand{\Dc}{D_c}
\newcommand{\Ad}{A_d}
\newcommand{\Bd}{B_d}
\newcommand{\Td}{T_d}

\begin{document}

\begin{center}
    {\LARGE \textbf{Formatted Derivation Notes}}\\[0.4em]
    {\large Dual-Gantry Model Derivation}
\end{center}

\vspace{0.8em}

\section*{Derivation}

\begin{derivationstep}{Equations of motion (logical coordinates)}

The gantry system with
\[
q = \begin{bmatrix} X & \Theta & Y \end{bmatrix}^{\top}
\]
and logical forces \(f_{\text{logical}}\) is given by
\[
M(Y)\ddot{q} + C_{\text{damp}}\dot{q} + Kq = f_{\text{logical}}.
\]

With
\[
M(Y)=
\begin{bmatrix}
m_1+m_2+m_b+m_h &
(m_1-m_2)\dfrac{L_b}{2} - m_h Y &
0
\\[0.6em]
(m_1-m_2)\dfrac{L_b}{2} - m_h Y &
J_b+J_h+(m_1+m_2)\dfrac{L_b^2}{4}+m_h d^2+m_h Y^2 &
-m_h d
\\[0.6em]
0 & -m_h d & m_h
\end{bmatrix},
\]

\[
C_{\text{damp}}=
\begin{bmatrix}
c_{g1}+c_{g2} &
(c_{g1}-c_{g2})\dfrac{L_b}{2} &
0
\\[0.6em]
(c_{g1}-c_{g2})\dfrac{L_b}{2} &
c_{b1}+c_{b2}+(c_{g1}+c_{g2})\dfrac{L_b^2}{4} &
0
\\[0.6em]
0 & 0 & c_y
\end{bmatrix},
\qquad
K=
\begin{bmatrix}
0 & 0 & 0 \\
0 & k_{b1}+k_{b2} & 0 \\
0 & 0 & 0
\end{bmatrix}.
\]

\end{derivationstep}

\begin{derivationstep}{First-order state-space (logical coordinates)}

Define the state
\[
x =
\begin{bmatrix}
q^{\top} & \dot{q}^{\top}
\end{bmatrix}^{\top}
=
\begin{bmatrix}
X & \Theta & Y & \dot{X} & \dot{\Theta} & \dot{Y}
\end{bmatrix}^{\top}
\in \R^6.
\]

Rewrite the second-order system as
\[
\ddot{q} = -M(Y)^{-1}C_{\text{damp}}\dot{q} - M(Y)^{-1}Kq + M(Y)^{-1}f_{\text{logical}}.
\]

Then
\[
\dot{x} = A_c(Y)x + B_c f_{\text{logical}},
\qquad
y = C_c x,
\]
with
\[
A_c(Y)=
\begin{bmatrix}
0_{3\times 3} & I_3 \\
-M(Y)^{-1}K & -M(Y)^{-1}C_{\text{damp}}
\end{bmatrix}
\in \R^{6\times 6},
\]
\[
B_c=
\begin{bmatrix}
0_{3\times 3} \\
M(Y)^{-1}
\end{bmatrix}
\in \R^{6\times 3},
\qquad
C_c=
\begin{bmatrix}
I_3 & 0_{3\times 3}
\end{bmatrix}
\in \R^{3\times 6},
\qquad
D_c = 0_{3\times 3}.
\]

\textbf{Note.}
\(A_c(Y)\) is singular. The top-left block is identically zero, and the zero rows of \(K\)
(the \(X\)- and \(Y\)-coordinates have no stiffness) propagate to \(\det(A_c)=0\).

\end{derivationstep}

\begin{derivationstep}{Stage coordinate transform}

Stage forces and positions relate to logical coordinates via
\[
P=
\begin{bmatrix}
1 & 1 & 0 \\
\dfrac{L_b}{2} & -\dfrac{L_b}{2} & 0 \\
0 & 0 & 1
\end{bmatrix},
\qquad
f_{\text{logical}} = P f_{\text{stage}},
\qquad
q_{\text{stage}} = P^{\top} q_{\text{logical}}.
\]

The internal states remain in logical coordinates. Substituting
\(f_{\text{logical}} = Pu\) and \(y_{\text{stage}} = P^\top y\) gives
\[
B_{c,\text{stage}} = B_c P \in \R^{6\times 3},
\]
\[
C_{c,\text{stage}} = P^\top C_c \in \R^{3\times 6},
\]
while
\[
A_c(Y) \quad \text{remains unchanged.}
\]

\end{derivationstep}

\begin{derivationstep}{ZOH discretization (Tóth complete method)}

Under Assumption 1, \(u(t)\) and \(Y(t)\approx Y[k]\) are held constant on
\[
[kT_d,(k+1)T_d).
\]

The continuous-time system therefore reduces to an LTI ODE on each interval. Solving exactly:
\[
x[k+1]
=
\underbrace{e^{A_c(Y[k])T_d}}_{A_d(Y[k])}x[k]
+
\underbrace{\left[\int_0^{T_d} e^{A_c(Y[k])\tau}\, d\tau\right] B_{c,\text{stage}}}_{B_d(Y[k])}u[k].
\]

The output equation is
\[
y[k] = C_{c,\text{stage}}x[k],
\qquad
D_d = 0.
\]

\end{derivationstep}

\begin{derivationstep}{Why the naive \(B_d\) formula fails}

When \(A_c\) is invertible, the integral simplifies as
\[
\int_0^{T_d} e^{A_c \tau}\, d\tau
=
A_c^{-1}(e^{A_c T_d}-I),
\]
which gives
\[
B_d = A_c^{-1}(A_d-I)\,B_{c,\text{stage}}.
\]

However, in our case \(A_c\) is singular, so \(A_c^{-1}\) does not exist.
Therefore, this formula is undefined.

\end{derivationstep}

\begin{derivationstep}{Augmented matrix exponential (general derivation)}

Form the augmented \((n+m)\times(n+m)=9\times 9\) matrix
\[
\mathcal{M}=
\begin{bmatrix}
A_c & B_{c,\text{stage}} \\
0_{3\times 6} & 0_{3\times 3}
\end{bmatrix}.
\]

For \(k\ge 1\),
\[
\mathcal{M}^k=
\begin{bmatrix}
A_c^k & A_c^{k-1}B_{c,\text{stage}} \\
0 & 0
\end{bmatrix}.
\]

Using the Taylor expansion,
\[
e^{\mathcal{M}T_d}
=
I+\sum_{k=1}^{\infty}\frac{\mathcal{M}^kT_d^k}{k!}
=
\begin{bmatrix}
\displaystyle I+\sum_{k=1}^{\infty}\frac{A_c^kT_d^k}{k!}
&
\displaystyle \sum_{k=1}^{\infty}\frac{A_c^{k-1}T_d^k}{k!}B_{c,\text{stage}}
\\[1.2em]
0 & I
\end{bmatrix}.
\]

The top-left block is
\[
e^{A_cT_d}=A_d.
\]

For the top-right block,
\[
\sum_{k=1}^{\infty}\frac{A_c^{k-1}T_d^k}{k!}
=
\int_0^{T_d}
\sum_{k=1}^{\infty}
\frac{A_c^{k-1}\tau^{k-1}}{(k-1)!}\, d\tau
=
\int_0^{T_d} e^{A_c\tau}\, d\tau.
\]

Therefore,
\[
\boxed{
e^{\mathcal{M}T_d}
=
\begin{bmatrix}
A_d & B_d \\
0 & I
\end{bmatrix}
}
\]
with no inversion of \(A_c\) required. Hence this method remains valid for singular \(A_c\).

This is exactly what \verb|torch.linalg.matrix_exp(M_aug * ts)| computes.

\end{derivationstep}

\end{document}