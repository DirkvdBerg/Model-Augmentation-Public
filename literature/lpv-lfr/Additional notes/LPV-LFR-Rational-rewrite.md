\documentclass[11pt]{article}

\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage[hidelinks]{hyperref}

\newcommand{\R}{\mathbb{R}}
\newcommand{\adj}{\operatorname{adj}}
\newcommand{\diag}{\operatorname{diag}}

\title{Rational / Polynomial Rewrite of the LPV-LFR Loop Matrix\\
\large Derivation for \boldmath{$L(Y) = I - D_{zw}\Delta(Y)$}}
\author{}
\date{}

\begin{document}
\maketitle

% -----------------------------------------------------------------------
\section*{Purpose}
% -----------------------------------------------------------------------

This note derives a symbolic rational/polynomial representation of the
LPV-LFR loop matrix
\[
    L(Y) := I - D_{zw}\Delta(Y)
\]
for the gantry baseline realization. The model remains in LPV-LFR form
throughout; the loop matrix is the primary object of analysis, and
$M(Y)$ appears only as an algebraic consequence of its Schur complement
structure.

Throughout this note the following assumption is in force.

\medskip
\noindent\textbf{Assumption~A.}\quad
$M_0$ is nonsingular. This holds for all physically admissible parameters
because $M_0 = M(0)$ is the nominal inertia matrix, which is positive
definite.

% -----------------------------------------------------------------------
\section{Mechanical Baseline Model}
% -----------------------------------------------------------------------

Let
\[
    q = \begin{bmatrix} X \\ \Theta \\ Y \end{bmatrix} \in \R^3,
    \qquad
    x = \begin{bmatrix} q \\ \dot{q} \end{bmatrix} \in \R^6,
    \qquad
    u \in \R^3.
\]
The second-order mechanical model is
\begin{equation}
    M(Y)\ddot{q} + C\dot{q} + Kq = u,
    \label{eq:mech}
\end{equation}
where $C,K\in\R^{3\times3}$ are the constant damping and stiffness matrices.
The net force term is
\begin{equation}
    f_{\mathrm{net}} := [-K,\,-C]\,x + u.
    \label{eq:fnet}
\end{equation}

\subsection{Inertia Matrix \boldmath{$M(Y)$}}

\begin{equation}
    M(Y) =
    \begin{bmatrix}
        m_1+m_2+m_b+m_h &
        \dfrac{(m_1-m_2)L_b}{2} - m_hY &
        0 \\[2mm]
        \dfrac{(m_1-m_2)L_b}{2} - m_hY &
        J_b+J_h+\dfrac{(m_1+m_2)L_b^2}{4}+m_hd^2+m_hY^2 &
        -m_hd \\[2mm]
        0 & -m_hd & m_h
    \end{bmatrix}.
    \label{eq:MY}
\end{equation}
$M(Y)$ is decomposed as
\begin{equation}
    M(Y) = M_0 + YM_1 + Y^2M_2,
    \label{eq:MYdecomp}
\end{equation}
with
\begin{equation}
    M_0 =
    \begin{bmatrix}
        m_1+m_2+m_b+m_h &
        \dfrac{(m_1-m_2)L_b}{2} &
        0 \\[2mm]
        \dfrac{(m_1-m_2)L_b}{2} &
        J_b+J_h+\dfrac{(m_1+m_2)L_b^2}{4}+m_hd^2 &
        -m_hd \\[2mm]
        0 & -m_hd & m_h
    \end{bmatrix},
    \label{eq:M0}
\end{equation}
\begin{equation}
    M_1 =
    \begin{bmatrix}
        0 & -m_h & 0 \\
        -m_h & 0 & 0 \\
        0 & 0 & 0
    \end{bmatrix},
    \qquad
    M_2 =
    \begin{bmatrix}
        0 & 0 & 0 \\
        0 & m_h & 0 \\
        0 & 0 & 0
    \end{bmatrix}.
    \label{eq:M1M2}
\end{equation}
By Assumption~A, $M_0^{-1}$ exists and all LFR matrices below are
well defined.

% -----------------------------------------------------------------------
\section{LPV-LFR Realization}
% -----------------------------------------------------------------------

The LPV-LFR realization is the interconnection
\begin{align}
    \dot{x} &= A_x x + B_w w + B_u u,
    \label{eq:lfrstate}\\
    z       &= C_z x + D_{zw} w + D_{zu} u,
    \label{eq:lfrz}\\
    y       &= C_y x + D_{yw} w + D_{yu} u,
    \label{eq:lfry}\\
    w       &= \Delta(Y)\,z.
    \label{eq:lfrw}
\end{align}
All matrices collected in $G$ are constant; the scheduling enters only
through $w = \Delta(Y)z$.

The constant matrices below are obtained by decomposing
$M(Y) = M_0 + YM_1 + Y^2M_2$ in \eqref{eq:MYdecomp} and pre-multiplying
the equations of motion by $M_0^{-1}$ to isolate the nominal dynamics
from the scheduling-dependent residuals (see the companion realization
note for the full construction).

\medskip\noindent
\textbf{State and input matrices:}
\begin{equation}
    A_x =
    \begin{bmatrix}
        0 & I_3 \\
        -M_0^{-1}K & -M_0^{-1}C
    \end{bmatrix},
    \quad
    B_w =
    \begin{bmatrix}
        0 & 0 \\
        -M_0^{-1}M_1 & -M_0^{-1}M_2
    \end{bmatrix},
    \quad
    B_u =
    \begin{bmatrix}
        0 \\ M_0^{-1}
    \end{bmatrix}.
    \label{eq:ABmats}
\end{equation}

\noindent
\textbf{Internal output matrices:}
\begin{equation}
    C_z =
    \begin{bmatrix}
        -M_0^{-1}K & -M_0^{-1}C \\
        0 & 0
    \end{bmatrix},
    \quad
    D_{zw} =
    \begin{bmatrix}
        -M_0^{-1}M_1 & -M_0^{-1}M_2 \\
        I_3 & 0
    \end{bmatrix},
    \quad
    D_{zu} =
    \begin{bmatrix}
        M_0^{-1} \\ 0
    \end{bmatrix}.
    \label{eq:CDmats}
\end{equation}

\noindent
\textbf{External output matrices:}
\begin{equation}
    C_y = \begin{bmatrix} I_3 & 0 \end{bmatrix},
    \qquad
    D_{yw} = 0,
    \qquad
    D_{yu} = 0.
    \label{eq:Cymats}
\end{equation}

\noindent
\textbf{Scheduling block:}
\begin{equation}
    \Delta(Y) = YI_6.
    \label{eq:Delta}
\end{equation}
The six scheduling channels arise because the latent variables are
$z = [a;\,a_1]\in\R^6$ with $a_1 = Ya$ (see the companion note), so
applying the single scheduler $Y$ to each component gives
$\Delta(Y) = YI_6$.

% -----------------------------------------------------------------------
\section{LPV-LFR Loop Matrix}
% -----------------------------------------------------------------------

Since $D_{zw}\neq 0$, the loop equations \eqref{eq:lfrz} and
\eqref{eq:lfrw} combine into the algebraic equation
\begin{equation}
    L(Y)\,z = C_z x + D_{zu} u,
    \label{eq:loopAlg}
\end{equation}
which must be solved for $z$ at each time instant. The loop matrix is
\begin{equation}
    L(Y) := I - D_{zw}\Delta(Y).
    \label{eq:LYdef}
\end{equation}
Using \eqref{eq:CDmats} and \eqref{eq:Delta},
\begin{equation}
    L(Y) =
    \begin{bmatrix}
        I_3 + YM_0^{-1}M_1 & YM_0^{-1}M_2 \\
        -YI_3 & I_3
    \end{bmatrix}.
    \label{eq:LYexplicit}
\end{equation}

% -----------------------------------------------------------------------
\section{Determinant and Invertibility of \boldmath{$L(Y)$}}
% -----------------------------------------------------------------------

Well-posedness of the LPV-LFR requires $L(Y)$ to be nonsingular for all
$Y$ in the operating range. Since the lower-right block $I_3$ is
invertible, the block determinant formula gives
\begin{equation}
    \det(L(Y))
    = \det(I_3)\cdot
      \det\!\Bigl(
          \underbrace{I_3 + YM_0^{-1}M_1
                      - \bigl(YM_0^{-1}M_2\bigr)(-YI_3)^{\phantom{1}}}_{=:\;S(Y)}
      \Bigr),
    \label{eq:blockdet}
\end{equation}
where $S(Y)$ is the Schur complement of $I_3$ in $L(Y)$:
\begin{equation}
    S(Y) := I_3 + YM_0^{-1}M_1 + Y^2M_0^{-1}M_2.
    \label{eq:SY}
\end{equation}
Using \eqref{eq:MYdecomp},
\begin{equation}
    S(Y)
    = M_0^{-1}(M_0 + YM_1 + Y^2M_2)
    = M_0^{-1}M(Y).
    \label{eq:SchurM}
\end{equation}
Hence
\begin{equation}
    \det(L(Y))
    = \det\!\bigl(M_0^{-1}M(Y)\bigr)
    = \underbrace{\det(M_0^{-1})}_{\neq\,0}\,\det(M(Y)).
    \label{eq:detidentity}
\end{equation}
Since $\det(M_0^{-1})$ is a nonzero constant (Assumption~A), the
zero-set of $\det(L(Y))$ is determined entirely by $\det(M(Y))$:
\begin{equation}
    L(Y)\ \text{invertible}
    \iff
    M(Y)\ \text{invertible}.
    \label{eq:invertibilityequiv}
\end{equation}

% -----------------------------------------------------------------------
\section{Polynomial Denominator}
% -----------------------------------------------------------------------

Define the shorthand constants
\begin{equation}
    \alpha := m_1+m_2+m_b+m_h,
    \qquad
    \beta  := \tfrac{(m_1-m_2)L_b}{2},
    \qquad
    \gamma := J_b+J_h+\tfrac{(m_1+m_2)L_b^2}{4}+m_hd^2.
    \label{eq:alphabetagamma}
\end{equation}
Then \eqref{eq:MY} reads
\begin{equation}
    M(Y) =
    \begin{bmatrix}
        \alpha          & \beta - m_hY & 0    \\
        \beta - m_hY    & \gamma + m_hY^2 & -m_hd \\
        0               & -m_hd           & m_h
    \end{bmatrix}.
    \label{eq:MYcompact}
\end{equation}
Expanding by cofactors along the third row,
\begin{equation}
    \det(M(Y))
    = m_h\!\left(
        \alpha\gamma - \beta^2
        + 2\beta m_h Y
        + m_h(\alpha - m_h)Y^2
      \right).
    \label{eq:detMYpoly}
\end{equation}
By \eqref{eq:detidentity}, $\det(L(Y))$ is therefore a quadratic polynomial
in $Y$, up to the nonzero constant $\det(M_0^{-1})$.

% -----------------------------------------------------------------------
\section{Symbolic Inverse of the Loop Matrix}
% -----------------------------------------------------------------------

Because $S(Y) = M_0^{-1}M(Y)$ is invertible whenever $L(Y)$ is
(see \eqref{eq:invertibilityequiv}), the block inverse formula applied to
\eqref{eq:LYexplicit} with the lower-right block $I_3$ gives
\begin{equation}
    L(Y)^{-1} =
    \begin{bmatrix}
        S(Y)^{-1} & -S(Y)^{-1}\bigl(YM_0^{-1}M_2\bigr) \\[2pt]
        YS(Y)^{-1} &
        I_3 - Y^2 S(Y)^{-1}M_0^{-1}M_2
    \end{bmatrix}.
    \label{eq:blockinverse}
\end{equation}
Substituting $S(Y)^{-1} = M(Y)^{-1}M_0$ from \eqref{eq:SchurM}
and cancelling $M_0M_0^{-1}=I_3$,
\begin{equation}
    L(Y)^{-1} =
    \begin{bmatrix}
        M(Y)^{-1}M_0    & -Y\,M(Y)^{-1}M_2 \\
        Y\,M(Y)^{-1}M_0 &  I_3 - Y^2M(Y)^{-1}M_2
    \end{bmatrix}.
    \label{eq:LYinverse}
\end{equation}

% -----------------------------------------------------------------------
\section{Rational Rewrite via Adjugate}
% -----------------------------------------------------------------------

\begin{equation}
    M(Y)^{-1} = \frac{\adj(M(Y))}{\det(M(Y))}.
    \label{eq:adjdet}
\end{equation}
Each $2\times2$ minor of $M(Y)$ is a determinant of entries of degree at
most $2$ in $Y$. Inspection of \eqref{eq:MYcompact} shows that only the
$(2,2)$-entry has degree $2$; the remaining entries have degree at most
$1$. Hence every $2\times2$ minor has degree at most $2$, and
\begin{equation}
    \adj(M(Y)) = N_0 + YN_1 + Y^2N_2,
    \label{eq:adjexpand}
\end{equation}
with
\begin{equation}
    N_0 =
    \begin{bmatrix}
        m_h(\gamma - m_h d^2) & -\beta m_h          & -\beta d\,m_h \\
        -\beta m_h            & \alpha m_h           & \alpha d\,m_h \\
        -\beta d\,m_h         & \alpha d\,m_h        & \alpha\gamma - \beta^2
    \end{bmatrix},
    \label{eq:N0}
\end{equation}
\begin{equation}
    N_1 =
    \begin{bmatrix}
        0        & m_h^2       & d\,m_h^2 \\
        m_h^2    & 0           & 0        \\
        d\,m_h^2 & 0           & 2\beta m_h
    \end{bmatrix},
    \label{eq:N1}
\end{equation}
\begin{equation}
    N_2 =
    \begin{bmatrix}
        m_h^2 & 0 & 0 \\
        0     & 0 & 0 \\
        0     & 0 & \alpha m_h - m_h^2
    \end{bmatrix}.
    \label{eq:N2}
\end{equation}
Note that $N_2$ has a zero $(2,2)$-entry: the numerator of the
$(2,2)$-entry of $M(Y)^{-1}$ has degree at most $1$ in $Y$.
Hence
\begin{equation}
    M(Y)^{-1}
    = \frac{N_0 + YN_1 + Y^2N_2}
           {m_h\!\left(\alpha\gamma-\beta^2
                       + 2\beta m_h Y
                       + m_h(\alpha-m_h)Y^2\right)}.
    \label{eq:MYinversepoly}
\end{equation}
Substituting \eqref{eq:MYinversepoly} into \eqref{eq:LYinverse} yields
a fully symbolic rational expression for $L(Y)^{-1}$.

% -----------------------------------------------------------------------
\section{Symbolic LPV-LFR Loop Solution}
% -----------------------------------------------------------------------

The right-hand side of the algebraic loop equation
\eqref{eq:loopAlg} evaluates to
\begin{equation}
    C_z x + D_{zu} u
    =
    \begin{bmatrix}
        M_0^{-1}f_{\mathrm{net}} \\
        0
    \end{bmatrix},
    \label{eq:rhsloop}
\end{equation}
using \eqref{eq:CDmats} and \eqref{eq:fnet}.
The unique solution for $z$ (guaranteed by \eqref{eq:invertibilityequiv}
whenever $M(Y)$ is invertible) is
\begin{equation}
    z
    = L(Y)^{-1}
      \begin{bmatrix}
          M_0^{-1}f_{\mathrm{net}} \\ 0
      \end{bmatrix}
    = \begin{bmatrix}
          M(Y)^{-1}f_{\mathrm{net}} \\
          Y\,M(Y)^{-1}f_{\mathrm{net}}
      \end{bmatrix},
    \label{eq:zsymbolic}
\end{equation}
and $w = \Delta(Y)z = YI_6\,z$ gives
\begin{equation}
    w =
    \begin{bmatrix}
        Y\,M(Y)^{-1}f_{\mathrm{net}} \\
        Y^2M(Y)^{-1}f_{\mathrm{net}}
    \end{bmatrix}.
    \label{eq:wsymbolic}
\end{equation}
Both $z$ and $w$ are rational functions of $Y$ through
\eqref{eq:MYinversepoly}. Equations \eqref{eq:zsymbolic}--\eqref{eq:wsymbolic}
are the symbolic solution of the LPV-LFR algebraic loop; they provide a
closed-form evaluator for the latent signals at each time instant, and
do not redefine the model as a collapsed LPV state-space system.

% -----------------------------------------------------------------------
\section*{Final Result}
% -----------------------------------------------------------------------

For the gantry LPV-LFR realization, the loop matrix
$L(Y) = I - D_{zw}\Delta(Y)$ satisfies:
\begin{align*}
    \det(L(Y))
    &= \det(M_0^{-1})\,\det(M(Y)),
    \quad
    \det(M(Y)) = m_h\!\left(
        \alpha\gamma-\beta^2+2\beta m_h Y+m_h(\alpha-m_h)Y^2
    \right),\\[4pt]
    L(Y)^{-1}
    &=
    \begin{bmatrix}
        M(Y)^{-1}M_0    & -Y\,M(Y)^{-1}M_2 \\
        Y\,M(Y)^{-1}M_0 &  I_3-Y^2M(Y)^{-1}M_2
    \end{bmatrix},
\end{align*}
where $M(Y)^{-1}$ is the rational function of $Y$ given in
\eqref{eq:MYinversepoly}. The symbolic loop solution is
\[
    z =
    \begin{bmatrix}
        M(Y)^{-1}f_{\mathrm{net}}\\
        Y\,M(Y)^{-1}f_{\mathrm{net}}
    \end{bmatrix},
    \qquad
    w =
    \begin{bmatrix}
        Y\,M(Y)^{-1}f_{\mathrm{net}}\\
        Y^2M(Y)^{-1}f_{\mathrm{net}}
    \end{bmatrix}.
\]
Well-posedness ($L(Y)$ invertible for all $Y$ in the operating range) is
equivalent to invertibility of $M(Y)$, which holds for all physically
admissible parameters.

\end{document}