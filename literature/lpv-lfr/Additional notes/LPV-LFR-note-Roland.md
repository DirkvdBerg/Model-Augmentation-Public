\documentclass{article}

\usepackage{amsmath,amssymb,mathtools}
\usepackage{arydshln} % dashed lines in matrices
\usepackage{tikz}
\usetikzlibrary{calc}

\begin{document}

\[
\begin{bmatrix}
\dot{x}\\[1mm]
y
\end{bmatrix}
=
\underbrace{\left[
\left[
\begin{array}{c:c}
A   & B_u \\ \hdashline
C_y & D_{yu}
\end{array}
\right]
+
\begin{bmatrix}
B_w\\[1mm]
D_{yw}
\end{bmatrix}
\Delta(p)\,
\tikz[remember picture,baseline=(invterm.base)]{
    \node[inner sep=1pt,outer sep=0pt] (invterm)
    {$\left(I-D_{zw}\Delta(p)\right)^{-1}$};
}
\begin{bmatrix}
C_z & D_{zu}
\end{bmatrix}
\right]
}_{\left[
\begin{array}{c:c}
A   & B_u \\ \hdashline
C_y & D_{yu}
\end{array}
\right]} 
\begin{bmatrix}
x\\[1mm]
u
\end{bmatrix}
\]

\begin{tikzpicture}[overlay,remember picture]
    \node[align=center] (note) at ($(invterm.north)+(0,1.4cm)$)
    {$\displaystyle
    polynomial \approx \left(I-D_{zw}\Delta(p)\right)^{-1}
    =
    \frac{1}{\det\!\left(I-D_{zw}\Delta(p)\right)}
    \,\operatorname{adj}\!\left(I-D_{zw}\Delta(p)\right)
    $};
    \draw[->] (note.south) -- (invterm.north);
\end{tikzpicture}

\end{document}