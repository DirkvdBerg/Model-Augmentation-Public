\subsection*{Aspect 1: Position-Dependent Baseline Modeling}
The physical baseline model that follows from Garc\'ia-Herreros et al. \cite{garcia2013model}, will need to be converted to discrete-time state-space for the augmentation step. The physical parameters are identified through structured experiments \cite{garcia2013model}. Since payload $Y$ enters the model without derivatives, it is the natural choice for the scheduling variable for the LPV reformulation. We need to check the resulting coefficient matrices for static scheduling dependence \cite{toth2010modeling}. Since $Y$ is a system state rather than an exogenous signal, the resulting formulation is quasi-LPV \cite{toth2010modeling}. An appropriate LPV discretization technique \cite{toth2010discretization} is applied to convert the continuous-time model to a discrete-time LPV state-space representation.
Furthermore, we need to verify the invertibility of the position-dependent inertia matrix for the full operational range. The model is first validated in simulation before evaluation on the real machine. Frozen LTI models at fixed values of $Y$ are used as a baseline comparison, since fixing the scheduling variable reduces the LPV model to a standard LTI representation.

\subsection*{Aspect 2: Augmentation Structure}
The chosen augmentation architecture is based on the LFR framework of \cite{hoekstra2026lfr}, which provides a unifying structure capable of representing a wide variety of augmentation architectures, including parallel and series interconnections with or without additional states beyond the baseline model.
A parallel structure is chosen, because the orthogonal projection-based regularization of \cite{gyorok2025l4dc} used in Aspect~3 is formulated specifically for parallel structures and is not readily applicable to a series interconnection.
Within the parallel class, a dynamic rather than static structure is chosen. Static augmentation does not introduce additional degrees of freedom beyond those of the baseline model, and therefore cannot capture behaviour present in the system but not modeled by the physics-based baseline \cite{hoekstra2026lfr}. Cross-coupling between axes and position-dependent flexible dynamics are examples of such behaviour, and additional augmentation states are therefore included.
The dynamic parallel LFR augmentation structure of \cite{hoekstra2026lfr} is extended to the LPV setting following \cite{drenth2025lpvlfr}, restricting attention to well-posed realizations, with the specific parameterization to be established during the project.

\subsection*{Aspect 3: Interpretability Preservation}
Orthogonal projection-based regularization \cite{kon2022physics, gyorok2025l4dc} is used, which penalizes the learning component for capturing dynamics already covered by the baseline, keeping the physical parameters meaningful \cite{gyorok2025l4dc}.
This method requires the baseline to be linear-in-parameters, meaning
the physical parameters appear linearly in the model equations
\cite{gyorok2025l4dc}. The Garc\'ia-Herreros baseline
\cite{garcia2013model} satisfies this in continuous time. We will need to verify
whether this also holds after ZOH discretization and LFR realization. If not, \cite{gyorok2025l4dc} provides a fallback using a Taylor
linearization around nominal parameter values $\theta_0$.
In the current context, using \cite{gyorok2025l4dc} requires three extensions: it is validated on a system without 
structural cross-coupling \cite{gyorok2025l4dc}, while the 
dual-gantry has two coupled outputs $(X, \Theta)$ 
\cite{garcia2013model}; the projection matrix is built from a fixed 
dataset with no scheduling dependence \cite{gyorok2025l4dc}, while 
here the baseline output subspace varies with $Y$ 
\cite{garcia2013model}; and the orthogonality regularization is derived for a 
parallel state-space structure \cite{gyorok2025l4dc}, not an LFR 
interconnection \cite{hoekstra2026lfr, drenth2025lpvlfr}.

\subsection*{Aspect 4: Model Generalization}
Two complementary evaluations are performed on held-out data: generalization to payload positions excluded from estimation, following the operating-point validation strategy of \cite{paijmans2008identification} and \cite{bachnas2014review}; and generalization to motion profiles not used during estimation, motivated by the requirements of the application. Best Fit Rate (BFR) is reported as the primary fit measure, consistent with \cite{drenth2025lpvlfr}.
A black-box state-space model is trained on the same data as a comparison baseline, with prediction accuracy reported alongside the augmented model; the specific model class is to be determined.

\subsection*{Data and Resources}
All aspects require access to the ASMPT dual-gantry experimental setup. 
Needed data include: structured identification experiments for baseline parameter identification; coupling-exciting settling trajectories for augmentation training; and held-out data at unseen payload positions and unseen motion profiles for validation. 