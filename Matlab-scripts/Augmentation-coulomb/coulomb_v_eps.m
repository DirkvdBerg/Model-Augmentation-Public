function [v_eps, margin] = coulomb_v_eps(cc1, cc2, m_total, ts)
%COULOMB_V_EPS  The stick band, read from the ONE place that defines it.
%
% The band lives inside gantrySystemExtendedCoulomb.m, which must stay
% codegen-compatible for the Simulink chart and therefore cannot call a shared
% helper. Duplicating the constant in every gate is how a gate silently stops
% testing the thing it names: when V_EPS_MARGIN went from 1 to 3 (D-204
% amendment), the gates kept their own copy of the old formula, classified every
% rail between 1x and 3x as sliding while the ODE had it stuck, and gate 2
% reported a 3.67e+01 N slip violation that was entirely an artefact of the two
% definitions disagreeing.
%
% So the margin is PARSED out of the ODE source rather than restated here. That
% keeps exactly one definition in the tree and makes a future change to
% V_EPS_MARGIN propagate to every gate automatically, or fail loudly if the line
% is renamed.

    src = fullfile(fileparts(mfilename('fullpath')), 'gantrySystemExtendedCoulomb.m');
    txt = fileread(src);
    tok = regexp(txt, 'V_EPS_MARGIN\s*=\s*([0-9.eE+-]+)\s*;', 'tokens', 'once');
    assert(~isempty(tok), ...
        ['Could not find "V_EPS_MARGIN = <value>;" in %s. The stick band is ' ...
         'defined there and parsed here so the gates cannot drift from it; if ' ...
         'that line was renamed, update this function rather than restating ' ...
         'the constant.'], src);
    margin = str2double(tok{1});
    assert(isfinite(margin) && margin > 0, 'V_EPS_MARGIN parsed as %g', margin);

    v_eps = margin * (cc1 + cc2) / m_total * ts;
end
