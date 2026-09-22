function v_brk = coulomb_v_brk(cc1, cc2, m_total, ts)
%COULOMB_V_BRK  The stick band, read from the ONE place that defines it.
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
% So the value is PARSED out of the ODE source rather than restated here. That
% keeps exactly one definition in the tree and makes a future change propagate to
% every gate automatically, or fail loudly if the line is renamed.
%
% D-209 (2026-09-22): the band is no longer derived from cc and ts. It is a
% physical breakaway velocity, V_BRK = 2.25e-3 m/s, THEORY: lee2020feeddrive
% Table 5 p. 2839 (identified on a THK SSR20XW LM guide). So V_BRK = V_BRK and
% the cc1, cc2, m_total and ts arguments no longer affect the result. They are
% RETAINED in the signature so every existing caller keeps working unchanged;
% the gates pass them and simply get a value that no longer depends on them.
% If a caller relies on the band scaling with ts, that caller is pre-D-209 and is
% wrong.

    src = fullfile(fileparts(mfilename('fullpath')), 'gantrySystemExtendedCoulomb.m');
    txt = fileread(src);
    tok = regexp(txt, 'V_BRK\s*=\s*([0-9.eE+-]+)\s*;', 'tokens', 'once');
    assert(~isempty(tok), ...
        ['Could not find "V_BRK = <value>;" in %s. The stick band is defined ' ...
         'there and parsed here so the gates cannot drift from it; if that line ' ...
         'was renamed, update this function rather than restating the constant. ' ...
         '(Before D-209 the band was "V_EPS_MARGIN = <value>;" and was scaled by ' ...
         '(cc1+cc2)/m_total*ts.)'], src);
    v_brk = str2double(tok{1});
    assert(isfinite(v_brk) && v_brk > 0, 'V_BRK parsed as %g', v_brk);
end
