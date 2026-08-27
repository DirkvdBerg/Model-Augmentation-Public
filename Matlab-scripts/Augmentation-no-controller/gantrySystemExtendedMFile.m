function dxdt = gantrySystemExtendedMFile(u, x, m1, m2, mb, mh, Jb, Jh, d, Lb, kb1, kb2, cg1, cg2, cb1, cb2, cy, ma, ka, ca, L0)
    dxdt = gantrySystemExtended(u, x, m1, m2, mb, mh, Lb, Jb, Jh, d, cg1, cg2, cb1, cb2, cy, kb1, kb2, ma, ka, ca, L0);
end
