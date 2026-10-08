function dxdt  = gantrySystem(u,x,m1,m2,mb,mh,Lb,Jb,Jh,d,cg1,cg2,cb1,cb2,cy,kb1,kb2)
    Y = x(3); % This works because Y is defined the same in stage & logical coordinates. 

    % Mass Matrix
    M = [          m1 + m2 + mb + mh,                          (m1 - m2) * Lb / 2 - mh * Y,        0;
         (m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb^2 / 4 + mh * d^2 + mh * Y^2,  -mh * d;
                                   0,                                              -mh * d,       mh];
    
    % Viscous Damping Matrix
    C = [           cg1 + cg2,               (cg1 - cg2) * Lb / 2,  0;
         (cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb^2 / 4,  0;
                            0,                                  0, cy];
    
    % Stiffness Matrix
    K = [0,         0, 0;
         0, kb1 + kb2, 0;
         0,         0, 0];

    A = [zeros(3)   eye(3);
         -M\K    -M\C];
    B = [zeros(3); pinv(M)];

    C = [eye(3), zeros(3)];

    dxdt = A*x + B*u; 
end