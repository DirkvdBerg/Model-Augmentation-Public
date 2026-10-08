function C = ruleOfThumb(fbw, sys, ts)
    s = tf('s'); 
    int_factor = 1/6; 
    int = (s+2*pi*fbw*int_factor)/s; 
    leadlag_factor = [1/3, 3]; 
    leadlag = (s+2*pi*fbw*leadlag_factor(1))/(s+2*pi*fbw*leadlag_factor(2)); 
    lowpass_factor = 10; 
    lowpass = 2*pi*lowpass_factor*fbw/(s+2*pi*lowpass_factor*fbw); 
    
    Cnorm = int*leadlag*lowpass; 
    K = 1/abs(freqresp(sys*Cnorm, 2*pi*fbw)); 
    C = K*Cnorm; 

    C = c2d(C, ts, 'tustin'); 
end