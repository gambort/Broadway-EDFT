import numpy as np

def fmap(zeta):
    return 2. - 4/3*zeta**2 + 1/6*(1.0187*zeta**3 + 0.9813*zeta**4)

def fmapInv(fb):
    return np.sqrt(3/4*(2-fb)) * (1. + (np.sqrt(4/3)-1.)*(2-fb))

def zetamap(fb): return fmapInv(fb)

def ts(rs, zeta=0.):
    return 1.10495/rs**2 *\
        ( (1+zeta)**(5/3) + (1-zeta)**(5/3) )/2

def ts_cofe(rs, fb=2.):
    return 1.10495/rs**2 * (2/fb)**(2/3)

def F(rs, P):
    A, alpha, beta1, beta2, beta3, beta4 = P

    D = np.sqrt(rs)*(beta1 + beta3*rs) + rs*(beta2 + beta4*rs)
    return -2.*A*(1 + alpha*rs) * np.log(1. + 1./(2*A*D))

def dF(rs, P):
    A, alpha, beta1, beta2, beta3, beta4 = P

    D = np.sqrt(rs)*(beta1 + beta3*rs) + rs*(beta2 + beta4*rs)
    dD = 1/np.sqrt(rs)*(0.5*beta1 + 1.5*beta3*rs) \
        + (beta2 + 2*beta4*rs)
    
    return -2.*A*alpha * np.log(1. + 1./(2*A*D)) \
        +2.*A*(1 + alpha*rs) * dD/(D*(1+2*A*D))

def FGen(rs, P, p=0, c=0.):
    A, alpha, beta1, beta2, beta3, beta4 = P
    if p>0: N = 1. + rs**p
    else: N = 1.

    D = np.sqrt(rs)*(beta1 + beta3*rs**(p+1)) + rs*(beta2 + beta4*rs**(p+1+c))
    return -2.*A*(1 + alpha*rs) * np.log(1. + N/(2*A*D))

def LDA_cofe(rs, fb=2., fb_c=None, Sum=False):
    if fb_c is None: fb_c = fb
    x  = (2-fb_c)
    x2 = (2-fb_c)*(fb_c-1)
    x3 = (2-fb_c)*(fb_c-1)*(3/2-fb_c)
    
    epsx = -0.458164/rs * (2/fb)**(1/3)

    ec0   = F(rs, (0.031091, 0.1825, 7.5961, 3.5879, 1.2666, 0.4169))
    ec34  = F(rs, (0.028833, 0.2249, 8.1444, 3.8250, 1.6479, 0.5279))
    ec66  = F(rs, (0.023303, 0.2946, 9.8903, 4.5590, 2.5564, 0.7525))
    ec100 = F(rs, (0.015545, 0.1260, 14.1229, 6.2011, 1.6503, 0.3954))
    
    F2 = 2*(2.*ec66 - ec0 - ec100)
    F3 = 40/357*(102*ec66 - 200*ec34 + 119*ec0 - 21*ec100)
    F3 = 13.33*ec0 - 22.41*ec34 + 11.43*ec66 - 2.35*ec100

    epsc = ec0 + x*(ec100-ec0) + x2*F2 + x3*F3

    if Sum: return epsx+epsc
    else: return epsx, epsc

def LDA_cofe_deriv(rs, fb=2., fb_c=None, Sum=False):
    if fb_c is None: fb_c = fb
    
    x  = (2-fb_c)
    x2 = (2-fb_c)*(fb_c-1)
    x3 = (2-fb_c)*(fb_c-1)*(3/2-fb_c)
    
    epsx = -0.458164/rs * (2/fb)**(1/3)
    depsx_rs = -epsx/rs
    depsx_fb = -epsx/3/fb

    ec0   = F(rs, (0.031091, 0.1825, 7.5961, 3.5879, 1.2666, 0.4169))
    ec34  = F(rs, (0.028833, 0.2249, 8.1444, 3.8250, 1.6479, 0.5279))
    ec66  = F(rs, (0.023303, 0.2946, 9.8903, 4.5590, 2.5564, 0.7525))
    ec100 = F(rs, (0.015545, 0.1260, 14.1229, 6.2011, 1.6503, 0.3954))

    dec0   = dF(rs, (0.031091, 0.1825, 7.5961, 3.5879, 1.2666, 0.4169))
    dec34  = dF(rs, (0.028833, 0.2249, 8.1444, 3.8250, 1.6479, 0.5279))
    dec66  = dF(rs, (0.023303, 0.2946, 9.8903, 4.5590, 2.5564, 0.7525))
    dec100 = dF(rs, (0.015545, 0.1260, 14.1229, 6.2011, 1.6503, 0.3954))

    F2 = 2*(2.*ec66 - ec0 - ec100)
    F3 = 40/357*(102*ec66 - 200*ec34 + 119*ec0 - 21*ec100)
    F3 = 13.33*ec0 - 22.41*ec34 + 11.43*ec66 - 2.35*ec100

    dF2 = 2*(2.*dec66 - dec0 - dec100)
    dF3 = 40/357*(102*dec66 - 200*dec34 + 119*dec0 - 21*dec100)
    dF3 = 13.33*dec0 - 22.41*dec34 + 11.43*dec66 - 2.35*dec100

    epsc = ec0 + x*(ec100-ec0) + x2*F2 + x3*F3
    depsc_rs = dec0 + x*(dec100-dec0) + x2*dF2 + x3*dF3
    depsc_fb = -(ec100-ec0) + (3-2*fb_c)*F2 \
        + (3*fb_c**2-9*fb_c+13/2)*F3

    if Sum: return epsx + epsc, depsx_rs + depsc_rs, depsx_fb + depsc_fb
    else:
        return epsx, depsx_rs, depsx_fb, \
            epsc, depsc_rs, depsc_fb


def LDA_rPW92(rs, zeta=0., Gx=None, Sum=False):
    if not(Gx is None):
        # Expresses things using on-top exchange hole
        zeta = np.sqrt(np.mininum(-1-2.*Gx,0.))
        
    A = (1.-zeta**2)
    B = zeta**2
    epsx = -0.458164/rs * ((1+zeta)**(4/3)+(1-zeta)**(4/3))/2

    ec0   = F(rs, (0.031091, 0.1825, 7.5961, 3.5879, 1.2666, 0.4169))
    ec34  = F(rs, (0.030096, 0.1842, 7.9233, 3.7787, 1.3510, 0.4326))
    ec66  = F(rs, (0.026817, 0.1804, 9.0910, 4.4326, 1.5671, 0.4610))
    ec100 = F(rs, (0.015546, 0.1259, 14.1225, 6.2009, 1.6496, 0.3952))
    
    Z2 = -10.95*ec0 +  13.32*ec34 +  -1.47*ec66 +  -0.90*ec100
    Z3 =  19.86*ec0 + -30.57*ec34 +  12.71*ec66 +  -2.00*ec100

    epsc = A*ec0 + B*ec100 + A*B*(Z2 + B*Z3)

    if Sum: return epsx+epsc
    else: return epsx, epsc

def LDA_PW92(rs, zeta=0., Sum=False):
    epsx = -0.458164/rs * ((1+zeta)**(4/3)+(1-zeta)**(4/3))/2

    fzeta = ((1+zeta)**(4/3)+(1-zeta)**(4/3)-2.)/(2**(4/3)-2)

    epsc0 = F(rs, (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294))
    epsc1 = F(rs, (0.015545, 0.20548,14.1189, 6.1977, 3.3662, 0.62517))
    alphc =-F(rs, (0.016887, 0.11125,10.3570, 3.6231,0.88026, 0.49671))

    epsc = epsc0 + zeta**4*fzeta * (epsc1-epsc0) \
        + (1.-zeta**4)*fzeta/1.709921 * alphc

    if Sum: return epsx+epsc
    else: return epsx, epsc


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    
    eta = 1e-6
    
    rs = np.logspace(-4, 1.5, 201)
    drs = eta * rs
    dfb = eta

    for fb in (2., 1.75, 1.5, 1.25, 1.):
        epsxc = LDA_cofe(rs, fb, Sum=True)
        epsxc_p = LDA_cofe(rs + drs, fb, Sum=True)
        epsxc_q = LDA_cofe(rs, fb + dfb, Sum=True)

        depsxc_rs_num = (epsxc_p - epsxc)/drs
        depsxc_fb_num = (epsxc_q - epsxc)/dfb

        epsxc, depsxc_rs, depsxc_fb = LDA_cofe_deriv(rs, fb, Sum=True)

        print(fb)
        print(depsxc_rs / depsxc_rs_num)
        print(depsxc_fb / depsxc_fb_num)

        V0 = 4.*np.pi/3
        n = 1/(V0*rs**3)
        dn = eta*n
        n_p = n + dn
        drs = (1/V0/n_p)**(1/3) - rs

        epsxc_p = LDA_cofe(rs + drs, fb, Sum=True)
        depsxc_n_num = (n_p*epsxc_p - n*epsxc)/dn

        depsxc_n = epsxc - rs/3*depsxc_rs
        print(depsxc_n / depsxc_n_num)
        
