#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on the ISGW2 implementation within EvtGen:
# Copyright (c) 1998-2020 CERN for the benefit of the EvtGen authors
# Copyright (c) 2026 Florian Herren
# Copyright (c) 2026 Raynette van Tonder
#
# This file is part of MCAmbulance.
# 
# MCAmbulance is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# MCAmbulance is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with MCAmbulance.
# If not, see <https://www.gnu.org/licenses/>. 

import numpy as np
from .misc import lam
from .semileptonic import BtoDstarstarlnu

def Getas(massq, massx = None):
    if massx == None:
        massx = massq
    lqcd2 = 0.04
    flav = 4
    als = 0.6
    if massx > 0.6:
        if massq < 1.85:
            nflav = 3
        als = 12. * np.pi / (33. - 2. * nflav) / np.log(massx ** 2 / lqcd2)
    return als

class BtoD0lnu_ISGW2(BtoDstarstarlnu):
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        fp, fm = self._ffs(q2, M2)
        
        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2, q2, self.m_1**2))
        sqrtq2 = np.sqrt(q2)

        f0 = (fp + ( fm / ( ( self.m_1**2 - M2 ) / q2 ) ))

        h0 = sqrtlam * fp / sqrtq2
        ht = (self.m_1 ** 2 - M2) * f0 / sqrtq2
        hp = 0.
        hm = 0.
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        mb = self.m_1
        mx = np.sqrt(M2)
        tm = ( mb - mx ) ** 2
        t = q2 * np.heaviside(tm - q2, 1.) + 0.99 * tm * np.heaviside(q2 - tm, 0.)

        r2 = 3.0 / ( 4.0 * self.msb * self.msq ) + 3 * self.msd ** 2 / ( 2 * self.mbb * self.mbx * self.bbx2 ) + ( 16.0 / ( self.mbb * self.mbx * ( 33.0 - 2.0 * self.nfp ) ) ) * np.log( Getas( self.mqm, self.lqcd2 ) / Getas( self.msq, self.lqcd2 ) )
        f5 = np.sqrt( self.mtx / self.mtb ) * ( np.sqrt( self.bx2 * self.bb2 ) / self.bbx2) ** 2.5 / ( ( 1.0 + r2 * ( tm - t ) / 18.0 ) ** 3.0 )
        f5uppum = f5 / np.sqrt( self.mbb / self.mtb ) * np.sqrt( self.mbx / self.mtx )
        f5upmum = f5 * np.sqrt( self.mbb / self.mtb ) / np.sqrt( self.mbx / self.mtx )

        uppum = -1.0 * f5uppum * np.sqrt( 2.0 / ( 3.0 * self.bb2 ) ) * self.msd
        upmum = 1.0 * f5upmum * np.sqrt( 2.0 / ( 3.0 * self.bb2 ) ) * self.msd * self.mtb / self.mtx
        return ( uppum + upmum ) / 2.0, ( uppum - upmum ) / 2.0



class BtoD2500lnu_ISGW2(BtoDstarstarlnu):
    # This is what EvtGen refers to as D(2S)0
    # It is a scalar state, so has the same current structure
    # as the D_0*0 but different FFs

    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        fp, fm = self._ffs(q2, M2)
        
        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2, q2, self.m_1**2))
        sqrtq2 = np.sqrt(q2)

        f0 = (fp + ( fm / ( ( self.m_1**2 - M2 ) / q2 ) ))

        h0 = sqrtlam * fp / sqrtq2
        ht = (self.m_1 ** 2 - M2) * f0 / sqrtq2
        hp = 0.
        hm = 0.
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        mb = self.m_1
        mx = np.sqrt(M2)
        msb=5.2
        msd=0.33
        bb2=0.431*0.431
        mbb=0.75*5.325+0.25*5.279
        msq=1.82
        bx2=0.45*0.45
        mbx=0.75*2.64+0.25*2.58
        nfp=3.0
        t = q2
        mtb = msb + msd
        mtx = msq + msd
        bbx2=0.5*(bb2+bx2)
        tm = ( mb - mx ) ** 2
        if t>tm:
            t = 0.99*tm
        wt=1.0+(tm-t)/(2.0*mbb*mbx)
        mqm = 0.1
        
        r2=3.0/(4.0*msb*msq)+3*msd*msd/(2*mbb*mbx*bbx2) + (16.0/(mbb*mbx*(33.0-2.0*nfp)))* np.log(Getas(mqm)/Getas(msq))
  
        f3 = np.sqrt(mtx/mtb)*(np.sqrt(bx2*bb2)/bbx2)**1.5 / ((1.0+r2*(tm-t)/24.0)**4)
  
        f3fppfm = f3/np.sqrt(( mbb / mtb ))*np.sqrt((mbx/mtx))
        f3fpmfm = f3*np.sqrt(( mbb / mtb ))/np.sqrt((mbx/mtx))
  
        tau = msd*msd*bx2*(wt-1)/(bb2*bbx2)
        udef = (( bb2-bx2)/(2.0*bbx2)) + ((bb2*tau)/(3.0*bbx2))
        vdef = (bb2*(1.0+(msq/msb))/(6.0*bbx2))*(7.0 - ((bb2/bbx2)*(5+tau)))

        fppfm = f3fppfm*np.sqrt(1.5)*((1.0-(msd/msq))*udef-(msd*vdef/msq));
        fpmfm = f3fpmfm*np.sqrt(1.5)*(mtb/msq)*(udef+(msd*vdef/mtx));

        fppf = (fppfm + fpmfm) /2.0;
        fpmf = (fppfm - fpmfm) /2.0;

        return fppf, fpmf


class BtoD1plnu_ISGW2(BtoDstarstarlnu):
    # This is what EvtGen refers to as D'_10
    # It is a vector type state
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        self.mum = 1. / (1. / self.msq - 1. / self.msb)
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        gf, ff, apf, amf = self._ffs(q2, M2)

        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2,q2,self.m_1**2))
        sqrtq2 = np.sqrt(q2)
        
        V = gf * (self.m_1 + M)
        A1 = ff / (self.m_1 + M)
        A2 = -apf * (self.m_1 + M)
        a3f = ff / 2. / M + apf * (self.m_1 ** 2 - M2) / 2. / M
        A0 = a3f + q2 * amf / 2. / M

        h0 = (A1 * (self.m_1 + M) ** 2 * (self.m_1 ** 2 - M2 - q2) - A2 * sqrtlam ** 2) / (2. * M * (self.m_1 + M) * sqrtq2)
        ht = A0 * sqrtlam / sqrtq2
        hp = (A1 * (self.m_1 + M) ** 2 - sqrtlam * V) / (self.m_1 + M)
        hm = (A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        mb = self.m_1
        mx = np.sqrt(M2)
        tm = ( mb - mx ) ** 2
        t = q2 * np.heaviside(tm - q2, 1.) + 0.99 * tm * np.heaviside(q2 - tm, 0.)
        wt = 1. + (tm - t) / 2. / self.mbb / self.mbx

        r2 = 3.0 / ( 4.0 * self.msb * self.msq ) + 3. * self.msd ** 2 / ( 2. * self.mbb * self.mbx * self.bbx2 ) + ( 16.0 / ( self.mbb * self.mbx * ( 33.0 - 2.0 * self.nfp ) ) ) * np.log( Getas( self.mqm ) / Getas( self.msq ) )
        f5 = np.sqrt( self.mtx / self.mtb ) * ( np.sqrt( self.bx2 * self.bb2 ) / self.bbx2) ** 2.5 / ( ( 1.0 + r2 * ( tm - t ) / 18.0 ) ** 3.0 )
        f5q = f5 / np.sqrt(self.mbb / self.mtb) / np.sqrt(self.mbx / self.mtx)
        f5l = f5 * np.sqrt(self.mbb / self.mtb) * np.sqrt(self.mbx / self.mtx)
        f5cppcm = f5 / (np.sqrt(self.mbb / self.mtb) ** 3) * np.sqrt(self.mbx / self.mtx)
        f5cpmcm = f5q

        ql = f5q * np.sqrt(1. / 6.) * self.msd / (np.sqrt(self.bb2) * self.mtx) * (1.0 - self.bb2 * self.mtb / 4. / self.msd / self.msq / self.msb)
        ll = f5l * np.sqrt(2. / 3.) * self.mtb * np.sqrt(self.bb2) * (0.5 / self.msq - 1.5 / self.msb + self.msd * self.mtx * (wt - 1.) / self.bb2 * (1.0 / self.msq - self.msd * self.bb2 / 2. / self.mum / self.mtx / self.bbx2))
        cppcm = f5cppcm * self.msd ** 2 * self.bx2 / (np.sqrt(6. * self.bb2) * self.mtb * self.msq * self.bbx2)
        cpmcm = - np.sqrt(2. / 3.) * self.msd * f5cpmcm / self.mtx / np.sqrt(self.bb2) * (1. + self.msd * self.bx2 / 2. / self.msq / self.bbx2)

        # smearing present in evtgen
        parMass = self.m_1
        q2max = parMass*parMass + M2 - 2.*parMass*np.sqrt(M2)
        massNom = self.m_nom
        q2maxNom = parMass*parMass + massNom*massNom - 2.*parMass*massNom
        q2maxin = np.sqrt(q2maxNom / q2max)
        if q2maxin > 1000:
            q2maxin = 1000.
        ql *= q2maxin
        ll *= q2maxin
        cppcm *= q2maxin
        cpmcm *= q2maxin

        return ql, ll, 0.5 * (cppcm + cpmcm), 0.5 * (cppcm - cpmcm)



class BtoD1stlnu_ISGW2(BtoDstarstarlnu):
    # This is what EvtGen refers to as a D_10
    # It is another vector type state so has the same
    # current structure as the D'_10 but different FFs
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        self.mum = 1. / (1. / self.msq - 1. / self.msb)
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        gf, ff, apf, amf = self._ffs(q2, M2)

        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2,q2,self.m_1**2))
        sqrtq2 = np.sqrt(q2)
        
        V = gf * (self.m_1 + M)
        A1 = ff / (self.m_1 + M)
        A2 = -apf * (self.m_1 + M)
        a3f = ff / 2. / M + apf * (self.m_1 ** 2 - M2) / 2. / M
        A0 = a3f + q2 * amf / 2. / M

        h0 = (A1 * (self.m_1 + M) ** 2 * (self.m_1 ** 2 - M2 - q2) - A2 * sqrtlam ** 2) / (2. * M * (self.m_1 + M) * sqrtq2)
        ht = A0 * sqrtlam / sqrtq2
        hp = (A1 * (self.m_1 + M) ** 2 - sqrtlam * V) / (self.m_1 + M)
        hm = (A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
                          
        return h0, ht, hp, hm


    def _ffs(self, q2, M2):
        """
        This is for the D_10, which represents the D(2420)
        """
        mb = self.m_1
        mx = np.sqrt(M2)

        msb=5.2
        msd=0.33
        bb2=0.431*0.431
        mbb=5.31
        msq=1.82
        bx2=0.33*0.33
        mbx=(5.0*2.46+3.0*2.42)/8.0
        nfp = 3.0
        mtb = msb + msd
        mtx = msq + msd
        mqm = 0.1

        mup=1.0/(1.0/msq+1.0/msb)
        mum=1.0/(1.0/msq-1.0/msb)
        bbx2=0.5*(bb2+bx2)
        tm=(mb-mx)*(mb-mx)
        t = q2
        if t>tm:
            t = 0.99*tm
        wt=1.0+(tm-t)/(2.0*mbb*mbx)

        r2=3.0/(4.0*msb*msq)+3*msd*msd/(2*mbb*mbx*bbx2)+(16.0/(mbb*mbx*(33.0-2.0*nfp)))* np.log(Getas(mqm,mqm)/Getas(msq,msq))
  
        f5 = np.sqrt(mtx/mtb) * (np.sqrt(bx2*bb2)/bbx2)**2.5 / ((1.0+r2*(tm-t)/18.0)**3)
        f5v = f5 / (np.sqrt(mbb / mtb)) / np.sqrt(mbx / mtx)
        f5r = f5 * np.sqrt(( mbb / mtb )) * np.sqrt((mbx/mtx));
        f5sppsm = f5/(( mbb / mtb )**1.5) * np.sqrt((mbx/mtx));
        f5spmsm = f5v

        vv = -1.0*msd*f5v/(2.0*np.sqrt(3.0*bb2)*mtx)*((wt+1)/2.0+bb2*mtb/(2.0*msd*msq*msb))
        rr = -2.0*mtb*np.sqrt(bb2/3.0)*f5r*(1.0/msq + mtx*msd*(wt-1)/(2.0*bb2)*((wt+1)/(2.0*msq)-msd*bb2/(2.0*mum*mtx*bbx2)))
        sppsm = -1.0*np.sqrt(3.0)*msd*f5sppsm/(2.0*np.sqrt(bb2)*mtb)*(1 - msd/(3.0*msq) -  msd*bb2/(3.0*bbx2)*(1.0/(2.0*mum)-1.0/mup))
        spmsm = -1.0*msd*f5spmsm/(2.0*np.sqrt(3.0*bb2)*mtx)*((2-wt)*mtx/msq +  msd*bb2/bbx2*(1.0/(2.0*mum)-1.0/mup))

        return vv, rr,  0.5 * (sppsm + spmsm), 0.5 * (sppsm - spmsm)



class BtoD2640lnu_ISGW2(BtoDstarstarlnu):
    # This is what EvtGen refers to as a D*2S0
    # This is another vector state, so has the same current
    # structure as D'_10 but with different FFs.
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        self.mum = 1. / (1. / self.msq - 1. / self.msb)
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        gf, ff, apf, amf = self._ffs(q2, M2)

        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2,q2,self.m_1**2))
        sqrtq2 = np.sqrt(q2)
        
        V = gf * (self.m_1 + M)
        A1 = ff / (self.m_1 + M)
        A2 = -apf * (self.m_1 + M)
        a3f = ff / 2. / M + apf * (self.m_1 ** 2 - M2) / 2. / M
        A0 = a3f + q2 * amf / 2. / M

        h0 = (A1 * (self.m_1 + M) ** 2 * (self.m_1 ** 2 - M2 - q2) - A2 * sqrtlam ** 2) / (2. * M * (self.m_1 + M) * sqrtq2)
        ht = A0 * sqrtlam / sqrtq2
        hp = (A1 * (self.m_1 + M) ** 2 - sqrtlam * V) / (self.m_1 + M)
        hm = (A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
                          
        return h0, ht, hp, hm


    def _ffs(self, q2, M2):
        """
        This is for the D*(2S)0, which represents the D(2640)
        """
        mb = self.m_1
        mx = np.sqrt(M2)
        tm = ( mb - mx ) ** 2
        msb=5.2
        msd=0.33
        bb2=0.431*0.431
        mbb=0.75*5.325+0.25*5.279
        cfp=0.929
        msq=1.82
        bx2=0.38*0.38
        mbx=0.75*2.64+0.25*2.58
        nfp=3.0
        mtb = msb + msd
        mtx = msq + msd
        mum=1.0/(1.0/msq-1.0/msb)
        bbx2=0.5*(bb2+bx2)
        mqm = 0.1

        tm=(mb-mx)*(mb-mx)
        t = q2 * np.heaviside(tm - q2, 1.) + 0.99 * tm * np.heaviside(q2 - tm, 0.)
        wt = 1. + (tm - t) / (2. * mbb * mbx)

        r2=3.0/(4.0*msb*msq)+3*msd*msd/(2*mbb*mbx*bbx2)+(16.0/(mbb*mbx*(33.0-2.0*nfp)))* np.log(Getas(mqm)/Getas(msq))
        f3 = np.sqrt(mtx/mtb)*(np.sqrt(bx2*bb2)/bbx2)**1.5 / (((1.0+r2*(tm-t)/24.0))**4)

        f3fp = f3*np.sqrt( mbb / mtb )*np.sqrt(mbx/mtx)
        f3gp = f3/np.sqrt( mbb / mtb )/np.sqrt(mbx/mtx)
        f3appam = f3*(( mbb / mtb )**-1.5)*np.sqrt(mbx/mtx)
        f3apmam = f3gp

        tau = msd*msd*bx2*(wt-1.0)/(bb2*bbx2);
        udef = (( bb2-bx2)/(2.0*bbx2));
        udef = udef + ((bb2*tau)/(3.0*bbx2));

        fpf = cfp*np.sqrt(1.5)*mtb*(1.0+wt)*udef*f3fp;

        gpf = np.sqrt(3.0/8.0)*f3gp*(((1.0/msq)-((msd*bb2)/(2.0*mum*mtx*bbx2)))* udef + ( (msd*bb2*bx2)/(3.0*mum*mtx*bbx2*bbx2)))
       
        appam = f3appam*np.sqrt(2.0/3.0)*(bb2/(msq*msb*bbx2))*((-7.0*msd*msd*bx2*bx2*(1.0+(tau/7.0))/(8.0*mtb*bbx2*bbx2))+(5.0*msd*bx2*(1.0+(tau/5.0))/(4.0*bbx2))+(3.0*msd*msd*bx2*bx2/(8.0*mtb*bb2*bbx2))-(3.0*msd*bx2/(4.0*bb2)))
         
        apmam = f3apmam*np.sqrt(3.0/2.0)*(mtb/(msb*mtx))*(1.0-(bb2*(1.0+(tau/7.0))/bbx2)-(msd*bx2*(1.0-(5.0*bb2*(1.0+(tau/5.0))/(3.0*bbx2)))/(2.0*mtb*bbx2))-(7.0*msd*msd*bb2*bx2/(12.0*msq*mtb*bbx2*bbx2))*(1.0-(bx2/bbx2)+(bb2*tau/(7.0*bbx2))));

        appf = (appam + apmam) /2.0
        apmf = (appam - apmam) /2.0
        return fpf, gpf, appf, apmf

