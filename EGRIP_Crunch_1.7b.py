##### EGRIP_Crunch_1.7.py

# based off of DYE3_Crunch_1.0 which was based off of EGRIP_Crunch_1.5.py 
# this is the functional code on the SIL laptop to take to the field
# Breaking up code into function that can be run with different isotopes and different inputs depending on methods and picarros to include 2140 d17o data
# Adding in loadFlag to know where depth is funcky from stacking cores versus using the carousel
# Adding in MeltRate from SpeedCam to help with Depth 6/4/17
# should remove speedcam and installing new depth algorithm to be consistant with all of the ice core projects 9/24/18
# added ec_value and melt_rate back in, changed directories to be local becuase of iffy internet at AWI VM 10/6/2023
# had to shorten nea memory corrections to 175 (lines 3702, 3708, 2737, 2752) to get the majority of files to go through, might be that melt rate starts closer to 3 for am nea

import matplotlib as mpl
mpl.use('Qt4Agg')

##### Initial variables ########################################################

crunchversion = "EGRIP_Crunch_1.7b.py"
#1.7a is interpolating all bag depths
#1.7b is interpolating only bag depths(but full bags) of identified problem areas


# PLEASE SELECT Core Name for data directory - please comment out the other cores
#corename = 'DYE3'
corename = 'EGRIP'

# VERBOSE FLAG, 1=ON, 0=OFF for batch mode after corrections made and flags set
verbose = 1
if verbose == 0:
    print "Running in Batch mode"

# d17O flag, set for 0 initially and set after data read in for 1 when 2140 data detected, after 2106
d17oflag = 0

##### TABLE OF CONTENTS ########################################################
# SECTION..........................................LINE#
# ------------------------------------------------------
# Initial variables.............................14
# REFERENCES..........................................68
# IMPORT PYTHON FUNCTIONS............................126 
# CONSTANTS..........................................156
# DEFINE FUNCTIONS...................................211
#       ALLAN CLASS..................................213
#       ALLAN FUNCTION...............................246
#       VALCOID FUNCTION.............................315
#       VALCOFUNCT FUNCTION..........................410
#       NEAFUNCT FUNCTION............................682
#       TRANSFER FUNCTION............................736
#       SMOOTHING FUNCTION..........................1134
#       PICK POINTS.................................1179
#       MEMORY FUNCTION.............................1183
# BEGIN OF FULL PROGRAM.............................1193
# READ IN DATA......................................1196 instrument flag
#       FILTER FOR LOW WATER CONCENTRATION..........1495 water flag
#       PLOT RAW DATA...............................1512
# ALLAN VARIANCE....................................1570
# CHECK LENGTHS.................................... 1764
# CHECK FOR MANUAL CORRECTIONS......................1771
# DEPTH CORRECTION FOR STACKING CORES...............1967
# FIRST DERIVATIVES.................................2023
# VALCO IDENTIFICATION FOR MEMORY AND CALIBRATION...2083 method flag 
# APPLY MEMORY TO ALL VALCO TRANSITIONS.............2286 memory flag
# NEAPOLITAN IDENTIFICATION FOR MEMORY..............2434 memory flag
# ISOTOPE CALIBRATION...............................2867
# ICE CORES.........................................3450
#       NEA MEMORY CORRECTION.......................3572 memory flag
#       DEPTH FILTER................................3700 depth flag
#       PARSE AND SAVE ALL ICE CORE DATA............3857 1st position prune flag
# OUTPUTS...........................................3982
# PLOT FOR QA/QC AND FLAGGING.......................4051 all second position flags
# READ IN ALL DATA AND PLOT FOR QA/QC...............4611 1st and 3rd position analytical flags
# END...............................................4740


##### REFERENCES USED THROUGHOUT THE CODE ######################################
# COMMENT FIELD CODES
# renumbered on 10/27/10 to ensure that the permuations of averages do not equal another comment value
# 0  = nothing
# 28 = other, look at written notes
# 101 = high isotope ice of neapolitan
# 103 = mid isotope ice of neapolitan (not being used 8/30/11)
# 104 = low isotope ice of neapolitan
# 106 (spare)
#       momentary flags:
#       113 = filter changed
#       114 = P2 stopped
#       117 = P2 restarted
#       119 = add constriction
#       126 = remove constriction
#       129 = gilson stuck
# 142 = beginning standard vials - Valco
# 153 = beginning of first neapolitan - AM
# 159 = end of first neapolitan - AM
# 161 = beginning of second neapolitan - PM
# 162 = end of second neapolitan - PM
# 172 = end of day standard vials - Valco
# 173 = push ice
# 175 = ice core stick
#
# VALCO STREAM SELECT VALVE POSITION MAP WAIS06A
# 1 = Waste
# 2 = Sample line from melter 
# 3 = kbw Std from vial
# 4 = kaw Std from vial
# 5 = kgw Std from vial
# 6 = kpw Std from vial
#
# VALCO STREAM SELECT VALVE POSITION MAP SPIce 2015
# 1 = Waste
# 2 = Sample line from melter 
# 3 = kaw Std from vial
# 4 = kgw Std from vial
# 5 = kpw Std from vial
# 6 = vw1f Std from vial
#
# VALCO STREAM SELECT VALVE POSITION MAP SPIce 2016 onwares
# 1 = Waste
# 2 = Sample line from melter 
# 3 = kaw Std from vial
# 4 = UW-WW Std from vial
# 5 = kpw Std from vial
# 6 = vw1f Std from vial
#
# Flagging scheme
# Flag1 - Reject flags: A, T, W, V
# Flag2 - Prune flags: P1, P2...P9, P0
# Flag3 - Manual flags: p, s, n, o, v, f
# Flag4 - Informational flags: a1, a2, a3, t1, t2, t3, p1, p2, p3, v, n, d, 
# Flag5 - Method flag, A, B, C, D
# Flag6 - Instrument flag: 1102, 2130, 2140
#

##### IMPORT PYTHON FUNCTIONS ##################################################
import numpy as np
import scipy as sp
import time
import sys
import string
import copy
import datetime
import os
import pickle 
from copy import deepcopy
from numpy import fft
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import stats
from scipy import optimize
from matplotlib import pyplot as plt
from math import sqrt, copysign, pi
import numpy.random as random
from numpy import where, zeros, ones, float64, array
from numpy import inner, kron
from numpy import exp as np_exp
from numpy import arctan as np_arctan
from scipy.stats import norm
from scipy.special import gamma as sp_gamma
from Tkinter import *
import tkMessageBox
from pick_points import PointGetter
import argparse

##### CONSTANTS ################################################################
watertoolow = 10000 # in ppm

## Standard Values, calibrated April 2016 (old 2010 values)
kbwd18o = -14.15 #(-14.19)
kbwdD = -111.65 #(-111.81)
kbwD17o = 27 #preliminary (was 13 previously)
kbwd17o = 1000*(sp.exp(kbwD17o/1000000+0.528*sp.log(kbwd18o/1000+1))-1) #preliminary

kawd18o = -30.30 #(-30.35)
kawdD = -239.13 #(-239.3)
kawD17o = 17 #preliminary 
kawd17o = 1000*(sp.exp(kawD17o/1000000+0.528*sp.log(kawd18o/1000+1))-1)     #-16.0880 preliminary, -16.1145347209 when calc with D17o=17

uwwwd18o = -33.82 #calibrated at UW, our measured value -33.86
uwwwdD = -268.30 #calibrated at UW, our measured value -287.60
uwwwD17o = 27 #calibrated at UW
uwwwd17o = 1000*(sp.exp(uwwwD17o/1000000+0.528*sp.log(uwwwd18o/1000+1))-1)     #-17.9754 #calibrated at UW, -18.0019014278 when calc with D17o=27

kgwd18o = -38.02 #(-38.09) #trap standard for WAIS06A
kgwdD = -298.37 #(-298.7)
kgwD17o = 15 ##have not calibrated this yet
kgwd17o = 1000*(sp.exp(kgwD17o/1000000+0.528*sp.log(kgwd18o/1000+1))-1)##have not calibrated this yet

kpwd18o = -45.41 #(-45.43) #trap standard for SPIceCore
kpwdD = -355.18 #(-355.6) 
kpwD17o = 6 #preliminary
kpwd17o = 1000*(sp.exp(kpwD17o/1000000+0.528*sp.log(kpwd18o/1000+1))-1)     #-24.2281 #preliminary, -24.2393212445 when calc with D17o=6

vw1fd18o = -56.59 #(-56 not calibrated in 2010)
vw1fdD = -438.43 #(-438 not calibrated in 2010) was incorrectly typed in as -427.65 before 9/27/16
vw1fD17o = 7 #preliminary
vw1fd17o = 1000*(sp.exp(vw1fD17o/1000000+0.528*sp.log(vw1fd18o/1000+1))-1)     #-30.2901 #preliminary, -30.2900518685 when calc with D17o=7

# EGRIP and DYE3 KBW(3), KAW(4), KGW(5), KPW(6)
knownd18o = [kbwd18o, kawd18o, kpwd18o]       
knowndD   = [kbwdD, kawdD, kpwdD]         

## Transfer function shape constatants
valcoshape1d18o = 1.8
valcoshape2d18o = 0.3
valcoshape1dD = 3.1
valcoshape2dD = 0.35
neashape1d18o = 1.6
neashape2d18o = 0.3
neashape1dD = 3.1
neashape2dD = 0.4

VALID_FLAGS = ('P', 'S', 'N', 'O', 'V', 'F', )
# P - manual pruning
# S - strange spike in data
# N - noisy data
# O - outliers
# V - unusual valco switch
# F - filter change

##### DEFINE FUNCTIONS USED THROUGHOUT THE CODE ################################

##### ALLAN VARIANCE MATHMATICS ################################################
## written by Vasileios Gkinis
class Allan():
    
    def __init__(self):
        return

    def allan(self, t, y):
        t1 = time.time()
        dt = t[1] - t[0]
        N = np.size(y)
        m = np.arange(2,np.floor(N/2)+1)# Number of subgroups
        allans = np.zeros(np.size(m))   # Array where Allan variances are stored
        tau = np.zeros(np.size(m))
        allan_i = 0                     # Initiates index for allans
        for mi in m:
            k = np.floor(N/mi)          # Size of subgroup
            tau[allan_i] = k*dt
            if tau[allan_i] == tau[allan_i-1]:
                tau[allan_i] = 0
                continue
            means_mi = np.zeros(mi)     # Means of subgroups array
            for s in np.arange(mi):
                means_mi[s] = np.mean(y[s*k:(s+1)*k])
            diffs_squares = (means_mi[1:] - means_mi[:-1])**2
            allans[allan_i] = 0.5*np.mean(diffs_squares)
            allan_i = allan_i+1
        allans = np.trim_zeros(allans)[::-1]
        tau = np.trim_zeros(tau)[::-1]
        if verbose ==1:
            print "Allan variance plot processing time: %0.3e" %(time.time() - t1)
        return np.array((tau, allans))

##### PERFORM ALLAN VARIANCE FUNCTION DEFINED ##################################
## written by Vasileios Gkinis, editted by Valerie Morris to automate
def perform_allan(name, isotope, AllanIndex1, AllanIndex2, secs, flag, water):
    Allansecs = secs[AllanIndex1:AllanIndex2]
    Allanwater_ppm = water[AllanIndex1:AllanIndex2]
    Allanisotope = isotope[AllanIndex1:AllanIndex2]
    if name == "d18o":
        colorsymbol = "b-o"
        graphnumber = 1
    if name == "dD":
        colorsymbol = "r-o"
        graphnumber = 3
    if name == "2140d18o":
        colorsymbol = "b-o"
        graphnumber = 5
    if name == "2140dD":
        colorsymbol = "r-o"
        graphnumber = 7
    if name == "2140d17o":
        colorsymbol = "g-o"
        graphnumber = 9
    time_step = mean_time_delay ## was "np.ceil(mean_time_delay)"really with all of the data, this equals 1.0 except when it is negative, then it is 0.0... would this be more appropriate to leave as just the mean_time_delay
    equal_secs = np.arange(Allansecs[0], Allansecs[-1], time_step)
    equal_Allanisotope = np.interp(equal_secs, Allansecs, Allanisotope)
    mean_water_ppm = np.mean(Allanwater_ppm)
    tau, allans_Allanisotope = Allan().allan(equal_secs, equal_Allanisotope)

    ## PLOTS OF ALLAN ##################################################
    fig_allan = plt.figure(graphnumber)
    fig_allan_ax1 = fig_allan.add_subplot(111)
    fig_allan_ax1.loglog(tau, allans_Allanisotope, colorsymbol)
    fig_allan_ax1.set_ylabel("Allan variance")
    fig_allan_ax1.set_xlabel("Integration time [sec]")
    fig_allan_ax1.set_title(name + " time series")
    fig_allan_ax1.grid(True)

    fig_series1 = plt.figure(graphnumber+1)
    fig_series1_ax1 = fig_series1.add_subplot(111)
    fig_series1_ax1.plot(equal_secs, equal_Allanisotope, colorsymbol)
    fig_series1_ax1.set_title(name + " time series")
    fig_series1_ax1.set_ylabel(name)
    fig_series1_ax1.set_xlabel("Time [sec]")
            
    ## OUTPUT TO SCREEN OF ALLAN AND TO FUNCTION OUTPUT ########################            
    allanout = np.zeros (7)
    allanout[0] = np.mean(Allanisotope)
    if verbose ==1:
        print(name + " average value for section: %0.3f" %(allanout[0]))
    allanout[1] = np.std(Allanisotope)
    if verbose ==1:
        print("Std d18o raw of section: %0.3f" %(allanout[1]))
        print("Std d18o of fixed spacing: %0.3f" %(np.std(equal_Allanisotope)))
    lin = sp.interpolate.interp1d(tau, allans_Allanisotope)
    xpoints = [10, 60, 600, 3600]
    if len(tau) <= 3600:
        xpoints = [10, 60, 600]
    ypoints = lin (xpoints)
    allanout[2] = ypoints[0] #10 sec d18o
    allanout[3] = ypoints[1] #60 sec d18o
    allanout[4] = ypoints[2] #600 sec d18o
    if len(tau) >= 3600:
        allanout[5] = ypoints[3] #3600 sec dD (40 min)
    if verbose ==1:
        print("water conc of this section: %0.3f" %(mean_water_ppm))
    allanout[6] =  mean_water_ppm 
    if verbose ==1:
        print name, allanout   
    return allanout

#### VALCO IDENTIFICATION FUNCTION #############################################
## writen by Valerie Morris
def valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope):
    if name == "d18o":
        colorsymbol = "b-o"
        graphnumber = 1
    if name == "dD":
        colorsymbol = "r-o"
        graphnumber = 3
    if name == "2140d18o":
        colorsymbol = "b-o"
        graphnumber = 5
    if name == "2140dD":
        colorsymbol = "r-o"
        graphnumber = 7
    if name == "2140d17o":
        colorsymbol = "g-o"
        graphnumber = 9
        
    ##### Identify each standard from valco array 
    # AM valco in 0 position PM valco in 1 position   
    trans0 = [x for x in index[amvalcobegin[0]:] if valco_pos[x]==3 and valco_pos[x-2]==1 and valco_pos[x+100]==3]   
    if len(trans0)==0:
        trans0 = [x for x in index[amvalcobegin[0]:] if valco_pos[x]==3 and valco_pos[x-2]==0 and valco_pos[x+100]==3]
    if len(trans0)==0:
        trans0 = amvalcobegin[0]
    trans1 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==3 and valco_pos[x]==4 and valco_pos[x+100]==4]
    trans2 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==4 and valco_pos[x]==5 and valco_pos[x+100]==5]
    trans3 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==5 and valco_pos[x]==6 and valco_pos[x+100]==6]
    trans4 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==6 and valco_pos[x]==5 and valco_pos[x+100]==5]
    trans5 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==5 and valco_pos[x]==4 and valco_pos[x+100]==4]
    trans6 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==4 and valco_pos[x]==3 and valco_pos[x+100]==3]
    trans7 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==3 and valco_pos[x]==1 and valco_pos[x-100]==3]
    if len(trans7)==0:
        trans7 = [x for x in index[amvalcobegin[0]:] if valco_pos[x-2]==3 and valco_pos[x]==0 and valco_pos[x-100]==3]
        
    if verbose ==1:
        print "valco pos changes"
        print trans0
        print trans1
        print trans2
        print trans3
        print trans4
        print trans5
        print trans6
        print trans7

    # below assgnments were done by time delay from start of valco run in Melter_Crunch_4.2.py due to valco_pos not being recorded
    amtransbegin = np.arange(8)
    amtransbegin[0] = amvalcobegin[0]
    amtransbegin[1] = trans1[0]
    amtransbegin[2] = trans2[0]
    amtransbegin[3] = trans3[0]
    amtransbegin[4] = trans4[0]
    amtransbegin[5] = trans5[0]
    amtransbegin[6] = trans6[0]
    amtransbegin[7] = trans7[0]
    print 'amtransbegin first', amtransbegin, len(amtransbegin)

    pmtransbegin = np.arange(8)
    pmtransbegin[0] = pmvalcobegin[0]
    pmtransbegin[1] = trans1[-1]
    pmtransbegin[2] = trans2[-1]
    pmtransbegin[3] = trans3[-1]
    pmtransbegin[4] = trans4[-1]
    pmtransbegin[5] = trans5[-1]
    pmtransbegin[6] = trans6[-1]
    pmtransbegin[7] = trans7[-1]

    ##### Then Identify each standard from isotope transition array (can then be fed to trasfer function)
    ##### FIRST FOR AM VALCO
    amtransindexisotope = []
    for i in amtransbegin:
        tranbegin = i
        tranend = tranbegin+150
        isotopetransition = np.max(abs(diffisotope[tranbegin:tranend]))
        transisotope = [x for x in index[tranbegin:tranend] if abs(diffisotope[x])==isotopetransition]
        amtransindexisotope.append(transisotope[0])

    ##### REPEAT FOR PM VALCO 
    pmtransindexisotope = []
    for i in pmtransbegin:
        tranbegin = i
        tranend = tranbegin+150
        isotopetransition = np.max(abs(diffisotope[tranbegin:tranend]))
        transisotope = [x for x in index[tranbegin:tranend] if abs(diffisotope[x])==isotopetransition]
        pmtransindexisotope.append(transisotope[0])

    if verbose ==1:
        print "index for am valco isotope transitions for "+name
        print amtransindexisotope

        print "index for pm valco isotope transitions for "+name
        print pmtransindexisotope
    
    return amtransbegin, amtransindexisotope, pmtransbegin, pmtransindexisotope, fullmelttime

##### Valco Characterization function defined ##################################
def valcofunct(name, isotope, index, amtransbegin, amtransindex, pmtransbegin, pmtransindex, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1):
    water_ppm = data_dict["water_ppm"]
    if name == "d18o":
        colorsymbol = "b-o"
        graphnumber = 1
    if name == "dD":
        colorsymbol = "r-o"
        graphnumber = 3
    if name == "2140d18o":
        colorsymbol = "b-o"
        graphnumber = 5
    if name == "2140dD":
        colorsymbol = "r-o"
        graphnumber = 7
    if name == "2140d17o":
        colorsymbol = "g-o"
        graphnumber = 9
        
    ## RUN THE TRANSFER FUNCTION IN LOOP THROUGH ALL AM VALCO TRANSITIONS ##########
    count = 1
    counter = 1
    extrapolatedisotope = []
    sigma1_Transisotope = []
    sigma2_Transisotope = []
    valco_skewsigma_isotope = []
    valco_normsigma_isotope = []
    
    for i in amtransindex:        # ignoring the first and last to take into account when already on 3 or ending too soon, can not do since it screws up the extrapolated values later
        Index1 = i - valcopreindex
        Index2 = i + valcopostindex
        previousbegin = i-valcopostlength
        previousend = i-valcoprelength
        currentbegin = i+valcoprelength
        currentend = i+valcopostlength
        valcosegindex = index[Index1:Index2]
        valcosegindexnorm = valcosegindex - Index1
        valcosegisotope = isotope[Index1:Index2]
        previousvalueisotope = np.mean(isotope[previousbegin:previousend])
        currentvalueisotope = np.mean(isotope[currentbegin:currentend])
        stepsizeisotope = (currentvalueisotope-previousvalueisotope) 
        valcosegnormisotope = (valcosegisotope-previousvalueisotope)/stepsizeisotope
        
        badtransh2o = [x for x in index[Index1:Index2] if flag1[x]=="W"]
        if len(badtransh2o)<=200:
            if counter == 1:
                valcoindex = valcosegindexnorm
                valcoisotope = valcosegnormisotope
                counter = 2
#            #### only for when first transitions buggered - start, only matters for method A and B for WAIS06A where push ice was not used, but need to insert special cases, which was not recorded
#            if counter == 2:
#                valcoindex = valcosegindexnorm
#                valcoisotope = valcosegnormisotope
#                counter = 3
#            #### only for when first transitions buggered - end
            else:               #if counter > 2 and counter < 7:
                valcoindex = np.append(valcoindex,valcosegindexnorm)
                valcoisotope = np.append(valcoisotope,valcosegnormisotope)
                counter = counter+1
        
            trantype = 'AMValco'
            StepIndex = i
            run_am_valco_transfer = PerformTransfer(trantype, name, isotope, water_ppm, Index1, Index2, StepIndex)
            # output is: extrapolateisotope, sigma1_Transisotope, sigma2_Transisotope, time_n_Transisotope, cdf_Transisotope_norm
            extrapolatedisotope.append(run_am_valco_transfer[0])
            sigma1_Transisotope.append(run_am_valco_transfer[1])
            sigma2_Transisotope.append(run_am_valco_transfer[2])
            valco_skewsigma_isotope.append(run_am_valco_transfer[6])
            valco_normsigma_isotope.append(run_am_valco_transfer[7])
            if count == 2:
                amtimeisotope = run_am_valco_transfer[3]
                amcdfisotope = run_am_valco_transfer[4]
        count = count + 1

    ## RUN THE TRANSFER FUNCTION  IN LOOP THROUGH ALL PM VALCO TRANSITIONS #########
    count = 1
    counter = 1
    for i in pmtransindex:        # ignoring the first and last to take into account when already on 3 or ending too soon, can not do since it screws up the extrapolated values later
        Index1 = i - valcopreindex
        Index2 = i + valcopostindex
        previousbegin = i-valcopostlength
        previousend = i-valcoprelength
        currentbegin = i+valcoprelength
        currentend = i+valcopostlength
        valcosegindex = index[Index1:Index2]
        valcosegindexnorm = valcosegindex - Index1
        valcosegisotope = isotope[Index1:Index2]
        previousvalueisotope = np.mean(isotope[previousbegin:previousend])
        currentvalueisotope = np.mean(isotope[currentbegin:currentend])
        stepsizeisotope = (currentvalueisotope-previousvalueisotope) 
        valcosegnormisotope = (valcosegisotope-previousvalueisotope)/stepsizeisotope
        
        badtransh2o = [x for x in index[Index1:Index2] if flag1[x]=="W"]
        if len(badtransh2o)<=200:
            if counter == 1:
                valcoindex = valcosegindexnorm
                valcoisotope = valcosegnormisotope
                counter = 2
#            #### only for when first transitions buggered - start, only matters for method A and B where push ice was not used, but need to insert special cases, which was not recorded
#            if counter == 2:
#                valcoindex = valcosegindexnorm
#                valcoisotope = valcosegnormisotope
#                counter = 3
#            #### only for when first transitions buggered - end
            else:               #if counter > 2 and counter < 7:
                valcoindex = np.append(valcoindex,valcosegindexnorm)
                valcoisotope = np.append(valcoisotope,valcosegnormisotope)
                counter = counter +1
            trantype = 'PMValco'
            StepIndex = i
            run_pm_valco_transfer = PerformTransfer(trantype, name, isotope, water_ppm, Index1, Index2, StepIndex) 
            # output is: 0extrapolateisotope, 2sigma1_Transisotope, 3sigma2_Transisotope, 6time_n_Transisotope, 7cdf_Transisotope_norm
            extrapolatedisotope.append(run_pm_valco_transfer[0])
            sigma1_Transisotope.append(run_pm_valco_transfer[1])
            sigma2_Transisotope.append(run_pm_valco_transfer[2])
            valco_skewsigma_isotope.append(run_am_valco_transfer[6])
            valco_normsigma_isotope.append(run_am_valco_transfer[7])
            if count == 2:
                pmtimeisotope = run_pm_valco_transfer[3]
                pmcdfisotope = run_pm_valco_transfer[4]
        count = count + 1

    sortindex = np.argsort(valcoindex)
    sortedvalcoindex = valcoindex[sortindex]
    sortedvalcoisotope = valcoisotope[sortindex]
    avevalcoisotope = deepcopy(valcosegnormisotope)
    for p in valcosegindexnorm:
        aveindex = np.where(sortedvalcoindex==p)[0]
        avevalcoisotope[p] = np.mean(sortedvalcoisotope[aveindex])
    smavevalcoisotope = smooth(avevalcoisotope)
    valcosplineisotope = UnivariateSpline(sortedvalcoindex,sortedvalcoisotope,k=4, s=10.5) # (k=4, s=4.5) for dD
    valcomomemcoisotope = valcosplineisotope(valcosegindexnorm)
    normvalcomemcoisotope = (valcomomemcoisotope-np.min(valcomomemcoisotope))/(np.max(valcomomemcoisotope)-np.min(valcomomemcoisotope))
    valcoresisotope = valcomomemcoisotope-avevalcoisotope
    
    fig420 = plt.figure(graphnumber+30)
    clear = plt.clf()
    fig420_ax1 = fig420.add_subplot(111)
    fig420_ax1.plot(sortedvalcoindex, sortedvalcoisotope, "g-", valcosegindexnorm, avevalcoisotope, colorsymbol, valcosegindexnorm, smavevalcoisotope, "k-") #valcosegindexnorm, valcosegnormisotope, "b-", valcosegindexnorm, valcosegnormneamemcoisotope, "g-", 
    fig420_ax1.set_ylabel(name)
    fig420_ax1.set_xlabel("Index")
    fig420_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    if verbose ==1:
        print extrapolatedisotope
        print sigma1_Transisotope
        print sigma2_Transisotope

    avesigma1_Transisotope = np.mean(sigma1_Transisotope)
    avesigma2_Transisotope = np.mean(sigma2_Transisotope)

    ##### CALCULATE RAW VALUES FOR STANDARDS IN VALCO
    rawisotope = []
    stdevrawisotope = []
    print 'amtransbegin', amtransbegin, len(amtransbegin)
    for i in amtransbegin:
        begin = i-352 
        end = i
        rawisotope.append(np.mean(isotope[begin:end]))
        stdevrawisotope.append(np.std(isotope[begin:end]))
    #rawisotope.append(np.mean(isotope[begin:end]))
    #stdevrawisotope.append(np.std(isotope[begin:end]))
    print 'pmtransbegin', pmtransbegin, len(pmtransbegin)
    for i in pmtransbegin:
        begin = i-352 
        end = i
        rawisotope.append(np.mean(isotope[begin:end]))
        stdevrawisotope.append(np.std(isotope[begin:end]))
    #rawisotope.append(np.mean(isotope[begin:end]))
    #stdevrawisotope.append(np.std(isotope[begin:end]))
    print "rawisotope", rawisotope
    print "extrapolatedisotope", extrapolatedisotope
    
    ##### ASSIGN RAW VALUES ############################################    
    ## Assign first standard measured/raw isotope values
    rawfirstisotope = [rawisotope[1],rawisotope[7],rawisotope[9],rawisotope[15]]
#    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
#        rawfirstisotope = [rawisotope[1],rawisotope[7],rawisotope[10],rawisotope[16]]
    averawfirstisotope = np.mean(rawfirstisotope)
    stdevrawfirstisotope = np.std(rawfirstisotope)
    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
        extrapfirstisotope = [extrapolatedisotope[0],extrapolatedisotope[5],extrapolatedisotope[7],extrapolatedisotope[12]]
    else:
        extrapfirstisotope = [extrapolatedisotope[0],extrapolatedisotope[6],extrapolatedisotope[8],extrapolatedisotope[14]]
    aveextrapfirstisotope = np.mean(extrapfirstisotope)
    stdevextrapfirstisotope = np.std(extrapfirstisotope)

    ##### Assign second measured isotope values
    rawsecondisotope = [rawisotope[2],rawisotope[6],rawisotope[10],rawisotope[14]]
#    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
#        rawsecondisotope = [rawisotope[2],rawisotope[6],rawisotope[11],rawisotope[15]]
    averawsecondisotope = np.mean(rawsecondisotope)
    stdevrawsecondisotope = np.std(rawsecondisotope)
    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
        extrapsecondisotope = [extrapolatedisotope[1],extrapolatedisotope[8]]
    else:
         extrapsecondisotope = [extrapolatedisotope[1],extrapolatedisotope[5],extrapolatedisotope[9],extrapolatedisotope[13]]   
    aveextrapsecondisotope = np.mean(extrapsecondisotope)
    stdevextrapsecondisotope = np.std(extrapsecondisotope)

    ## Assign third measured isotope values
    rawthirdisotope = [rawisotope[3],rawisotope[5],rawisotope[11],rawisotope[13]]
    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
        rawthirdisotope = [rawisotope[3],rawisotope[12]]
    averawthirdisotope = np.mean(rawthirdisotope)
    stdevrawthirdisotope = np.std(rawthirdisotope)
    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
        extrapthirdisotope = [extrapolatedisotope[2],extrapolatedisotope[4],extrapolatedisotope[9],extrapolatedisotope[11]]
    else:
        extrapthirdisotope = [extrapolatedisotope[2],extrapolatedisotope[4],extrapolatedisotope[10],extrapolatedisotope[12]]
    aveextrapthirdisotope = np.mean(extrapthirdisotope)
    stdevextrapthirdisotope = np.std(extrapthirdisotope)

    ##### Assign fourth measured isotope values
    rawfourthisotope = [rawisotope[4],rawisotope[12]]
#    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
#        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
#        rawfourthisotope = [rawisotope[4],rawisotope[13]]
    averawfourthisotope = np.mean(rawfourthisotope)
    stdevrawfourthisotope = np.std(rawfourthisotope)
    if filename == "HIDS2143-20180710-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180711-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180712-000006Z-DataLog_User.dat" or \
        filename == "HIDS2143-20180713-000006Z-DataLog_User.dat":
        extrapfourthisotope = [extrapolatedisotope[3],extrapolatedisotope[10]]
    else:
        extrapfourthisotope = [extrapolatedisotope[3],extrapolatedisotope[11]]    
    aveextrapfourthisotope = np.mean(extrapfourthisotope)
    stdevextrapfourthisotope = np.std(extrapfourthisotope)

    if verbose ==1:
        print name
        print "first standard raw isotope valco values"
        print rawfirstisotope
        print averawfirstisotope, stdevrawfirstisotope
        print aveextrapfirstisotope, stdevextrapfirstisotope
        print "second standard raw isotope valco values"
        print rawsecondisotope
        print averawsecondisotope, stdevrawsecondisotope
        print aveextrapsecondisotope, stdevextrapsecondisotope
        print "third standard raw isotope valco values"
        print rawthirdisotope
        print averawthirdisotope, stdevrawthirdisotope
        print aveextrapthirdisotope, stdevextrapthirdisotope
        print "fourth raw isotope valco values"
        print rawfourthisotope
        print averawfourthisotope, stdevrawfourthisotope
        print aveextrapfourthisotope, stdevextrapfourthisotope
    
    return averawfirstisotope, stdevrawfirstisotope, averawsecondisotope, stdevrawsecondisotope, averawthirdisotope, \
        stdevrawthirdisotope, averawfourthisotope, stdevrawfourthisotope, smavevalcoisotope, valco_skewsigma_isotope, \
        valco_normsigma_isotope, aveextrapfirstisotope, stdevextrapfirstisotope, aveextrapsecondisotope, \
        stdevextrapsecondisotope, aveextrapthirdisotope, stdevextrapthirdisotope, aveextrapfourthisotope, stdevextrapfourthisotope

#### Neapolitan Characterization Function
def neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin):
    # only on low to high transitions
    low2high = [x for x in index[neabegin[-1]:] if comments[x-2]!=101 and comments[x]==101]
    high2low = [x for x in index[neabegin[-1]:] if comments[x-2]==101 and comments[x]==104] ## experimenting
    tranbegin = low2high[0]
    #if len(high2low) == 0:
    tranend = tranbegin+350  #was 350 for smooth cubic spline, was 150 for a while, 20231127 changing back to 350 since EC meter inline
    #else:
    #    tranend = high2low[0]
    print "nea transition range ", name, tranbegin, tranend
    isotopetransition = np.max(diffisotope[tranbegin:tranend])
    low2hightrans = [x for x in index[tranbegin:tranend] if diffisotope[x]==isotopetransition]
    print "low2hightrans", low2hightrans
    previousbegin = low2hightrans[0]-260
    previousend = low2hightrans[0]-200
    currentbegin = low2hightrans[0]+200
    currentend = low2hightrans[0]+260
    print "nea previous begin and end", previousbegin, previousend
    print "nea current begin and end", currentbegin, currentend
    previousvalue = np.mean(isotope[previousbegin:previousend])
    currentvalue = np.mean(isotope[currentbegin:currentend])
    stepsize = (currentvalue-previousvalue)
    midvalue = (currentvalue+previousvalue)/2
    if name == "d18o" or name == "2140d18o":
        midpoints = [x for x in index[tranbegin:tranend] if isotope[x] >= midvalue-0.4 and isotope[x] <= midvalue+0.4]
    if name == "dD" or name == "2140dD":
        midpoints = [x for x in index[tranbegin:tranend] if isotope[x] >= midvalue-2 and isotope[x] <= midvalue+2]
    if name == "d17o" or name == "2140d17o":
        midpoints = [x for x in index[tranbegin:tranend] if isotope[x] >= midvalue-.2 and isotope[x] <= midvalue+0.2]
    print "midpoints", midpoints
    nummidpoints = len(midpoints)
    if nummidpoints>0:
        midpoint = np.ceil(np.mean(midpoints[:]))
    else:
        midpoint = low2hightrans[0]
    print "midpoint", midpoint
    Index1 = midpoint-250 #was 260 but was too big for method B
    Index2 = midpoint+350 #20231127 changing from 250 to 300 to see if it fixes issue cause be EC meter causing delay, then 300-350 to get 20231106 to go through
    neaindex = index[Index1:Index2]
    neaindexnorm = neaindex - Index1
    neaisotope = isotope[Index1:Index2]
    neanormisotope = (neaisotope-previousvalue)/stepsize
    
    ##### RUN THE TRANSFER FUNCTION ################################################
    nea_skewsigma = []
    nea_normsigma = []
    StepIndex = low2hightrans[0]
    run_nea_transfer = PerformTransfer(trantype, name, isotope, water_ppm, Index1, Index2, StepIndex)
    nea_skewsigma.append(run_nea_transfer[6])
    nea_normsigma.append(run_nea_transfer[7])

    return neaindexnorm, neaisotope, neanormisotope, nea_skewsigma, nea_normsigma, tranend

##### TRANSFER FUNCTION DEFINED ################################################
## writen by Vasileios Gkinis, modified by Valerie Morris

def lognorm_cdf(x, s, mu, sigma):
    return stats.lognorm.cdf(x, s, loc = mu, scale = sigma)

def lognorm_pdf(x, s, mu, sigma):
    return stats.lognorm.pdf(x, s, loc = mu, scale = sigma)

def log_product(x, a, b, shape1, mu, r1, shape2, r2):
    f1 = lognorm_cdf(x, shape1, mu, r1)
    f2 = lognorm_cdf(x, shape2, mu, r2)
    return a*(f1*f2)+b

def lsq_log_product(p, x, a, b, c, d):
    f1 = lognorm_cdf(x, c, p[0], p[1])
    f2 = lognorm_cdf(x, d, p[0], p[2])
    return a*(f1*f2)+b

def err_lsq_log_product(p, x, y, a, b, c, d):
    return lsq_log_product(p, x, a, b, c, d)-y

def skew(p,x):                                                                  #skew pdf
    #t = (x-e)/w
    # 2/w * lognorm_pdf(t,s,mu,sigma) * lognorm_cdf(a*t,s,mu,sigma)
    #p[0]=e, p[1]=w, p[2]=a, p[3]=s, p[4]=mu, p[5]=sigma, p_i = [-0.5,1,5,3,0,10]
    t = x                      #t = (x-e) / w, e=0, w=1, a=1.06
    return 2* lognorm_pdf(t,0.34,p[0],p[1]) * lognorm_cdf(1.06*t,0.34,p[0],p[1])

def err_lsq_skew(p,x,y):
    return skew(p,x)-y

def skew_cdf1(p,x):                                                            #skew cdf take 1, with special case listed on wolfram
    return lognorm_cdf(1.06*x,0.34,p[0],p[1])-2*(1/8*erf(-x/np.sqrt(2))*erf(x/np.sqrt(2)))

def err_lsq_skew_cdf1(p,x,y):
    return skew_cdf1(p,x)-y

#####################################################################################
"""
Created on Sat Jan 26 16:19:12 2013

@author: Janwillem van Dijk
@email: jwe.van.dijk@xs4all.nl

Module for generating skew normal random numbers (Adelchi Azzalini)
===================================================================
http://azzalini.stat.unipd.it/SN/

Licensing:
This code is distributed under the GNU LGPL license.

-   rnd_skewnormal: returns random valuse for sn distribution with given
        location scale and shape
-   random_skewnormal: returns random valuse for sn distribution with given
        mean, stdev and skewness
-   skewnormal_parms: returns location, scale and shape given
        mean, stdev and skewnessof the sn distribution
-   skewnormal_stats: returns mean, stdev and skewness given
        location scale and shape
-   pdf_skewnormal: returns values for the pdf of a skew normal distribution
-   cdf_skewnormal: returns values for the cdf of a skew normal distribution
-   T_owen returns: values for Owens T as used by cdf_skewnormal
-   skew_max: returns the maximum skewness of a sn distribution
"""
try:
    """
    Try to use owen.f90 compiled into python module with
    f2py -c -m owens owens.f90
    ginving owens.so
    http://people.sc.fsu.edu/~jburkardt/f_src/owens/owens.f90
    """
    owens = None
    #import owens
except:
    print 'owens not found'

def T_Owen_int(h, a, jmax=50, cut_point=6):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owen's T
    @param: a   the a parameter of Owen's T (-1 <= a <= 1)
    Python-numpy-scipy version for Owen's T translated from matlab version
        T_owen.m of R module sn.T_int
    """
    if type(h) in (float, float64):
        h = array([h])
    low = where(h <= cut_point)[0]
    high = where(h > cut_point)[0]
    n_low = low.size
    n_high = high.size
    irange = np.arange(0, jmax)
    series = zeros(h.size)
    if n_low > 0:
        h_low = h[low].reshape(n_low, 1)
        b = fui(h_low, irange)
        cumb = b.cumsum(axis=1)
        b1 = np_exp(-0.5 * h_low ** 2) * cumb
        matr = ones((jmax, n_low)) - b1.transpose()  # matlab ' means transpose
        jk = kron(ones(jmax), [1.0, -1.0])
        jk = jk[0: jmax] / (2 * irange + 1)
        matr = inner((jk.reshape(jmax, 1) * matr).transpose(),
                     a ** (2 * irange + 1))
        series[low] = (np_arctan(a) - matr.flatten(1)) / (2 * pi)
    if n_high > 0:
        h_high = h[high]
        atana = np_arctan(a)
        series[high] = (atana * np_exp(-0.5 * (h_high ** 2) * a / atana) *
                    (1.0 + 0.00868 * (h_high ** 4) * a ** 4) / (2.0 * pi))
    return series

def fui(h, i):
    return (h ** (2 * i)) / ((2 ** i) * sp_gamma(i + 1))

def T_Owen_series(h, a, jmax=50, cut_point=6):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owen's T
    @param: a   the a parameter of Owen's T
    Python-numpy-scipy version for Owen's T
    Python-numpy-scipy version for Owen's T translated from matlab version
        T_owen.m of R module sn.T_Owen
    """
    if abs(a) <= 1.0:
        return T_Owen_int(h, a, jmax=jmax, cut_point=cut_point)
    else:
        """D.B. Owen Ann. Math. Stat. Vol 27, #4 (1956), 1075-1090
         eqn 2.3, 2.4 and 2.5
         Available at: http://projecteuclid.org/DPubS/Repository/1.0/
            Disseminate?view=body&id=pdf_1&handle=euclid.aoms/1177728074"""
        signt = copysign(1.0, a)
        a = abs(a)
        h = abs(h)
        ha = a * h
        gh = norm.cdf(h)
        gha = norm.cdf(ha)
        t = 0.5 * gh + 0.5 * gha - gh * gha - \
                T_Owen_int(ha, 1.0 / a, jmax=jmax, cut_point=cut_point)
        return signt * t

def T_Owen(h, a):
    """
    Return Owens T
    ==============
    @param: h   the h parameter of Owens T
    @param: a   the a parameter of Owens T
    Try to use owens.f90 version else python version
    owens.f90 is approximately a factor 100 faster
    """
    if owens:
        """Owen's T using owens.f90 by Patefield and Brown
            http://www.jstatsoft.org/v05/a05/paper
            Fortran source by Burkhard
            http://people.sc.fsu.edu/~jburkardt/f_src/owens/owens.f90"""
        if type(h) in [float, float64]:
            return owens.t(h, a)
        else:
            t = zeros(h.size)
            for i in range(h.size):
                t[i] = owens.t(h[i], a)
            return t
    else:
        """
        Owens T after sn.T_Owen(H, a) D.B. Owen (1956)
        """
        return T_Owen_series(h, a)

def cdf_skewnormal(p, x):
    """
    Return skew normal cdf
    ======================
    p[0] = @param location:    location of sn distribution(e)
    p[1] = @param scale:       scale of sn distribution(w)
    p[2] = @param shape:       shape of sn distribution(a)
    http://azzalini.stat.unipd.it/SN/
    """
    xi = (x - p[0]) / p[1]
    return norm.cdf(xi) - 2.0 * T_Owen(xi, p[2])

def err_lsq_cdf_skewnormal(p, x, y):
    return cdf_skewnormal(p, x)-y

def pdf_skewnormal(p, x):
    """
    Return skew normal pdf
    ======================
    p[0] = @param location:    location of sn distribution(e)
    p[1] = @param scale:       scale of sn distribution(w)
    p[2] = @param shape:       shape of sn distribution(a)
    http://azzalini.stat.unipd.it/SN/
    """
    t = (x - p[0]) / p[1]
    return 2.0 / p[1] * norm.pdf(t) * norm.cdf(p[2] * t)
def err_lsq_pdf_skewnormal(p, x, y):
    return pdf_skewnormal(p, x)-y

def impulse(sig,time):                                                          #normal distribution pdf with sig =1, (1/(np.sqrt(2*np.pi))*np.exp(-1*(np.square(time)/(2)))
    return (1/(sig*(np.sqrt(2*np.pi))))*np.exp(-1*(np.square(time)/(2*np.square(sig))))

def err_lsq_impulse(sig,time,y):
    return impulse(sig,time)-y

def PerformTransfer(trantype, name, isotope, water_ppm, TransferIndex1, TransferIndex2, StepIndex):
    ## INPUT INDICES FOR TRANSFRER FUNCTION ############################
    TransferIndex1 = np.float(TransferIndex1)
    TransferIndex2 = np.float(TransferIndex2)
    StepIndex = secs[np.float(StepIndex)]
    
    if name == "d18o":
        colorsymbol = "b-o"
        graphnumber = 1
        p_i = [StepIndex, 2.0, 35] #[mu, sigma1, sigma2]
        s1_Transisotope = valcoshape1d18o
        s2_Transisotope = valcoshape2d18o
        if trantype == "AMNea" or trantype == "PMNea":
            p_i = [StepIndex, 2.0, 50]  #[mu, sigma1, sigma2]
            s1_Transisotope = neashape1d18o
            s2_Transisotope = neashape2d18o
            graphnumber = 11
        y_i_isotope = [-15.0,25.0,4.0] # [location(e), scale (w), shape(a)] not necessarily mu, scale and sigma
    if name == "dD":
        colorsymbol = "r-o"
        graphnumber = 3
        p_i = [StepIndex, 0.4, 32]  #[mu, sigma1, sigma2]
        s1_Transisotope = valcoshape1dD
        s2_Transisotope = valcoshape2dD
        if trantype == "AMNea" or trantype == "PMNea":
            p_i = [StepIndex, 2.0, 50]  #[mu, sigma1, sigma2], same for both isotopes
            s1_Transisotope = neashape1dD
            s2_Transisotope = neashape2dD
            graphnumber = 13
        y_i_isotope =  [-15.0,25.0,4.0] # [location, scale, shape] not necessarily mu, scale and sigma
    if name == "2140d18o":
        colorsymbol = "b-o"
        graphnumber = 5
        p_i = [StepIndex, 2.0, 35]  #[mu, sigma1, sigma2]
        s1_Transisotope = valcoshape1d18o
        s2_Transisotope = valcoshape2d18o
        if trantype == "AMNea" or trantype == "PMNea":
            p_i = [StepIndex, 2.0, 50]  #[mu, sigma1, sigma2]
            s1_Transisotope = neashape1d18o
            s2_Transisotope = neashape2d18o
            graphnumber = 15
        y_i_isotope = [-15.0,25.0,4.0] # [location(e), scale (w), shape(a)] not necessarily mu, scale and sigma
    if name == "2140dD":
        colorsymbol = "r-o"
        graphnumber = 7
        p_i = [StepIndex, 0.4, 32]  #[mu, sigma1, sigma2]
        s1_Transisotope = valcoshape1dD
        s2_Transisotope = valcoshape2dD
        if trantype == "AMNea" or trantype == "PMNea":
            p_i = [StepIndex, 2.0, 50]  #[mu, sigma1, sigma2], same for both isotopes
            s1_Transisotope = neashape1dD
            s2_Transisotope = neashape2dD
            graphnumber = 17
        y_i_isotope =  [-15.0,25.0,4.0] # [location, scale, shape] not necessarily mu, scale and sigma
    if name == "2140d17o":
        colorsymbol = "g-o"
        graphnumber = 9
        p_i = [StepIndex, 2.0, 35]  #[mu, sigma1, sigma2]
        s1_Transisotope = valcoshape1d18o
        s2_Transisotope = valcoshape2d18o
        if trantype == "AMNea" or trantype == "PMNea":
            p_i = [StepIndex, 2.0, 50]  #[mu, sigma1, sigma2]
            s1_Transisotope = neashape1d18o
            s2_Transisotope = neashape2d18o
            graphnumber = 18
        y_i_isotope = [-15.0,25.0,4.0] # [location(e), scale (w), shape(a)] not necessarily mu, scale and sigma
            
    Transsecs = secs[TransferIndex1:TransferIndex2]
    Transwater_ppm = water_ppm[TransferIndex1:TransferIndex2]
    Transisotope = isotope[TransferIndex1:TransferIndex2]
    startvalueisotope = np.mean(Transisotope[0:40])
    endvalueisotope = np.mean(Transisotope[-41:-1])
    if len(Transisotope)<=800:
        stepisotope = (endvalueisotope - startvalueisotope)/0.9957314 # (0.9879513 for dD)
    if len (Transisotope)>=800:
        stepisotope = endvalueisotope - startvalueisotope
    newisotopeend = startvalueisotope+stepisotope
            
    if verbose ==1:
        print startvalueisotope, newisotopeend, stepisotope

    time_step = np.ceil(mean_time_delay)
    equal_secs = np.arange(Transsecs[0], Transsecs[-1], time_step)
    equal_Transisotope = np.interp(equal_secs, Transsecs, Transisotope)
    norm_Transisotope = (startvalueisotope-equal_Transisotope)/(startvalueisotope-newisotopeend)
    
    ## Fit log*log cdf on isotope data  ################################################
    p_opt = sp.optimize.leastsq(err_lsq_log_product, p_i, args=(equal_secs, equal_Transisotope, stepisotope, startvalueisotope, s1_Transisotope, s2_Transisotope), maxfev=10000)[0]
    if verbose ==1:
        print("\n\n\n" + 80*"-" + "\nFitting isotope step with model a*(f1*f2)+b with lsq and shapes fixed")
        print "**** p_opt *****", p_opt
    mu_Transisotope = p_opt[0]
    sigma1_Transisotope = p_opt[1]
    sigma2_Transisotope = p_opt[2]
    isotope_Diff_Len = p_opt[-1]
    extrapolateisotope = log_product(4000, stepisotope, startvalueisotope, s1_Transisotope, 0, sigma1_Transisotope, s2_Transisotope, sigma2_Transisotope)
    if verbose ==1:
        print extrapolateisotope
    cdf_Transisotope = lsq_log_product(p_opt, equal_secs, stepisotope, startvalueisotope, s1_Transisotope, s2_Transisotope)
    cdf_Transisotope_norm = (cdf_Transisotope - startvalueisotope)/stepisotope
    cdf_residualisotope = err_lsq_log_product(p_opt, equal_secs, equal_Transisotope, stepisotope, startvalueisotope, s1_Transisotope, s2_Transisotope)
    time_n_Transisotope = equal_secs-equal_secs[cdf_Transisotope_norm>=0.5][0]

    #### Fit skew_cdf on isotope data #################################################
    y_opt_isotope = sp.optimize.leastsq(err_lsq_cdf_skewnormal, y_i_isotope, args=(time_n_Transisotope, norm_Transisotope))[0]
    skew_cdf_Transisotope = cdf_skewnormal(y_opt_isotope, time_n_Transisotope)
    skew_cdf_residualisotope = err_lsq_cdf_skewnormal(y_opt_isotope, time_n_Transisotope, norm_Transisotope)*stepisotope
    skew_alpha_isotope = y_opt_isotope[2]
    skew_omega_isotope = y_opt_isotope[1]
    skew_delta_isotope = skew_alpha_isotope/(np.sqrt(1+np.square(skew_alpha_isotope)))
    skew_var_isotope = np.sqrt(np.square(skew_omega_isotope)*(1-(2*np.square(skew_delta_isotope)/np.pi)))
    print "skewnormal cdf variance for ", trantype, skew_var_isotope

    ## Plot CDF's  #####################################################
    fig4 = plt.figure(graphnumber+40)
    fig4_ax1 = fig4.add_subplot(311)
    fig4_ax1.plot(equal_secs, equal_Transisotope, colorsymbol)
    fig4_ax1.plot(equal_secs, cdf_Transisotope, "k", linewidth = 2)
    fig4_ax1.set_ylabel(name)
    fig4_ax1.set_xlabel("Time [sec]")
    fig4_ax3 = fig4.add_subplot(312)
    fig4_ax3.plot(time_n_Transisotope, norm_Transisotope, colorsymbol)
    fig4_ax3.plot(time_n_Transisotope, cdf_Transisotope_norm, "k-", linewidth = 2)    #log*log
    fig4_ax3.plot(time_n_Transisotope, skew_cdf_Transisotope, "g-", linewidth = 1)    #skew normal
    fig4_ax3.set_ylabel(name)
    fig4_ax3.set_xlabel("Time [sec]")
    fig4_ax5 = fig4.add_subplot(313)
    fig4_ax5.plot(equal_secs, cdf_residualisotope, "k-", linewidth = 2)            #log*log
    fig4_ax5.plot(equal_secs, skew_cdf_residualisotope, "g-", linewidth = 1)       #skew normal
    fig4_ax5.set_ylabel(name)
    fig4_ax5.set_xlabel("Time [sec]")
    fig4_ax1.set_title("%s - %0.0f - %0.0f" %(os.path.splitext(filepath)[0], TransferIndex1, TransferIndex2))
    
    ## Frequencies  #################################################### ?
    dt = equal_secs[1] - equal_secs[0]
    N = np.size(equal_secs)
    pad = 2048-N
    Nf = np.size(equal_secs) + pad
    freq = fft.fftfreq(Nf, dt)
    f_nyq = 1./(2*dt)
    if verbose ==1:
        print "\ndt: %0.2f    N: %i    fnyq: %.3f Hz\n" %(dt, Nf, f_nyq)
    
    ## Transisotope transfer function   ################################### ?
    diff_Transisotope = np.gradient(equal_Transisotope, dt)/stepisotope                  #first derivative of equally spaced isotope data normaized to step size
    fft_diff_Transisotope = fft.fft(diff_Transisotope, Nf)*dt                         #fourierre transform of data
    absolute_Transisotope = np.abs(fft_diff_Transisotope)                             #absolute value of fourierre transform of data
    diff_cdf_Transisotope = np.gradient(cdf_Transisotope_norm, dt)                    #first derivate of normalized fit cdf function
    fft_diff_cdf_Transisotope = fft.fft(diff_cdf_Transisotope, Nf)*dt                 #fourierre transform of fit
    absolute_cdf_Transisotope = np.abs(fft_diff_cdf_Transisotope)                     #absolute value of fourierre transform of fit
    
    # Get diffusion length in time ####################################
    ## Normal distribution
    sigi_isotope = 10                                                              # apriori sigma
    sig_opt_isotope = sp.optimize.leastsq(err_lsq_impulse, sigi_isotope, args=(time_n_Transisotope, diff_Transisotope))[0] #(was being fitted to diff of cdf, not data, does that really equal pdf?)
    print "isotope sigma", trantype, sig_opt_isotope
    impulse_isotope = impulse(sig_opt_isotope,time_n_Transisotope)
    
    # Get diffusion length in time ####################################
    w_i_isotope = [-15.0,25.0,4.0]
    w_opt_isotope = sp.optimize.leastsq(err_lsq_pdf_skewnormal, w_i_isotope, args=(time_n_Transisotope, diff_Transisotope))[0]
    print "isotope pdf_skewnormal variables ", trantype, w_opt_isotope
    skewnormalimpulse_isotope = pdf_skewnormal(w_opt_isotope, time_n_Transisotope)
    resisdualimpulse_isotope = err_lsq_pdf_skewnormal(w_opt_isotope, time_n_Transisotope,diff_Transisotope)
    skew_cdf_Transisotope = cdf_skewnormal(w_opt_isotope, time_n_Transisotope)
    skew_cdf_residualisotope = err_lsq_cdf_skewnormal(w_opt_isotope, time_n_Transisotope, norm_Transisotope)*stepisotope
    skew_alpha_isotope = w_opt_isotope[2]
    skew_omega_isotope = w_opt_isotope[1]
    skew_delta_isotope = skew_alpha_isotope/(np.sqrt(1+np.square(skew_alpha_isotope)))
    skew_var_isotope = np.sqrt(np.square(skew_omega_isotope)*(1-(2*np.square(skew_delta_isotope)/np.pi)))
    print "skewnormal pdf variance for ", trantype, skew_var_isotope
    
    ## PLOT IMPULSE RESPONSE ###########################################
    fig5 = plt.figure(graphnumber+41)
    fig5_ax1 = fig5.add_subplot(311)
    fig5_ax1.plot(time_n_Transisotope, diff_Transisotope, colorsymbol)                       #diff of data
    fig5_ax1.plot(time_n_Transisotope, impulse_isotope, 'k', linewidth = 1)           #normal pdf
    fig5_ax1.plot(time_n_Transisotope, skewnormalimpulse_isotope, colorsymbol, linewidth = 1) #skew normal pdf
    fig5_ax1.grid(True)
    fig5_ax1.set_xlabel("Time [sec]")
    fig5_ax1.set_ylabel(name)
    fig5_ax3 = fig5.add_subplot(313)
    fig5_ax3.grid(True)
    fig5_ax3.plot(time_n_Transisotope, resisdualimpulse_isotope, colorsymbol)
    fig5_ax1.set_title("%s - %0.0f - %0.0f" %(os.path.splitext(filepath)[0], TransferIndex1, TransferIndex2))

    fig4_ax3.plot(time_n_Transisotope, skew_cdf_Transisotope, "y-", linewidth = 1)    #pdf transform to cdf
    fig4_ax5.plot(equal_secs, skew_cdf_residualisotope, "y-", linewidth = 1)       #pdf transform to cdf
    
    ##### First two are VALCO OUTPUT VARIABLES TO ARRAY TO BE USED LATER IN Calibriation if 5 min valco
    ##### Three onwards are curve fit variables to be used in performance file, memory correction, and deconvolution
    ##### IF NEAPOLITAN, WRITE TO COMMON FILE TO BE USED IN DECONVOLUTION... INDICATORS AM = FIRST BEG DEPTH, PM = LAST END DEPTH  
    return extrapolateisotope, sigma1_Transisotope, sigma2_Transisotope, time_n_Transisotope, cdf_Transisotope_norm, mu_Transisotope, skew_var_isotope, sig_opt_isotope

##### SMOOTHING FUNCTION #######################################################
## written by Vasileios Gkinis
def smooth(x, window_len=21, window='hamming'): # window length was 11 in previous versions
    """
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
    """
    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    y = y[window_len-1:-window_len+1]

    return y

##### PICKPOINTS TO SELCT DATA TO FLAG AT THE END OF THE CODE ##################
def pickpoints(y,x,ffile):
    flagfile = open(ffile, "w")
    with open(ffile, "w") as flagfile:
        pg = PointGetter(file=flagfile) 
        pg.ax.plot(x, y, 'r.', picker=15)
        pg.ax.set_title('Zoom into area, \n get out of zoom function, \n hold down a flag key to select points ({keys})'.format(
            keys=','.join(VALID_FLAGS)))
        plt.show()
    flagfile.close()
    
##### MEMORY APPLICATION FUNCTION ##############################################
#later will be broken out into a function, but for now, repeated various places in the code

##########################################################################################################################################
##### BEGIN FULL PROGRAM #################################################################################################################
##########################################################################################################################################
    
##### READ IN DATA #############################################################
# Batch running raw_dictionaries
#for root, dirs, files in os.walk('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/raw/'): #'/raw_dictionaries'
files = []
folders = []
#for (path, dirnames, filenames) in os.walk("/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/raw/"):
for (path, dirnames, filenames) in os.walk("/Users/frio/EGRIP_2023/Data/raw_dictionaries/"): # raw_dictionaries/
    folders.extend(os.path.join(path, name) for name in dirnames)
    files.extend(os.path.join(path, name) for name in filenames)
    files = sorted(files)
    print files
    if files[0] == "/Users/frio/EGRIP_2023/Data/raw_dictionaries/.DS_Store":
    	print len(files)
    	files = files[1:]
    	print len(files)

for file in files[:]:   
    filepath = file
    splitfilepath = filepath.rpartition("/")
    filename = splitfilepath[-1]
    print filepath
    splitfilename = filename.rpartition("-")

    ##### READ IN RAW DATA FILE FROM MELTER ########################################
    if verbose ==1:
        filepath = raw_input("Give filepath: ")
        splitfilepath = filepath.rpartition("/")
        filename = splitfilepath[-1]
    d18odate = filename[9:17]
    if filename.startswith('raw'):
            d18odate = filename[12:20]
            filename = filename [3:]

    ## If already processed, and reading from binary dictionary file
    #filetype = [i for i in range(len(filepath)) if filepath.startswith('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/raw_dictionaries/raw')] 
    filetype = [i for i in range(len(filepath)) if filepath.startswith('/Users/frio/EGRIP_2023/Data/raw_dictionaries/raw')] 
    if len(filetype) >= 1:
        data = open(filepath, "r")                                   # open bininary file to read
        data_dict = pickle.load(data)  
        data_dict["flag"] = data_dict["flag"][:].astype("S")
        data_dict["flag1_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_d18o"][:] = '.' 
        data_dict["flag1_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_dD"][:] = '.' 
        data_dict["flag1_ec"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_ec"][:] = '.'
        data_dict["flag1_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_2140_d18o"][:] = '.' 
        data_dict["flag1_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_2140_dD"][:] = '.' 
        data_dict["flag1_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag1_2140_d17o"][:] = '.' 
        data_dict["flag2"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag2"][:] = '.' 
        data_dict["flag3"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag3"][:] = '.'  
        data_dict["flag4_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag4_d18o"][:] = '.' 
        data_dict["flag4_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag4_dD"][:] = '.' 
        data_dict["flag4_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag4_2140_d18o"][:] = '.' 
        data_dict["flag4_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag4_2140_dD"][:] = '.' 
        data_dict["flag4_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag4_2140_d17o"][:] = '.' 
        data_dict["flag5"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag5"][:] = '.' 
        data_dict["flag6"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag6"][:] = '2130'
        data_dict["flag7_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag7_d18o"][:] = '.' 
        data_dict["flag7_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag7_dD"][:] = '.' 
        data_dict["flag7_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag7_2140_d18o"][:] = '.' 
        data_dict["flag7_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag7_2140_dD"][:] = '.' 
        data_dict["flag7_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag7_2140_d17o"][:] = '.' 
        data_dict["flag8"] = deepcopy(data_dict["flag"]).astype("S")
        data_dict["flag8"][:] = '.' 
        #if filepath.startswith('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/raw_dictionaries/rawHIDS'):
        if filepath.startswith('/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS'):
            data_dict["flag6"][:] = '2130' # automatic instrument flag
    else:
        data = np.loadtxt(filepath, skiprows = 1, dtype = "S")
        data_dict = {}
        
        ## If run on Picarro 2130, KES read in with this formatting ####################
        ## If run on Picarro 2130 and a test, KES read in with this formatting #########
        #filetype = [i for i in range(len(filepath)) if filepath.startswith("/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/raw/HIDS") or filepath.startswith('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/test/HIDS')] 
        filetype = [i for i in range(len(filepath)) if filepath.startswith("/Users/frio/EGRIP_2023/Data/raw/HIDS") or filepath.startswith('/Users/frio/EGRIP_2023/Data/test/HIDS')] 
        if len(filetype) >= 1:
            data_dict["j_days"] = data[:,4].astype("float") #in 2023 Julian day was in 5th column (4) but was listed in here as 3rd (2)
            data_dict["j_days"] = data_dict["j_days"] * 60 * 60 * 24         # convert into seconds
            data_dict["time_delay"] = np.diff(data_dict["j_days"])
            data_dict["epoch"] = data[:,5].astype("float")[:-1]
            data_dict["water_ppm"] = data[:,16].astype("float")[:-1]
            data_dict["d18o"] = data[:,17].astype("float")[:-1] 
            data_dict["dD"] = data[:,18].astype("float")[:-1] 
            data_dict["start_depth"] = data[:,27].astype("float")[:-1]
            data_dict["end_depth"] = data[:,28].astype("float")[:-1]
            data_dict["laser_distance"] = data[:,29].astype("float")[:-1]
            data_dict["true_depth"] = data[:,30].astype("float")[:-1]
            data_dict["ec_value"] = data[:,31].astype("float")[:-1]
            data_dict["valco_pos"] = data[:,32].astype("float")[:-1]
            data_dict["comments"] = data[:,33].astype("float")[:-1]
            data_dict["melt_rate"] = data[:,37].astype("float")[:-1]
            data_dict["flag"] = np.arange(len(data_dict["time_delay"]))
            data_dict["flag1_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_d18o"][:] = '.' 
            data_dict["flag1_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_dD"][:] = '.' 
            data_dict["flag1_ec"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_ec"][:] = '.'
            data_dict["flag1_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_2140_d18o"][:] = '.' 
            data_dict["flag1_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_2140_dD"][:] = '.' 
            data_dict["flag1_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag1_2140_d17o"][:] = '.' 
            data_dict["flag2"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag2"][:] = '.' 
            data_dict["flag3"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag3"][:] = '.'  
            data_dict["flag4_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag4_d18o"][:] = '.' 
            data_dict["flag4_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag4_dD"][:] = '.' 
            data_dict["flag4_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag4_2140_d18o"][:] = '.' 
            data_dict["flag4_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag4_2140_dD"][:] = '.' 
            data_dict["flag4_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag4_2140_d17o"][:] = '.' 
            data_dict["flag5"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag5"][:] = '.' 
            data_dict["flag6"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag6"][:] = '2130'
            data_dict["flag7_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag7_d18o"][:] = '.' 
            data_dict["flag7_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag7_dD"][:] = '.' 
            data_dict["flag7_2140_d18o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag7_2140_d18o"][:] = '.' 
            data_dict["flag7_2140_dD"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag7_2140_dD"][:] = '.' 
            data_dict["flag7_2140_d17o"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag7_2140_d17o"][:] = '.' 
            data_dict["flag8"] = deepcopy(data_dict["flag"]).astype("S")
            data_dict["flag8"][:] = '.' 
            data_dict["iceflag"] = data[:,34].astype("float")[:-1]
            data_dict["smedlyflag"] = data[:,35].astype("float")[:-1]
            data_dict["loadflag"] = data[:,36].astype("float")[:-1]
            data_dict["MeltRate"] = data[:,37].astype("float")[:-1]

	data_dict["dexcess"] = data_dict["dD"]-8*data_dict["d18o"]
	data_dict["index"] = np.arange(len(data_dict["time_delay"]))
	data_dict["2140d18o"] = np.arange(len(data_dict["time_delay"])) - np.arange(len(data_dict["time_delay"]))
	data_dict["2140dD"] = np.arange(len(data_dict["time_delay"])) - np.arange(len(data_dict["time_delay"]))
	data_dict["2140d17o"] = np.arange(len(data_dict["time_delay"])) - np.arange(len(data_dict["time_delay"]))
	data_dict["2140_water_ppm"] = np.arange(len(data_dict["time_delay"])) - np.arange(len(data_dict["time_delay"]))

    allpositive = np.where(data_dict["time_delay"]>= 0)[0]
    data_dict["time_delay"] = data_dict["time_delay"][allpositive]
    mean_time_delay = np.mean(data_dict["time_delay"])
    secs = np.cumsum(data_dict["time_delay"])
    
            
    if data_dict.has_key('ec_value')==False:
		data_dict["ec_value"] = np.arange(len(data_dict["time_delay"])) - np.arange(len(data_dict["time_delay"]))
		    
##### Leaving in, for 2140 data from CIC    
    #if data_dict["epoch"][0] >= 1466694499:
    d17oflag = 1
    if corename == 'EGRIP' and d17oflag == 1:
        # KAW, UWWW, KPW(trap), VW1F
    	knownd18o = [kbwd18o, kawd18o, kpwd18o]       
        knowndD   = [kbwdD, kawdD, kpwdD]
        knownd17o = [kbwd17o, kawd17o, kpwd17o]
        knownD17o = [kbwD17o, kawD17o, kpwD17o]
        
        #read in data file 
        #for root, dirs, d17ofiles in os.walk('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/2140_raw'):
        for root, dirs, d17ofiles in os.walk('/Users/frio/EGRIP_2023/Data/2140_raw'):
            if verbose ==1:
                print d17ofiles
        if d17ofiles[0] == '.DS_Store':
            d17ofiles = d17ofiles[1:]
        if d17ofiles[0] == 'Icon\r':
            d17ofiles = d17ofiles[1:]
        if d17ofiles[-1] == 'Icon\r':
            d17ofiles = d17ofiles[:-1]
        for d17ofile in d17ofiles[:]:   
            #d17ofilepath = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/2140_raw/" + d17ofile
            d17ofilepath = "/Users/frio/EGRIP_2023/Data/2140_raw/" + d17ofile
            d17osplitfilepath = d17ofilepath.rpartition("/")
            d17ofilename = d17osplitfilepath[-1]
            d17odate = d17ofilename[9:17]
            if d17ofilename.startswith('raw'):
                d17odate = d17ofilename[12:20]
            if d17odate == d18odate:
                #read in data
                d17odata = np.loadtxt(d17ofilepath, skiprows = 1, dtype = "S")
                d17odata_dict = {}
                d17odata_dict["j_days"] = d17odata[:,4].astype("float")
                d17odata_dict["j_days"] = d17odata_dict["j_days"] * 60 * 60 * 24         # convert into seconds
                d17odata_dict["time_delay"] = np.diff(d17odata_dict["j_days"])
                d17odata_dict["index"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["epoch"] = d17odata[:,5].astype("float")[:-1]
                d17odata_dict["water_ppm"] = d17odata[:,14].astype("float")[:-1]
                d17odata_dict["2140_water_ppm"] = d17odata[:,14].astype("float")[:-1]
                d17odata_dict["d18o"] = d17odata[:,16].astype("float")[:-1] 
                d17odata_dict["2140d18o"] = d17odata_dict["d18o"]
                d17odata_dict["dD"] = d17odata[:,17].astype("float")[:-1] 
                d17odata_dict["2140dD"] = d17odata_dict["dD"]
                d17odata_dict["2140d17o"] = d17odata[:,15].astype("float")[:-1] 
                d17odata_dict["dexcess"] = d17odata_dict["dD"]-8*d17odata_dict["d18o"]
                d17odata_dict["start_depth"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["end_depth"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["true_depth"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["ec_value"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["valco_pos"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["comments"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["flag"] = np.arange(len(d17odata_dict["time_delay"]))
                d17odata_dict["flag1_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag1_d18o"][:] = '.' 
                d17odata_dict["flag1_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag1_dD"][:] = '.' 
                d17odata_dict["flag1_2140_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag1_2140_d18o"][:] = '.' 
                d17odata_dict["flag1_2140_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag1_2140_dD"][:] = '.' 
                d17odata_dict["flag1_2140_d17o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag1_2140_d17o"][:] = '.' 
                d17odata_dict["flag2"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag2"][:] = '.' 
                d17odata_dict["flag3"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag3"][:] = '.'  
                d17odata_dict["flag4_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag4_d18o"][:] = '.' 
                d17odata_dict["flag4_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag4_dD"][:] = '.' 
                d17odata_dict["flag4_2140_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag4_2140_d18o"][:] = '.' 
                d17odata_dict["flag4_2140_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag4_2140_dD"][:] = '.' 
                d17odata_dict["flag4_2140_d17o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag4_2140_d17o"][:] = '.' 
                d17odata_dict["flag5"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag5"][:] = '.' 
                d17odata_dict["flag6"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag6"][:] = '2140'
                d17odata_dict["flag7_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag7_d18o"][:] = '.' 
                d17odata_dict["flag7_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag7_dD"][:] = '.' 
                d17odata_dict["flag7_2140_d18o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag7_2140_d18o"][:] = '.' 
                d17odata_dict["flag7_2140_dD"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag7_2140_dD"][:] = '.' 
                d17odata_dict["flag7_2140_d17o"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag7_2140_d17o"][:] = '.' 
                d17odata_dict["flag8"] = deepcopy(d17odata_dict["flag"]).astype("S")
                d17odata_dict["flag8"][:] = '.' 

        #assign auxilary data to d17o data, by concatenating, sorting by epoch time and picking the nearest 2130 data for 2140
        raw_dict_both = {}
        for i in d17odata_dict.keys():
            raw_dict_both[i] = np.concatenate([d17odata_dict[i], data_dict[i]])
        sortrawindex = np.argsort(raw_dict_both["epoch"])
        sorted_raw_dict_both = {}
        for i in raw_dict_both.keys():
            sorted_raw_dict_both[i] = raw_dict_both[i][sortrawindex]
        sorted_raw_dict_both["index"] = np.arange(len(sorted_raw_dict_both["epoch"]))
        for n in sorted_raw_dict_both["index"][:-2]:
            #2130 data into 2140 dictionary
            if sorted_raw_dict_both["flag6"][n] == '2140' and sorted_raw_dict_both["flag6"][n+1] == '2130':
                sorted_raw_dict_both["start_depth"][n] = sorted_raw_dict_both["start_depth"][n+1]
                sorted_raw_dict_both["end_depth"][n] = sorted_raw_dict_both["end_depth"][n+1]
                sorted_raw_dict_both["true_depth"][n] = sorted_raw_dict_both["true_depth"][n+1]
                sorted_raw_dict_both["ec_value"][n] = sorted_raw_dict_both["ec_value"][n+1]
                sorted_raw_dict_both["valco_pos"][n] = sorted_raw_dict_both["valco_pos"][n+1]
                sorted_raw_dict_both["comments"][n] = sorted_raw_dict_both["comments"][n+1]
            if sorted_raw_dict_both["flag6"][n] == '2140' and sorted_raw_dict_both["flag6"][n+2] == '2130':
                sorted_raw_dict_both["start_depth"][n] = sorted_raw_dict_both["start_depth"][n+2]
                sorted_raw_dict_both["end_depth"][n] = sorted_raw_dict_both["end_depth"][n+2]
                sorted_raw_dict_both["true_depth"][n] = sorted_raw_dict_both["true_depth"][n+2]
                sorted_raw_dict_both["ec_value"][n] = sorted_raw_dict_both["ec_value"][n+2]
                sorted_raw_dict_both["valco_pos"][n] = sorted_raw_dict_both["valco_pos"][n+2]
                sorted_raw_dict_both["comments"][n] = sorted_raw_dict_both["comments"][n+2]
            #2140 data into 2130 dictionary
            if sorted_raw_dict_both["flag6"][n] == '2130' and sorted_raw_dict_both["flag6"][n+1] == '2140':
                sorted_raw_dict_both["2140d18o"][n] = sorted_raw_dict_both["2140d18o"][n+1]
                sorted_raw_dict_both["2140dD"][n] = sorted_raw_dict_both["2140dD"][n+1]
                sorted_raw_dict_both["2140d17o"][n] = sorted_raw_dict_both["2140d17o"][n+1]
                sorted_raw_dict_both["2140_water_ppm"][n] = sorted_raw_dict_both["2140_water_ppm"][n+1]
            if sorted_raw_dict_both["flag6"][n] == '2130' and sorted_raw_dict_both["flag6"][n+2] == '2140':
                sorted_raw_dict_both["2140d18o"][n] = sorted_raw_dict_both["2140d18o"][n+2]
                sorted_raw_dict_both["2140dD"][n] = sorted_raw_dict_both["2140dD"][n+2]
                sorted_raw_dict_both["2140d17o"][n] = sorted_raw_dict_both["2140d17o"][n+2]
                sorted_raw_dict_both["2140_water_ppm"][n] = sorted_raw_dict_both["2140_water_ppm"][n+2]
                
        data2140 = np.where(sorted_raw_dict_both["flag6"] == '2140')[0] 
        for i in sorted_raw_dict_both.keys():
            d17odata_dict[i] = sorted_raw_dict_both[i][data2140]
        d17odata_dict["index"] = np.arange(len(d17odata_dict["time_delay"]))
        
        data2130 = np.where(sorted_raw_dict_both["flag6"] == '2130')[0]
        for i in sorted_raw_dict_both.keys():
            data_dict[i] = sorted_raw_dict_both[i][data2130]
        data_dict["index"] = np.arange(len(data_dict["time_delay"]))

    ##### H20 CONCENTRATION FILTER PRIOR TO PLOTTING AND CALCULATIONS ##############
    ## flag set at "." for all good
    ## change first character to W if water concentration low
    h2obad = np.where(data_dict["water_ppm"]<= watertoolow)[0]
    for f in h2obad:
        data_dict["flag1_d18o"][f] = 'W'
        data_dict["flag1_dD"][f] = 'W'
    if corename == 'EGRIP' and d17oflag ==1:
        h2obad = np.where(data_dict["2140_water_ppm"]<= watertoolow)[0]
        for f in h2obad:
            data_dict["flag1_2140_d18o"][f] = 'W'
            data_dict["flag1_2140_dD"][f] = 'W'
            data_dict["flag1_2140_d17o"][f] = 'W'
		
    ##### PLOT RAW DATA ############################################################
    plt.close("all")
   
    ## PLOT DATA ###################################################################
    fig21 = plt.figure(21)
    fig21_ax1 = fig21.add_subplot(411)
    fig21_ax1.plot(data_dict["index"], data_dict["d18o"], "b-")
    fig21_ax1.set_ylabel("d18o")
    fig21_ax1.axis([0,data_dict["index"][-1],-60,0])
    fig21_ax2 = fig21.add_subplot(412)
    fig21_ax2.plot(data_dict["index"], data_dict["dD"], "r-")
    fig21_ax2.set_ylabel("dD")
    fig21_ax2.axis([0,data_dict["index"][-1],-500,0])
    fig21_ax3 = fig21.add_subplot(413)
    fig21_ax3.plot(data_dict["index"], data_dict["water_ppm"], "k-")
    fig21_ax3.set_ylabel("water ppm")
    fig21_ax4 = fig21.add_subplot(414)
    fig21_ax4.plot(data_dict["index"], data_dict["ec_value"], "g-")
    fig21_ax4.set_ylabel("EC")
    fig21_ax4.set_xlabel("Index")
    fig21_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))

    fig23 = plt.figure(23)
    fig23_ax1 = fig23.add_subplot(411)
    fig23_ax1.plot(data_dict["index"], data_dict["valco_pos"], "b-")
    fig23_ax1.set_ylabel("valco position")
    fig23_ax2 = fig23.add_subplot(412)
    fig23_ax2.plot(data_dict["index"], data_dict["comments"], "r-")
    fig23_ax2.set_ylabel("comments")
    fig23_ax3 = fig23.add_subplot(413)
    fig23_ax3.plot(data_dict["index"], data_dict["loadflag"], "g-")
    fig23_ax3.set_ylabel("load flag")
    fig23_ax4 = fig23.add_subplot(414)
    fig23_ax4.plot(data_dict["index"], data_dict["true_depth"], "r-")
    fig23_ax4.set_ylabel("true depth")
    fig23_ax3.set_xlabel("Index")
    fig23_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    if corename == 'EGRIP' and d17oflag == 1:
        fig24 = plt.figure(24)
        fig24_ax1 = fig24.add_subplot(511)
        fig24_ax1.plot(data_dict["index"], data_dict["2140d18o"], "b-")
        fig24_ax1.set_ylabel("2140d18o")
        fig24_ax1.axis([0,data_dict["index"][-1],-60,0])
        fig24_ax2 = fig24.add_subplot(512)
        fig24_ax2.plot(data_dict["index"], data_dict["2140dD"], "r-")
        fig24_ax2.set_ylabel("2140dD")
        fig24_ax2.axis([0,data_dict["index"][-1],-500,0])
        fig24_ax3 = fig24.add_subplot(513)
        fig24_ax3.plot(data_dict["index"], data_dict["2140d17o"], "m-")
        fig24_ax3.set_ylabel("2140d17o")
        fig24_ax3.axis([0,data_dict["index"][-1],-25,0])
        fig24_ax4 = fig24.add_subplot(514)
        fig24_ax4.plot(data_dict["index"], data_dict["loadflag"], "k-")
        fig24_ax4.set_ylabel("load flag")
        fig24_ax5 = fig24.add_subplot(515)
        fig24_ax5.plot(data_dict["index"], data_dict["laser_distance"], "g-")
        fig24_ax5.set_ylabel("laser_distance")
        fig24_ax5.set_xlabel("Index")
        fig24_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))

    ##### BEGINING OF ALLAN FUNCTION ###############################################
    ## AM ALLAN VARIANCE FROM MIDNIGHT TO 1 HOUR BEFORE AM VALCO TO AVOID H20 DROP OUT IN INITIATION
    amallanbegin = 100
    amvalco = np.where(data_dict["comments"]==142)[0]  
    amallanaved18o = 0
    amallanstdevd18o = 0
    amallan10secd18o = 0
    amallan60secd18o = 0
    amallan600secd18o = 0
    amallan3600secd18o = 0
    amallanavedD = 0
    amallanstdevdD = 0
    amallan10secdD = 0
    amallan60secdD = 0
    amallan600secdD = 0
    amallan3600secdD = 0
    amallanave2140d18o = 0
    amallanstdev2140d18o = 0
    amallan10sec2140d18o = 0
    amallan60sec2140d18o = 0
    amallan600sec2140d18o = 0
    amallan3600sec2140d18o = 0
    amallanave2140dD = 0
    amallanstdev2140dD = 0
    amallan10sec2140dD = 0
    amallan60sec2140dD = 0
    amallan600sec2140dD = 0
    amallan3600sec2140dD = 0
    amallanave2140d17o = 0
    amallanstdev2140d17o = 0
    amallan10sec2140d17o = 0
    amallan60sec2140d17o = 0
    amallan600sec2140d17o = 0
    amallan3600sec2140d17o = 0
    amallanh2o = 0
    allansecs = data_dict["j_days"]
    epoch = data_dict["epoch"]
    water = data_dict["water_ppm"]
    index = data_dict["index"]
    comments = data_dict["comments"]
    if len(amvalco)!=0 and amvalco[0]>4200:
        amallanend = amvalco[0]-4000
        if len(h2obad)!=0:
            if h2obad[0] <= amallanend:
                amallanend = h2obad[0]-100
        if amallanend-amallanbegin > 3000:
            name = "d18o"
            flag1 = data_dict["flag1_d18o"]
            amrun_allan = perform_allan(name, data_dict["d18o"], amallanbegin, amallanend, allansecs, flag1, water) 
            amallanaved18o = amrun_allan[0]
            amallanstdevd18o = amrun_allan[1]
            amallan10secd18o = amrun_allan[2]
            amallan60secd18o = amrun_allan[3]
            amallan600secd18o = amrun_allan[4]
            amallan3600secd18o = amrun_allan[5]
            amallanh2o = amrun_allan[6]
            name = "dD"
            flag1 = data_dict["flag1_dD"]
            amrun_allan = perform_allan(name, data_dict["dD"], amallanbegin, amallanend, allansecs, flag1, water)
            amallanavedD = amrun_allan[0]
            amallanstdevdD = amrun_allan[1]
            amallan10secdD = amrun_allan[2]
            amallan60secdD = amrun_allan[3]
            amallan600secdD = amrun_allan[4]
            amallan3600secdD = amrun_allan[5]
            amallanh2o = amrun_allan[6]
            if corename == 'EGRIP' and d17oflag == 1:
                name = "2140d18o"
                flag1 = data_dict["flag1_2140_d18o"]
                amallanbegin = data2140[0]+100
                amrun_allan = perform_allan(name, data_dict["2140d18o"], amallanbegin, amallanend, allansecs, flag1, water) 
                amallanave2140d18o = amrun_allan[0]
                amallanstdev2140d18o = amrun_allan[1]
                amallan10sec2140d18o = amrun_allan[2]
                amallan60sec2140d18o = amrun_allan[3]
                amallan600sec2140d18o = amrun_allan[4]
                amallan3600sec2140d18o = amrun_allan[5]
                amallanh2o2140 = amrun_allan[6]
                name = "2140dD"
                flag1 = data_dict["flag1_2140_dD"]
                amrun_allan = perform_allan(name, data_dict["2140dD"], amallanbegin, amallanend, allansecs, flag1, water)
                amallanave2140dD = amrun_allan[0]
                amallanstdev2140dD = amrun_allan[1]
                amallan10sec2140dD = amrun_allan[2]
                amallan60sec2140dD = amrun_allan[3]
                amallan600sec2140dD = amrun_allan[4]
                amallan3600sec2140dD = amrun_allan[5]
                amallanh2o2140 = amrun_allan[6]
                name = "2140d17o"
                flag1 = data_dict["flag1_2140_d17o"]
                amrun_allan = perform_allan(name, data_dict["2140d17o"], amallanbegin, amallanend, allansecs, flag1, water)
                amallanave2140d17o = amrun_allan[0]
                amallanstdev2140d17o = amrun_allan[1]
                amallan10sec2140d17o = amrun_allan[2]
                amallan60sec2140d17o = amrun_allan[3]
                amallan600sec2140d17o = amrun_allan[4]
                amallan3600sec2140d17o = amrun_allan[5]
                amallanh2o2140 = amrun_allan[6]
    ## PM ALLAN VARIANCE FROM 10 MIN AFTER END OF PM VALCO (VALCO_POS==3), TO END OF FILE
    pmvalcoend = np.where(data_dict["comments"]==172)[0] 
    pmallanaved18o = 0
    pmallanstdevd18o = 0
    pmallan10secd18o = 0
    pmallan60secd18o = 0
    pmallan600secd18o = 0
    pmallan3600secd18o = 0
    pmallanavedD = 0
    pmallanstdevdD = 0
    pmallan10secdD = 0
    pmallan60secdD = 0
    pmallan600secdD = 0
    pmallan3600secdD = 0
    pmallanave2140d18o = 0
    pmallanstdev2140d18o = 0
    pmallan10sec2140d18o = 0
    pmallan60sec2140d18o = 0
    pmallan600sec2140d18o = 0
    pmallan3600sec2140d18o = 0
    pmallanave2140dD = 0
    pmallanstdev2140dD = 0
    pmallan10sec2140dD = 0
    pmallan60sec2140dD = 0
    pmallan600sec2140dD = 0
    pmallan3600sec2140dD = 0
    pmallanave2140d17o = 0
    pmallanstdev2140d17o = 0
    pmallan10sec2140d17o = 0
    pmallan60sec2140d17o = 0
    pmallan600sec2140d17o = 0
    pmallan3600sec2140d17o = 0
    pmallanh2o = 0
    if len(pmvalcoend)!=0:
        pmallanbegin = pmvalcoend[-1]+2000 #was 1000
        pmallanend = data_dict["index"][-1]
        if len(h2obad)!=0:
            if h2obad[-1] >= pmallanbegin:
                pmallanbegin = h2obad[-1]+100
        if pmallanbegin <= (pmallanend-3000):
            name = "d18o"
            flag1 = data_dict["flag1_d18o"]
            print pmallanbegin, pmallanend
            pmrun_allan = perform_allan(name, data_dict["d18o"], pmallanbegin, pmallanend, allansecs, flag1, water)
            pmallanaved18o = pmrun_allan[0]
            pmallanstdevd18o = pmrun_allan[1]
            pmallan10secd18o = pmrun_allan[2]
            pmallan60secd18o = pmrun_allan[3]
            pmallan600secd18o = pmrun_allan[4]
            pmallan3600secd18o = pmrun_allan[5]
            pmallanh2o = pmrun_allan[6]
            name = "dD"
            flag1 = data_dict["flag1_dD"]
            pmrun_allan = perform_allan(name, data_dict["dD"], pmallanbegin, pmallanend, allansecs, flag1, water)
            pmallanavedD = pmrun_allan[0]
            pmallanstdevdD = pmrun_allan[1]
            pmallan10secdD = pmrun_allan[2]
            pmallan60secdD = pmrun_allan[3]
            pmallan600secdD = pmrun_allan[4]
            pmallan3600secdD = pmrun_allan[5]
            pmallanh2o = pmrun_allan[6]
            if corename == 'EGRIP' and d17oflag == 1:
                name = "2140d18o"
                flag1 = data_dict["flag1_2140_d18o"]
                nonzero = np.where(data_dict["2140d18o"]!=0)[0]
                pmallanend = nonzero[-1]
                pmrun_allan = perform_allan(name, data_dict["2140d18o"], pmallanbegin, pmallanend, allansecs, flag1, water)
                pmallanave2140d18o = pmrun_allan[0]
                pmallanstdev2140d18o = pmrun_allan[1]
                pmallan10sec2140d18o = pmrun_allan[2]
                pmallan60sec2140d18o = pmrun_allan[3]
                pmallan600sec2140d18o = pmrun_allan[4]
                pmallan3600sec2140d18o = pmrun_allan[5]
                pmallanh2o2140 = pmrun_allan[6]
                name = "2140dD"
                flag1 = data_dict["flag1_2140_dD"]
                pmrun_allan = perform_allan(name, data_dict["2140dD"], pmallanbegin, pmallanend, allansecs, flag1, water)
                pmallanave2140dD = pmrun_allan[0]
                pmallanstdev2140dD = pmrun_allan[1]
                pmallan10sec2140dD = pmrun_allan[2]
                pmallan60sec2140dD = pmrun_allan[3]
                pmallan600sec2140dD = pmrun_allan[4]
                pmallan3600sec2140dD = pmrun_allan[5]
                pmallanh2o2140 = pmrun_allan[6]
                name = "2140d17o"
                flag1 = data_dict["flag1_2140_d17o"]
                pmrun_allan = perform_allan(name, data_dict["2140d17o"], pmallanbegin, pmallanend, allansecs, flag1, water)
                pmallanave2140d17o = pmrun_allan[0]
                pmallanstdev2140d17o = pmrun_allan[1]
                pmallan10sec2140d17o = pmrun_allan[2]
                pmallan60sec2140d17o = pmrun_allan[3]
                pmallan600sec2140d17o = pmrun_allan[4]
                pmallan3600sec2140d17o = pmrun_allan[5]
                pmallanh2o2140 = pmrun_allan[6]
    ##### END ALLAN FUNCTION########################################################
    
    ##### check lengths of arrays in data_dict before proceeding:
    lengthcomp = len(data_dict["d18o"])
    for i in data_dict.keys():
        if len(data_dict[i]) == lengthcomp+1:
            data_dict[i] = data_dict[i][:-1]
            print "changed length of ", i
            
    ##### Stop and ask if need to edit?
    if verbose == 1:   
        checkcomments = raw_input("Do you want to edit any of the valco_pos?")
        if checkcomments in ('y', 'ye', 'yes'):
            commentstartindex = input("Please type in start index...")
            commentendindex = input("Please type in end index...")
            newcomment = input("Please type in new valco_pos number...")
            data_dict["valco_pos"][commentstartindex:commentendindex] = newcomment
        checkstartdepth = raw_input("Do you want to edit the beginning core depths?")
        if checkstartdepth in ('y', 'ye', 'yes'):
            wrong_depth = input("Please type in wrong depth...")
            right_depth = input("Please type in right depth...")
            fixtime = input("Please type in epoch time just prior to correction...")
            fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                data_dict["epoch"][x]>=fixtime]
            for x in fixdepth:
                data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
            fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "c-")
            checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
            if checkstartdepth in ('y', 'ye', 'yes'):
                wrong_depth = input("Please type in wrong depth...")
                right_depth = input("Please type in right depth...")
                fixtime = input("Please type in epoch time just prior to correction...")
                fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                    data_dict["epoch"][x]>=fixtime]
                for x in fixdepth:
                    data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                    data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "b-")
                checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
                if checkstartdepth in ('y', 'ye', 'yes'):
                    wrong_depth = input("Please type in wrong depth...")
                    right_depth = input("Please type in right depth...")
                    fixtime = input("Please type in epoch time just prior to correction...")
                    fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                        data_dict["epoch"][x]>=fixtime]
                    for x in fixdepth:
                        data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                        data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                    fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "k-")
                    checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
                    if checkstartdepth in ('y', 'ye', 'yes'):
                        wrong_depth = input("Please type in wrong depth...")
                        right_depth = input("Please type in right depth...")
                        fixtime = input("Please type in epoch time just prior to correction...")
                        fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                            data_dict["epoch"][x]>=fixtime]
                        for x in fixdepth:
                            data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                            data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                        fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "w-")
                        checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
                        if checkstartdepth in ('y', 'ye', 'yes'):
                            wrong_depth = input("Please type in wrong depth...")
                            right_depth = input("Please type in right depth...")
                            fixtime = input("Please type in epoch time just prior to correction...")
                            fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                                data_dict["epoch"][x]>=fixtime]
                            for x in fixdepth:
                                data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                                data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                            fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "w-")
                            checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
                            if checkstartdepth in ('y', 'ye', 'yes'):
                                wrong_depth = input("Please type in wrong depth...")
                                right_depth = input("Please type in right depth...")
                                fixtime = input("Please type in epoch time just prior to correction...")
                                fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                                    data_dict["epoch"][x]>=fixtime]
                                for x in fixdepth:
                                    data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                                    data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                                fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "w-")
                                checkstartdepth = raw_input("Do you want to edit more of the beginning core depths?")
                                if checkstartdepth in ('y', 'ye', 'yes'):
                                    wrong_depth = input("Please type in wrong depth...")
                                    right_depth = input("Please type in right depth...")
                                    fixtime = input("Please type in epoch time just prior to correction...")
                                    fixdepth = [x for x in data_dict["index"][:] if data_dict["start_depth"][x]==wrong_depth and \
                                        data_dict["epoch"][x]>=fixtime]
                                    for x in fixdepth:
                                        data_dict["true_depth"][x] = data_dict["true_depth"][x] - wrong_depth + right_depth
                                        data_dict["start_depth"][x] = data_dict["start_depth"][x] - wrong_depth + right_depth
                                    fig23_ax3.plot(data_dict["index"], data_dict["start_depth"], "w-")
        checkenddepth = raw_input("Do you want to edit the ending core depths?")
        if checkenddepth in ('y', 'ye', 'yes'):
            wrong_depth = input("Please type in wrong depth...")
            right_depth = input("Please type in right depth...")
            fixtime = input("Please type in epoch time just prior to correction...")
            fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                data_dict["epoch"][x]>=fixtime]
            for x in fixdepth:
                data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
            fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "m-")
            checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
            if checkenddepth in ('y', 'ye', 'yes'):
                wrong_depth = input("Please type in wrong depth...")
                right_depth = input("Please type in right depth...")
                fixtime = input("Please type in epoch time just prior to correction...")
                fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                    data_dict["epoch"][x]>=fixtime]
                for x in fixdepth:
                    data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "r-")
                checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                if checkenddepth in ('y', 'ye', 'yes'):
                    wrong_depth = input("Please type in wrong depth...")
                    right_depth = input("Please type in right depth...")
                    fixtime = input("Please type in epoch time just prior to correction...")
                    fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                        data_dict["epoch"][x]>=fixtime]
                    for x in fixdepth:
                        data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                    fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "y-")
                    checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                    if checkenddepth in ('y', 'ye', 'yes'):
                        wrong_depth = input("Please type in wrong depth...")
                        right_depth = input("Please type in right depth...")
                        fixtime = input("Please type in epoch time just prior to correction...")
                        fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                            data_dict["epoch"][x]>=fixtime]
                        for x in fixdepth:
                            data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                        fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "g-")
                        checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                        if checkenddepth in ('y', 'ye', 'yes'):
                            wrong_depth = input("Please type in wrong depth...")
                            right_depth = input("Please type in right depth...")
                            fixtime = input("Please type in epoch time just prior to correction...")
                            fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                                data_dict["epoch"][x]>=fixtime]
                            for x in fixdepth:
                                data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                            fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "g-")
                            checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                            if checkenddepth in ('y', 'ye', 'yes'):
                                wrong_depth = input("Please type in wrong depth...")
                                right_depth = input("Please type in right depth...")
                                fixtime = input("Please type in epoch time just prior to correction...")
                                fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                                    data_dict["epoch"][x]>=fixtime]
                                for x in fixdepth:
                                    data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                                fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "g-")
                                checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                                if checkenddepth in ('y', 'ye', 'yes'):
                                    wrong_depth = input("Please type in wrong depth...")
                                    right_depth = input("Please type in right depth...")
                                    fixtime = input("Please type in epoch time just prior to correction...")
                                    fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                                        data_dict["epoch"][x]>=fixtime]
                                    for x in fixdepth:
                                        data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                                    fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "g-")
                                    checkenddepth = raw_input("Do you want to edit more of the ending core depths?")
                                    if checkenddepth in ('y', 'ye', 'yes'):
                                        wrong_depth = input("Please type in wrong depth...")
                                        right_depth = input("Please type in right depth...")
                                        fixtime = input("Please type in epoch time just prior to correction...")
                                        fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                                            data_dict["epoch"][x]>=fixtime]
                                        for x in fixdepth:
                                            data_dict["end_depth"][x] = data_dict["end_depth"][x] - wrong_depth + right_depth
                                        fig23_ax3.plot(data_dict["index"], data_dict["end_depth"], "g-")
        checkcomments = raw_input("Do you want to edit any of the comments?")
        if checkcomments in ('y', 'ye', 'yes'):
            commentstartindex = input("Please type in start index...")
            commentendindex = input("Please type in end index...")
            newcomment = input("Please type in new comment number...")
            data_dict["comments"][commentstartindex:commentendindex] = newcomment
            checkcomments = raw_input("Do you want to edit any more of the comments?")
            if checkcomments in ('y', 'ye', 'yes'):
                commentstartindex = input("Please type in start index...")
                commentendindex = input("Please type in end index...")
                newcomment = input("Please type in new comment number...")
                data_dict["comments"][commentstartindex:commentendindex] = newcomment
                checkcomments = raw_input("Do you want to edit any more of the comments?")
                if checkcomments in ('y', 'ye', 'yes'):
                    commentstartindex = input("Please type in start index...")
                    commentendindex = input("Please type in end index...")
                    newcomment = input("Please type in new comment number...")
                    data_dict["comments"][commentstartindex:commentendindex] = newcomment
                    checkcomments = raw_input("Do you want to edit any more of the comments?")
                    if checkcomments in ('y', 'ye', 'yes'):
                        commentstartindex = input("Please type in start index...")
                        commentendindex = input("Please type in end index...")
                        newcomment = input("Please type in new comment number...")
                        data_dict["comments"][commentstartindex:commentendindex] = newcomment
                        checkcomments = raw_input("Do you want to edit any more of the comments?")
                        if checkcomments in ('y', 'ye', 'yes'):
                            commentstartindex = input("Please type in start index...")
                            commentendindex = input("Please type in end index...")
                            newcomment = input("Please type in new comment number...")
                            data_dict["comments"][commentstartindex:commentendindex] = newcomment
                            checkcomments = raw_input("Do you want to edit any more of the comments?")
                            if checkcomments in ('y', 'ye', 'yes'):
                                commentstartindex = input("Please type in start index...")
                                commentendindex = input("Please type in end index...")
                                newcomment = input("Please type in new comment number...")
                                data_dict["comments"][commentstartindex:commentendindex] = newcomment
            
    ##### NEW DEPTH CALCULATION FOR TOWER, STACKING ICE CORES
    # maybe to be done after ID of ice cores?
    # ID when load flag changed from 1 to 2 as startfixdepth
    #if filepath.startswith("/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/raw/HIDS"): # so only do this operation on the raw files, not one the raw dictionaries
    if filepath.startswith("/Users/frio/EGRIP_2023/Data/raw/HIDS"):
        checkloadflag = raw_input("Do you want to edit any of the loadflags?")
        if checkloadflag in ('y', 'ye', 'yes'):
            flagstartindex = input("Please type in start index...")
            flagendindex = input("Please type in end index...")
            newloadflag = input("Please type in new loadflag number...")
            data_dict["loadflag"][flagstartindex:flagendindex] = newloadflag
#            fig24_ax4.plot(data_dict["index"], data_dict["loadflag"], "r-")
            checkloadflag = raw_input("Do you want to edit any more of the loadflags?")
            if checkloadflag in ('y', 'ye', 'yes'):
                flagstartindex = input("Please type in start index...")
                flagendindex = input("Please type in end index...")
                newloadflag = input("Please type in new loadflag number...")
                data_dict["loadflag"][flagstartindex:flagendindex] = newloadflag
#                fig24_ax4.plot(data_dict["index"], data_dict["loadflag"], "m-")
                checkloadflag = raw_input("Do you want to edit any more of the loadflags?")
                if checkloadflag in ('y', 'ye', 'yes'):
                    flagstartindex = input("Please type in start index...")
                    flagendindex = input("Please type in end index...")
                    newloadflag = input("Please type in new loadflag number...")
                    data_dict["loadflag"][flagstartindex:flagendindex] = newloadflag
#                    fig24_ax4.plot(data_dict["index"], data_dict["loadflag"], "g-")
                    checkloadflag = raw_input("Do you want to edit any more of the loadflags?")
                    if checkloadflag in ('y', 'ye', 'yes'):
                        flagstartindex = input("Please type in start index...")
                        flagendindex = input("Please type in end index...")
                        newloadflag = input("Please type in new loadflag number...")
                        data_dict["loadflag"][flagstartindex:flagendindex] = newloadflag
#                        fig24_ax4.plot(data_dict["index"], data_dict["loadflag"], "k-")
        startfixdepth = [x for x in index[1:-1] if data_dict["loadflag"][x-1]!=2 and data_dict["loadflag"][x]==2]
        print "start of depth fix", startfixdepth
        numdepthfix = np.arange(len(startfixdepth))
        print "number of fixes", numdepthfix
        # ID when load flag changed from 2 to 0 as endfixdepth
        endfixdepth = [x for x in index[1:-1] if data_dict["loadflag"][x]==2 and data_dict["loadflag"][x+1]!=2]
        print "end of depth fix", endfixdepth
        calcoffset = deepcopy(endfixdepth)
        depthoffset = deepcopy(endfixdepth)
#        for i in numdepthfix[:]:
#            calcoffset[i] = endfixdepth[i]-3
#            depthoffset[i] = data_dict["end_depth"][calcoffset[i]]-data_dict["true_depth"][calcoffset[i]]
        # calculate offset from the last value before end of fixdepth to enddepth value
#        depthoffset = data_dict["end_depth"][calcoffset]-data_dict["true_depth"][calcoffset]
#        print "offset", depthoffset
        # add offset for all of flag 2 data, [startixdepth:endfixdepth]
        for i in numdepthfix:
            calcoffset[i] = endfixdepth[i]-6
            depthoffset[i] = data_dict["end_depth"][calcoffset[i]]-data_dict["true_depth"][calcoffset[i]]
            for d in index[startfixdepth[i]:endfixdepth[i]-6]:
                print "i = ", i, " , d = ", d, " , true depth = ", data_dict["true_depth"][d], " , end depth = ", data_dict["end_depth"][d], ", offset = ", depthoffset[i], ", index = ", calcoffset[i]
                data_dict["true_depth"][d]=data_dict["true_depth"][d]+depthoffset[i]
                data_dict["laser_distance"][d]=data_dict["laser_distance"][d]+(depthoffset[i]*1000)

    ##### PERFORM DIFF FUNCTION (first derivative approximation) ON ISOTOPES AND EC 
    ## PRECEEDED BY SMOOTHING
    smoothd18o = smooth(data_dict["d18o"])
    smoothdD = smooth(data_dict["dD"])
    smooth2140d18o = smooth(data_dict["2140d18o"])
    smooth2140dD = smooth(data_dict["2140dD"])
    smooth2140d17o = smooth(data_dict["2140d17o"])
    depth_cm = data_dict["true_depth"]*100  # true_depth in meters*100 = depth in cm
    smooth_depth = smooth(depth_cm)
    smooth_ec = smooth(data_dict["ec_value"])
    smooth_video_rate = smooth(data_dict["MeltRate"])
    minutes = data_dict["j_days"]/60  # j_days in sec/60 = time in minutes
    smooth_time =smooth(minutes)

    diffindex = data_dict["index"][:-1] 
    diffd18o = np.diff(smoothd18o)
    diffdD = np.diff(smoothdD)
    diff2140d18o = np.diff(smooth2140d18o)
    diff2140dD = np.diff(smooth2140dD)
    diff2140d17o = np.diff(smooth2140d17o)
    diff_depth = np.diff(smooth_depth)
    diffec = np.diff(smooth_ec)
    diff_laser = np.diff(data_dict["laser_distance"])
    diff_start_depth = np.diff(data_dict["start_depth"])
    diff_time = np.diff(smooth_time)
    meltrate = diff_depth/diff_time
    smooth_meltrate = smooth(meltrate)      

    fig25 = plt.figure(25) 
    fig25_ax1 = fig25.add_subplot(411)
    fig25_ax1.plot(diffindex, diffd18o, "b-")
    fig25_ax1.set_ylabel("diffd18o")
    fig25_ax2 = fig25.add_subplot(412)
    fig25_ax2.plot(diffindex, diffdD, "r-")
    fig25_ax2.set_ylabel("diffdD")
    fig25_ax3 = fig25.add_subplot(413)
    fig25_ax3.plot(diffindex, diff_depth, "k-")
    fig25_ax3.set_ylabel("diff_depth")
    fig25_ax4 = fig25.add_subplot(414)
    fig25_ax4.plot(diffindex, meltrate, "k-", diffindex, smooth_meltrate, "b-",diffindex, smooth_video_rate[:-1], "-m")
    fig25_ax4.axis([0,diffindex[-1],-5,15])
    fig25_ax4.set_ylabel("meltrate")
    fig25_ax4.set_xlabel("Index")
    fig25_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    fig26 = plt.figure(26) 
    fig26_ax1 = fig26.add_subplot(411)
    fig26_ax1.plot(diffindex, diff2140d18o, "b-")
    fig26_ax1.set_ylabel("diff2140d18o")
    fig26_ax2 = fig26.add_subplot(412)
    fig26_ax2.plot(diffindex, diff2140dD, "r-")
    fig26_ax2.set_ylabel("diff2140dD")
    fig26_ax3 = fig26.add_subplot(413)
    fig26_ax3.plot(diffindex, diff2140d17o, "m-")
    fig26_ax3.set_ylabel("diff2140d17o")
    fig26_ax4 = fig26.add_subplot(414)
    fig26_ax4.plot(diffindex, diff_depth, "k-")
    fig26_ax4.axis([0,diffindex[-1],-5,15])
    fig26_ax4.set_ylabel("diff_depth")
    fig26_ax4.set_xlabel("Index")
    fig26_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))

    ##### VALCO IDENTIFICATION #####################################################
    ##### Look at comments to locate amvalco and pmvalco
    amvalcobegin = [x for x in index[1:] if comments[x-1]!=142 and comments[x]==142]
    print "AM valco begin " ,amvalcobegin
    pmvalcobegin = [x for x in index[1:] if comments[x-1]!=172 and comments[x]==172]
    print "PM valco begin ", pmvalcobegin

    if len(amvalcobegin) > 1:
        amvalcobegin = [amvalcobegin[-1]]
    if len(amvalcobegin) < 1:
        amvalcobegin = [pmvalcobegin[0]]
    if len(pmvalcobegin) > 1:
        pmvalcobegin = [pmvalcobegin[-1]]
    if len(pmvalcobegin) < 1:
        pmvalcobegin = [amvalcobegin[-1]]
    starttime = epoch[amvalcobegin]
    endtime = epoch[pmvalcobegin]
    fullmelttime = ((endtime-starttime)/60)/60
    
    if verbose ==1:
        print "AM valco begin " ,amvalcobegin
        print "PM valco begin ", pmvalcobegin
        print "Full melt time (hours)", fullmelttime

    #### Stop and ask if need to edit?
    if verbose ==1:
        checkbegin = raw_input("Do you want to edit the beginning am valco indices?")
        if checkbegin in ('y', 'ye', 'yes'):
            amvalcobegin = input("Please type new indices list in [ ]...")
            print "New AM valco begin ", amvalcobegin
        checkbegin = raw_input("Do you want to edit the beginning pm valco indices for?")
        if checkbegin in ('y', 'ye', 'yes'):
            pmvalcobegin = input("Please type new indices list in [ ]...")
            print "New PM valco begin ", pmvalcobegin
            
    index = data_dict["index"]
    comments = data_dict["comments"]
    epoch = data_dict["epoch"]
    valco_pos = data_dict["valco_pos"]
    name = "2130d18o"
    isotope = data_dict["d18o"]
    diffisotope = diffd18o
    valcoid2130d18o = valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope)
    name = "2130dD"
    isotope = data_dict["dD"]
    diffisotope = diffdD
    valcoid2130dD = valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope)
    fullmelttime = valcoid2130d18o [4]
    if corename == 'EGRIP' and d17oflag ==1: 
        name = "2140d18o"
        isotope = data_dict["2140d18o"]
        diffisotope = diff2140d18o
        valcoid2140d18o = valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope)
        name = "2140dD"
        isotope = data_dict["2140dD"]
        diffisotope = diff2140dD
        valcoid2140dD = valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope)
        name = "2140d17o"
        isotope = data_dict["2140d17o"]
        diffisotope = diff2140d17o
        valcoid2140d17o = valcoid(name, isotope, index, comments, epoch, valco_pos, diffisotope)

    ##### PUT IN TRANSFER FUNCTION CALL FOR EACH OF THE VALCO TRANSITIONS LISTED ABOVE, then Apply Memory correction to all valco transitions ##########################
    ## Start by replicating full isotope arrays
    valcomemd18o = deepcopy(data_dict["d18o"])
    valcomemdD = deepcopy(data_dict["dD"])
    valcomem2140d18o = deepcopy(data_dict["2140d18o"])
    valcomem2140dD = deepcopy(data_dict["2140dD"])
    valcomem2140d17o = deepcopy(data_dict["2140d17o"])
    secs = np.cumsum(data_dict["time_delay"])
    count = 1
    counter = 1
    
    ### Call valco function for each method and isotope for each standard
    if data_dict["epoch"][0] < 1325376000:  # before 20120101, WAIS06A, 1102 with longer sampling time in between each data point, 5 minute valco
        print "Data processing 1102 data and 5 min valco data"
        for m in data_dict["index"]:
            data_dict["flag5"][m] = 'A'
        valcopreindex = 50
        valcopostindex = 100
        valcoprelength = 50
        valcopostlength = 100
        badh2o = 50
        
    if data_dict["epoch"][0] < 1354320001 and data_dict["epoch"][0] > 1325376000:  # before 20121201, WAIS06A, 2130, 5 minute valco and 1,5,1 for drift, but after switch to 2130
        print "Data processing 2130 and 5 min valco data"
        for m in data_dict["index"]:
            data_dict["flag5"][m] = 'B'
        valcopreindex = 215 #was 250 but catching too much
        valcopostindex = 215 #was 250 but catching too much
        valcoprelength = 150
        valcopostlength = 250
        badh2o = 200
        
    if data_dict["epoch"][0] > 1354320001 and data_dict["epoch"][0] < 1420095599:  # after 20121201, WAIS06A, 20 min valco, and idle water as drift OR push waters inbetween cores
        print "Data processing 2130 and 20 min valco data"
        for m in data_dict["index"]:
            data_dict["flag5"][m] = 'C'
        valcopreindex = 500
        valcopostindex = 500
        valcoprelength = 400
        valcopostlength = 500
        badh2o = 200
        
    if data_dict["epoch"][0] > 1420095600 and data_dict["epoch"][0] < 1466694499:  # after 20150101, SPIce Core, 20 min valco, and idle water as drift OR push waters inbetween cores  
        print "Data processing 2130 and 20 min valco data"
        for m in data_dict["index"]:
            data_dict["flag5"][m] = 'C'
        valcopreindex = 500
        valcopostindex = 500
        valcoprelength = 400
        valcopostlength = 500
        badh2o = 200
        
    if data_dict["epoch"][0] > 1466694499:  # after 20150101, SPIce Core, 20 min valco, and push ice inbetween cores and different standards, and 2140
        print "Data processing 2130 and 2140 and 20 min valco data"
        for m in data_dict["index"]:
            data_dict["flag5"][m] = 'D'
        valcopreindex = 500
        valcopostindex = 650 #20231127, adjusting number from 500 to 650 to see if this fixes code issue where it started bombing out during valco correction
        valcoprelength = 400
        valcopostlength = 500
        badh2o = 200
        
    index = data_dict["index"]
    name = "d18o"
    flag1 = data_dict["flag1_d18o"]
    isotope = data_dict["d18o"]
    amtransbegin2130d18o = valcoid2130d18o[0]
    amtransindex2130d18o = valcoid2130d18o[1]
    pmtransbegin2130d18o = valcoid2130d18o[2]
    pmtransindex2130d18o = valcoid2130d18o[3]
    valcofunctd18o = valcofunct(name, isotope, index, amtransbegin2130d18o, amtransindex2130d18o, pmtransbegin2130d18o, pmtransindex2130d18o, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1)
    # raw isotopes 0-7, extrapolated istopes 11-18
    smavevalcod18o = valcofunctd18o[8]
    valco_skewsigma_d18o = valcofunctd18o[9]
    valco_normsigma_d18o = valcofunctd18o[10]
    extrapolatedd18o = valcofunctd18o[11]
    name = "dD"
    flag1 = data_dict["flag1_dD"]
    isotope = data_dict["dD"]
    amtransbegin2130dD = valcoid2130dD[0]
    amtransindex2130dD = valcoid2130dD[1]
    pmtransbegin2130dD = valcoid2130dD[2]
    pmtransindex2130dD = valcoid2130dD[3]
    valcofunctdD = valcofunct(name, isotope, index, amtransbegin2130dD, amtransindex2130dD, pmtransbegin2130dD, pmtransindex2130dD, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1)
    # raw isotopes 0-7, extrapolated istopes 11-18
    smavevalcodD = valcofunctdD[8]
    valco_skewsigma_dD = valcofunctdD[9]
    valco_normsigma_dD = valcofunctdD[10]
    extrapolateddD = valcofunctdD[11]
    if corename == 'EGRIP' and d17oflag == 1:   # after 20160101, SPIce Core with 2140, 20 min valco, and idle water as drift OR push waters inbetween cores and different standards, and d17o
        name = "2140d18o"
        flag1 = data_dict["flag1_2140_d18o"]
        isotope = data_dict["2140d18o"]
        amtransbegin2140d18o = valcoid2140d18o[0]
        amtransindex2140d18o = valcoid2140d18o[1]
        pmtransbegin2140d18o = valcoid2140d18o[2]
        pmtransindex2140d18o = valcoid2140d18o[3]
        valcofunct2140d18o = valcofunct(name, isotope, index, amtransbegin2140d18o, amtransindex2140d18o, pmtransbegin2140d18o, pmtransindex2140d18o, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1)
        smavevalco2140d18o = valcofunct2140d18o[8]
        valco_skewsigma_2140d18o = valcofunct2140d18o[9]
        valco_normsigma_2140d18o = valcofunct2140d18o[10]
        extrapolated2140d18o = valcofunct2140d18o[11]
        name = "2140dD"
        flag1 = data_dict["flag1_2140_dD"]
        isotope = data_dict["2140dD"]
        amtransbegin2140dD = valcoid2140dD[0]
        amtransindex2140dD = valcoid2140dD[1]
        pmtransbegin2140dD = valcoid2140dD[2]
        pmtransindex2140dD = valcoid2140dD[3]
        valcofunct2140dD = valcofunct(name, isotope, index, amtransbegin2140dD, amtransindex2140dD, pmtransbegin2140dD, pmtransindex2140dD, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1)
        smavevalco2140dD = valcofunct2140dD[8]
        valco_skewsigma_2140dD = valcofunct2140dD[9]
        valco_normsigma_2140dD = valcofunct2140dD[10]
        extrapolated2140dD = valcofunct2140dD[11]
        name = "2140d17o"
        flag1 = data_dict["flag1_2140_d17o"]
        isotope = data_dict["2140d17o"]
        amtransbegin2140d17o = valcoid2140d17o[0]
        amtransindex2140d17o = valcoid2140d17o[1]
        pmtransbegin2140d17o = valcoid2140d17o[2]
        pmtransindex2140d17o = valcoid2140d17o[3]
        valcofunct2140d17o = valcofunct(name, isotope, index, amtransbegin2140d17o, amtransindex2140d17o, pmtransbegin2140d17o, pmtransindex2140d17o, valcopreindex, valcopostindex, valcoprelength, valcopostlength, badh2o, flag1)
        smavevalco2140d17o = valcofunct2140d17o[8]
        valco_skewsigma_2140d17o = valcofunct2140d17o[9]
        valco_normsigma_2140d17o = valcofunct2140d17o[10]
        extrapolated214017o = valcofunct2140d17o[11]
        
    ave_valco_skewsigma_d18o = np.mean(valco_skewsigma_d18o)
    ave_valco_normsigma_d18o = np.mean(valco_normsigma_d18o)
    ave_valco_skewsigma_dD = np.mean(valco_skewsigma_dD)  
    ave_valco_normsigma_dD = np.mean(valco_normsigma_dD)
    
    if corename == 'EGRIP' and d17oflag == 1:
        ave_valco_skewsigma_2140d18o = np.mean(valco_skewsigma_2140d18o)
        ave_valco_normsigma_2140d18o = np.mean(valco_normsigma_2140d18o)
        ave_valco_skewsigma_2140dD = np.mean(valco_skewsigma_2140dD)  
        ave_valco_normsigma_2140dD = np.mean(valco_normsigma_2140dD)
        ave_valco_skewsigma_2140d17o = np.mean(valco_skewsigma_2140d17o)
        ave_valco_normsigma_2140d17o = np.mean(valco_normsigma_2140d17o)

    ### MEMORY CORRECT ISOTOPE VALUES FOLLOWING each valco transition #####################
    ## USE SAME VALCO TRANSITIONS FROM ABS(DIFFD18O[:]) TYPE EQUATIONS

    valcochange = [x for x in data_dict["index"][amvalcobegin[0]:] if data_dict["valco_pos"][x-1]!=data_dict["valco_pos"][x]] ### ALL VALCO TRANSITIONS, VALCO RUNS, IDLE WATER, ICE CORES prior to push ice implementation
    numvalcochange = len(valcochange)
    ecdelays = []
    if verbose ==1:
        print valcochange
        print numvalcochange

    for i in valcochange: # change to diff min or max?
        if data_dict["flag5"][0] == 'A':          #### 1102
            previousbegin = i-20
            previousend = i
            currentbegin = i+50
            currentend = i+100
            correctionlength = 40
        if data_dict["flag5"][0] == 'B':          #### 2130 5 min valco
            previousbegin = i-50
            previousend = i
            currentbegin = i+150
            currentend = i+250 
            correctionlength = 240
        if data_dict["flag5"][0] == 'C' or data_dict["flag5"][0] == 'D':          #### 2130 20 min valco
            previousbegin = i-50
            previousend = i
            currentbegin = i+350 #was 150
            currentend = i+450 #was 250, changed on 20170123
            correctionlength = 450 # was 450 but adjusted to match melter_crunch_4.2
        if currentend >= data_dict["index"][-1]:
            currentend = data_dict["index"][-1]
        
        # for d18o
        previousvalue = np.mean(data_dict["d18o"][previousbegin:previousend])
        currentvalue = np.mean(data_dict["d18o"][currentbegin:currentend])
        stepsize = (currentvalue-previousvalue)
        middlevalue = (previousvalue+currentvalue)/2
        midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["d18o"][x] >= middlevalue-1 and data_dict["d18o"][x] <= middlevalue+1]
        nummidpoints = len(midpoints)
        if nummidpoints>0:
            d18omidpoint = np.ceil(np.mean(midpoints[:]))
            print "midvalue d18o"
        else:
            midvalue = np.max(np.abs(diffd18o[i:currentend]))
            midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diffd18o[x]) == midvalue]
            d18omidpoint = np.ceil(np.mean(midpoints[:]))-10
            print "diff d18o"
        smavevalcod18oindex = np.arange(len(smavevalcod18o))
        smavevalcod18omidpoint = [x for x in smavevalcod18oindex if smavevalcod18o[x] >= 0.5 and smavevalcod18o[x] <= 0.56]
        smavevalcod18omidpoint = smavevalcod18omidpoint[0]
        for z in data_dict["index"][d18omidpoint:d18omidpoint+correctionlength]:
            valcomemd18o[z] = (data_dict["d18o"][z]-previousvalue*(1-smavevalcod18o[z-d18omidpoint+smavevalcod18omidpoint]))/(smavevalcod18o[z-d18omidpoint+smavevalcod18omidpoint])
            data_dict["flag1_d18o"][z] = 'v'
        #for dD
        previousvalue = np.mean(data_dict["dD"][previousbegin:previousend])
        currentvalue = np.mean(data_dict["dD"][currentbegin:currentend])
        stepsize = (currentvalue-previousvalue)
        middlevalue = (previousvalue+currentvalue)/2
        midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["dD"][x] >= middlevalue-4 and data_dict["dD"][x] <= middlevalue+4]
        nummidpoints = len(midpoints)
        if nummidpoints>0:
            dDmidpoint = np.ceil(np.mean(midpoints[:]))
            print "midvalue dD"
        else:
            midvalue = np.max(np.abs(diffdD[i:currentend]))
            midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diffdD[x]) == midvalue]
            dDmidpoint = np.ceil(np.mean(midpoints[:]))-10
            print "diff dD"
        smavevalcodDindex = np.arange(len(smavevalcodD))
        smavevalcodDmidpoint = [x for x in smavevalcodDindex if smavevalcodD[x] >= 0.5 and smavevalcodD[x] <= 0.56]
        smavevalcodDmidpoint = smavevalcodDmidpoint[0]
        for z in data_dict["index"][dDmidpoint:dDmidpoint+correctionlength]:
                valcomemdD[z] = (data_dict["dD"][z]-previousvalue*(1-smavevalcodD[z-dDmidpoint+smavevalcodDmidpoint]))/(smavevalcodD[z-dDmidpoint+smavevalcodDmidpoint])
                data_dict["flag1_dD"][z] = 'v'
        #for ec to calculate ecdelays
        previousvalue = np.mean(data_dict["ec_value"][previousbegin:previousend])
        currentvalue = np.mean(data_dict["ec_value"][currentbegin:currentend])
        stepsize = (currentvalue-previousvalue)
        middlevalue = (previousvalue+currentvalue)/2
        midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["ec_value"][x] >= middlevalue-0.2 and data_dict["ec_value"][x] <= middlevalue+0.2]
        nummidpoints = len(midpoints)
        if nummidpoints>0:
            ecmidpoint = np.ceil(np.mean(midpoints[:]))
            print "midvalue ec"
        else:
            midvalue = np.max(np.abs(diffec[i:currentend]))
            midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diffec[x]) == midvalue]
            ecmidpoint = np.ceil(np.mean(midpoints[:]))-10
            print "diff ec"
        if (data_dict["valco_pos"][i]==6 and data_dict["valco_pos"][i-1]==5) or (data_dict["valco_pos"][i]==5 and data_dict["valco_pos"][i-1]==6):
            ecdelays.append(ecmidpoint-dDmidpoint)
            print "ed midpoint", ecmidpoint
            print "dD midpoint", dDmidpoint
            print "ec delays", ecdelays
        
        if corename == 'EGRIP' and d17oflag ==1:
            # for 2140d18o
            previousvalue = np.mean(data_dict["2140d18o"][previousbegin:previousend])
            currentvalue = np.mean(data_dict["2140d18o"][currentbegin:currentend])
            stepsize = (currentvalue-previousvalue)
            middlevalue = (previousvalue+currentvalue)/2
            midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["2140d18o"][x] >= middlevalue-1 and data_dict["2140d18o"][x] <= middlevalue+1]
            nummidpoints = len(midpoints)
            if nummidpoints>0:
                midpoint = np.ceil(np.mean(midpoints[:]))
            else:
                midvalue = np.max(np.abs(diff2140d18o[i:currentend]))
                midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140d18o[x]) == midvalue]
                midpoint = np.ceil(np.mean(midpoints[:]))-10
            normalizedtime2140d18o = data_dict["j_days"][midpoint:currentend+600]-data_dict["j_days"][midpoint]
            smavevalco2140d18oindex = np.arange(len(smavevalco2140d18o))
            smavevalco2140d18omidpoint = [x for x in smavevalco2140d18oindex if smavevalco2140d18o[x] >= 0.5 and smavevalco2140d18o[x] <= 0.56]
            smavevalco2140d18omidpoint = smavevalco2140d18omidpoint[0]
            for z in data_dict["index"][midpoint:midpoint+correctionlength]:
                valcomem2140d18o[z] = (data_dict["2140d18o"][z]-previousvalue*(1-smavevalco2140d18o[z-midpoint+smavevalco2140d18omidpoint]))/(smavevalco2140d18o[z-midpoint+smavevalco2140d18omidpoint])
                data_dict["flag7_2140_d18o"][z] = 'v'
            #for 2140dD
            previousvalue = np.mean(data_dict["2140dD"][previousbegin:previousend])
            currentvalue = np.mean(data_dict["2140dD"][currentbegin:currentend])
            stepsize = (currentvalue-previousvalue)
            middlevalue = (previousvalue+currentvalue)/2
            midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["2140dD"][x] >= middlevalue-4 and data_dict["2140dD"][x] <= middlevalue+4]
            nummidpoints = len(midpoints)
            if nummidpoints>0:
                midpoint = np.ceil(np.mean(midpoints[:]))
            else:
                midvalue = np.max(np.abs(diff2140dD[i:currentend]))
                midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140dD[x]) == midvalue]
                midpoint = np.ceil(np.mean(midpoints[:]))-10
            normalizedtime2140dD = data_dict["j_days"][midpoint:currentend+600]-data_dict["j_days"][midpoint]
            smavevalco2140dDindex = np.arange(len(smavevalco2140dD))
            smavevalco2140dDmidpoint = [x for x in smavevalco2140dDindex if smavevalco2140dD[x] >= 0.5 and smavevalco2140dD[x] <= 0.56]
            smavevalco2140dDmidpoint = smavevalco2140dDmidpoint[0]
            for z in data_dict["index"][midpoint:midpoint+correctionlength]:
                    valcomem2140dD[z] = (data_dict["2140dD"][z]-previousvalue*(1-smavevalco2140dD[z-midpoint+smavevalco2140dDmidpoint]))/(smavevalco2140dD[z-midpoint+smavevalco2140dDmidpoint])
                    data_dict["flag7_2140_dD"][z] = 'v'
            # for 2140d17o
            previousvalue = np.mean(data_dict["2140d17o"][previousbegin:previousend])
            currentvalue = np.mean(data_dict["2140d17o"][currentbegin:currentend])
            stepsize = (currentvalue-previousvalue)
            middlevalue = (previousvalue+currentvalue)/2
            midpoints = [x for x in data_dict["index"][i:currentend] if data_dict["2140d17o"][x] >= middlevalue-1 and data_dict["2140d17o"][x] <= middlevalue+1]
            nummidpoints = len(midpoints)
            if nummidpoints>0:
                midpoint = np.ceil(np.mean(midpoints[:]))
            else:
                midvalue = np.max(np.abs(diff2140d17o[i:currentend]))
                midpoints = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140d17o[x]) == midvalue]
                midpoint = np.ceil(np.mean(midpoints[:]))-10
            normalizedtime2140d17o = data_dict["j_days"][midpoint:currentend+600]-data_dict["j_days"][midpoint]
            smavevalco2140d17oindex = np.arange(len(smavevalco2140d17o))
            smavevalco2140d17omidpoint = [x for x in smavevalco2140d17oindex if smavevalco2140d17o[x] >= 0.5 and smavevalco2140d17o[x] <= 0.6]
            smavevalco2140d17omidpoint = smavevalco2140d17omidpoint[0]
            for z in data_dict["index"][midpoint:midpoint+correctionlength]:
                valcomem2140d17o[z] = (data_dict["2140d17o"][z]-previousvalue*(1-smavevalco2140d17o[z-midpoint+smavevalco2140d17omidpoint]))/(smavevalco2140d17o[z-midpoint+smavevalco2140d17omidpoint])
                data_dict["flag7_2140_d17o"][z] = 'v'
    
    valcomemdexcess = valcomemdD-8*valcomemd18o

    #### plot memory corrected values onto original graph
    fig21_ax1.plot(data_dict["index"], valcomemd18o, "g-")
    fig21_ax2.plot(data_dict["index"], valcomemdD, "m-")
    fig21_ax3.plot(data_dict["index"], valcomemdexcess, "k-")
    
    if corename == 'EGRIP' and d17oflag == 1:
        fig24_ax1.plot(data_dict["index"], valcomem2140d18o, "g-")
        fig24_ax2.plot(data_dict["index"], valcomem2140dD, "m-")
        fig24_ax3.plot(data_dict["index"], valcomem2140d17o, "k-")
    
    #### do not need to do for method D, except for maybe 2 runs where there was not a push ice, but a valco transition right beofre the ice.

    ##### NEA IDENTIFICATION #######################################################
    ## Look at comments to locate amnea and pmnea
    amneabegin = [x for x in data_dict["index"][1:] if data_dict["comments"][x-1]!=153 and data_dict["comments"][x]==153]
    pmneabegin = [x for x in data_dict["index"][1:] if data_dict["comments"][x-1]!=161 and data_dict["comments"][x]==161]

    if len(amvalcobegin) > 1:
        amneabegin = [amneabegin[-1]]
    if len(amneabegin) < 1:
        amneabegin = [pmneabegin[-1]]
    if len(pmneabegin) > 1:
        pmneabegin = [pmneabegin[-1]]
    if len(pmneabegin) < 1:
        pmneabegin = [amneabegin[-1]]
        
    if verbose ==1:
        print "AM neapolitan begin ", amneabegin
        print "PM neapolitan begin ", pmneabegin
        
    ## Stop and ask if need to edit?
        checkbegin = raw_input("Do you want to edit the beginning am neapolitan indices?")
        if checkbegin in ('y', 'ye', 'yes'):
            amneabegin = input("Please type new indices list in [ ]...")
            print "New AM neapolitan begin ", amneabegin
        checkbegin = raw_input("Do you want to edit the beginning pm neapolitan indices?")
        if checkbegin in ('y', 'ye', 'yes'):
            pmneabegin = input("Please type new indices list in [ ]...")
            print "New PM neapolitan begin ", pmneabegin
        
    #### run each isotope for both am and pm, then combine am and pm neas to get average OUTSIDE OF FUNCTION
    
    ## For 2130
    index = data_dict["index"]
    comments = data_dict["comments"]
    epoch = data_dict["epoch"]
    water_ppm = data_dict["water_ppm"]
    
    #d18o AM and PM
    name = 'd18o'
    isotope = data_dict["d18o"]
    diffisotope = diffd18o
    trantype = 'AMNea'
    neabegin = amneabegin
    amnead18o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
    trantype = 'PMNea'
    neabegin = pmneabegin
    pmnead18o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
    neaindex = np.append(amnead18o[0],pmnead18o[0])
    nead18o = np.append(amnead18o[2],pmnead18o[2])
    sortindex = np.argsort(neaindex)
    sortedneaindex = neaindex[sortindex]
    sortednead18o = nead18o[sortindex]
    nea_skewsigma_d18o = np.append(amnead18o[3],pmnead18o[3])
    nea_normsigma_d18o = np.append(amnead18o[4],pmnead18o[4])
    ave_nea_skewsigma_d18o = np.mean(nea_skewsigma_d18o)
    ave_nea_normsigma_d18o = np.mean(nea_normsigma_d18o)
    
    #dD AM and PM
    name = 'dD'
    isotope = data_dict["dD"]
    diffisotope = diffdD
    trantype = 'AMNea'
    neabegin = amneabegin
    amneadD = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
    trantype = 'PMNea'
    neabegin = pmneabegin
    pmneadD = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
    neaindex = np.append(amneadD[0],pmneadD[0])
    neadD = np.append(amneadD[2],pmneadD[2])
    sortindex = np.argsort(neaindex)
    sortedneaindex = neaindex[sortindex]
    sortedneadD = neadD[sortindex]
    nea_skewsigma_dD = np.append(amneadD[3],pmneadD[3])
    nea_normsigma_dD = np.append(amneadD[4],pmneadD[4])
    ave_nea_skewsigma_dD = np.mean(nea_skewsigma_dD)
    ave_nea_normsigma_dD = np.mean(nea_normsigma_dD)
    
    if corename == 'EGRIP' and d17oflag == 1:
        ## For 2140
        # 2140d18o AM and PM
        name = '2140d18o'
        isotope = data_dict["2140d18o"]
        diffisotope = diff2140d18o
        trantype = 'AMNea'
        neabegin = amneabegin
        amnea2140d18o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        trantype = 'PMNea'
        neabegin = pmneabegin
        pmnea2140d18o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        nea2140index = np.append(amnea2140d18o[0],pmnea2140d18o[0])
        nea2140d18o = np.append(amnea2140d18o[2],pmnea2140d18o[2])
        sort2140index = np.argsort(nea2140index)
        sortednea2140index = nea2140index[sort2140index]
        sortednea2140d18o = nea2140d18o[sort2140index]
        nea_skewsigma_2140d18o = np.append(amnea2140d18o[3],pmnea2140d18o[3])
        nea_normsigma_2140d18o = np.append(amnea2140d18o[4],pmnea2140d18o[4])
        ave_nea_skewsigma_2140d18o = np.mean(nea_skewsigma_2140d18o)
        ave_nea_normsigma_2140d18o = np.mean(nea_normsigma_2140d18o)
        
        # 2140dD AM and PM
        name = '2140dD'
        isotope = data_dict["2140dD"]
        diffisotope = diff2140dD
        trantype = 'AMNea'
        neabegin = amneabegin
        amnea2140dD = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        trantype = 'PMNea'
        neabegin = pmneabegin
        pmnea2140dD = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        nea2140index = np.append(amnea2140dD[0],pmnea2140dD[0])
        nea2140dD = np.append(amnea2140dD[2],pmnea2140dD[2])
        sort2140index = np.argsort(nea2140index)
        sortednea2140index = nea2140index[sort2140index]
        sortednea2140dD = nea2140dD[sortindex]
        nea_skewsigma_2140dD = np.append(amnea2140dD[3],pmnea2140dD[3])
        nea_normsigma_2140dD = np.append(amnea2140dD[4],pmnea2140dD[4])
        ave_nea_skewsigma_2140dD = np.mean(nea_skewsigma_2140dD)
        ave_nea_normsigma_2140dD = np.mean(nea_normsigma_2140dD)
        
        # 2140d17o AM and PM
        name = '2140d17o'
        isotope = smooth2140d17o
        diffisotope = diff2140d17o
        trantype = 'AMNea'
        neabegin = amneabegin
        amnea2140d17o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        trantype = 'PMNea'
        neabegin = pmneabegin
        pmnea2140d17o = neafunct(trantype, name, index, isotope, diffisotope, comments, epoch, water_ppm, neabegin)
        nea2140index = np.append(amnea2140d17o[0],pmnea2140d17o[0])
        nea2140d17o = np.append(amnea2140d17o[2],pmnea2140d17o[2])
        sort2140index = np.argsort(nea2140index)
        sortednea2140index = nea2140index[sort2140index]
        sortednea2140d17o = nea2140d17o[sort2140index]
        nea_skewsigma_2140d17o = np.append(amnea2140d17o[3],amnea2140d17o[3])
        nea_normsigma_2140d17o = np.append(amnea2140d17o[4],pmnea2140d17o[4])
        ave_nea_skewsigma_2140d17o = np.mean(nea_skewsigma_2140d17o)
        ave_nea_normsigma_2140d17o = np.mean(nea_normsigma_2140d17o)
        
    print "nea results", "d18o skew", nea_skewsigma_d18o, \
        "d18o norm", nea_normsigma_d18o, \
        "dD skew", nea_skewsigma_dD, \
        "dD norm", nea_normsigma_dD
    
    avenead18o = amnead18o[2].copy()
    aveneadD = amneadD[2].copy()
    for p in amnead18o[0]:
        aveindex = np.where(sortedneaindex==p)[0]
        avenead18o[p] = np.mean(sortednead18o[aveindex])
        aveneadD[p] = np.mean(sortedneadD[aveindex])
    smavenead18o = smooth(avenead18o)
    smaveneadD = smooth(aveneadD)
    
    fig421 = plt.figure(421)
    clear = plt.clf()
    fig421_ax1 = fig421.add_subplot(211)
    fig421_ax1.plot(amnead18o[0], amnead18o[2], "b-", pmnead18o[0], pmnead18o[2], "g-",  \
        sortedneaindex, sortednead18o, "k-",  amnead18o[0], smavenead18o, "b-")
    fig421_ax1.set_ylabel("d18o")
    fig421_ax2 = fig421.add_subplot(212)
    fig421_ax2.plot(amneadD[0], amneadD[2], "r-",  pmneadD[0], pmneadD[2], "m-",  \
        sortedneaindex, sortedneadD, "k-",  amneadD[0], smaveneadD, "r-")
    fig421_ax2.set_ylabel("dD")
    fig421_ax2.set_xlabel("Index")
    fig421_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
        
    if corename == 'EGRIP' and d17oflag == 1:
        print "2140 nea results", "2140 d18o skew", nea_skewsigma_2140d18o, \
            "2140 d18o norm", nea_normsigma_2140d18o, \
            "2140 dD skew", nea_skewsigma_2140dD, \
            "2140 dD norm", nea_normsigma_2140dD, \
            "2140 d17o skew", nea_skewsigma_2140d17o, \
            "2140 d18o norm", nea_normsigma_2140d17o
        avenea2140d18o = amnea2140d18o[2].copy()
        avenea2140dD = amnea2140dD[2].copy()
        avenea2140d17o = amnea2140d17o[2].copy()
        for p in amnea2140d18o[0]:
            ave2140index = np.where(sortednea2140index==p)[0]
            avenea2140d18o[p] = np.mean(sortednea2140d18o[ave2140index])
            avenea2140dD[p] = np.mean(sortednea2140dD[ave2140index])
            avenea2140d17o[p] = np.mean(sortednea2140d17o[ave2140index])
        smavenea2140d18o = smooth(avenea2140d18o)
        smavenea2140dD = smooth(avenea2140dD)
        smavenea2140d17o = smooth(avenea2140d17o)
        
        fig422 = plt.figure(422)
        clear = plt.clf()
        fig422_ax1 = fig422.add_subplot(311)
        fig422_ax1.plot(amnea2140d18o[0], amnea2140d18o[2], "b-", pmnea2140d18o[0], pmnea2140d18o[2], "g-",  \
            sortednea2140index, sortednea2140d18o, "k-",  amnea2140d18o[0], smavenea2140d18o, "b-")
        fig422_ax1.set_ylabel("d18o")
        fig422_ax2 = fig422.add_subplot(312)
        fig422_ax2.plot(amnea2140dD[0], amnea2140dD[2], "r-",  pmnea2140dD[0], pmnea2140dD[2], "m-",  \
            sortednea2140index, sortednea2140dD, "k-",  amnea2140dD[0], smavenea2140dD, "r-")
        fig422_ax2.set_ylabel("dD")
        fig422_ax3 = fig422.add_subplot(313)
        fig422_ax3.plot(amnea2140d17o[0], amnea2140d17o[2], "m-", pmnea2140d17o[0], pmnea2140d17o[2], "g-",  \
            sortednea2140index, sortednea2140d17o, "k-",  amnea2140d17o[0], smavenea2140d17o, "b-")
        fig422_ax3.set_ylabel("d17o")
        fig422_ax3.set_xlabel("Index")
        fig422_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    ###### MEMORY CORRECTION APPLIED TO NEAS TO CHECK CORRECTION
    amneachange = [x for x in data_dict["index"][amneabegin[-1]:amneabegin[-1]+2000] if (data_dict["comments"][x-1]!=101 and \
        data_dict["comments"][x]==101) or (data_dict["comments"][x-1]!=104 and data_dict["comments"][x]==104)] ### low to high and high to low
    pmneachange = [x for x in data_dict["index"][pmneabegin[-1]:pmneabegin[-1]+2000] if (data_dict["comments"][x-1]!=101 and \
        data_dict["comments"][x]==101) or (data_dict["comments"][x-1]!=104 and data_dict["comments"][x]==104)] ### low to high and high to low
    meltchange = np.append(amneachange,pmneachange)
    nummeltchange = len(meltchange)
    if verbose == 1:
        print meltchange
        print nummeltchange

    neamemd18o = deepcopy(valcomemd18o)
    neamemdD = deepcopy(valcomemdD)
    neamem2140d18o = deepcopy(valcomem2140d18o)
    neamem2140dD = deepcopy(valcomem2140dD)
    neamem2140d17o = deepcopy(valcomem2140d17o)

    if data_dict["epoch"][0] < 1325376000:                                           #### 1102
        for i in meltchange: # change to diff min or max?
            previousbegin = i-100
            previousend = i
            currentbegin = i+50
            currentend = i+100
            previousvalued18o = np.mean(data_dict["d18o"][previousbegin:previousend])
            previousvaluedD = np.mean(data_dict["dD"][previousbegin:previousend])
            currentvalued18o = np.mean(data_dict["d18o"][currentbegin:currentend])
            currentvaluedD = np.mean(data_dict["dD"][currentbegin:currentend])
            stepsized18o = (currentvalued18o-previousvalued18o)
            stepsizedD = (currentvaluedD-previousvaluedD)
            middlevalued18o = (previousvalued18o+currentvalued18o)/2
            middlevaluedD = (previousvaluedD+currentvaluedD)/2
            midpointsd18o = [x for x in data_dict["index"][i:currentend] if data_dict["d18o"][x] >= middlevalued18o-2 and data_dict["d18o"][x] <= middlevalued18o+2]
            nummidpointsd18o = len(midpointsd18o)
            if nummidpointsd18o>0:
                midpointd18o = np.ceil(np.mean(midpointsd18o[:]))
            else:
                midvalued18o = np.max(np.abs(diffd18o[i:currentend]))
                midpointsd18o = [x for x in data_dict["index"][i:currentend] if np.abs(diffd18o[x]) == midvalued18o]
                midpointd18o = np.ceil(np.mean(midpointsd18o[:]))
            smavenead18oindex = np.arange(len(smavenead18o))
            smavenead18omidpoint = [x for x in smavenead18oindex if smavenead18o[x] >= 0.5 and smavenead18o[x] <= 0.55]
            smavenead18omidpoint = smavenead18omidpoint[0]
            for z in data_dict["index"][midpointd18o:midpointd18o+100]:
                neamemd18o[z] = (valcomemd18o[z]-previousvalued18o*(1-smavenead18o[z-midpointd18o+smavenead18omidpoint]))/(smavenead18o[z-midpointd18o+smavenead18omidpoint])
                data_dict["flag1_d18o"][z] = 'n'
            #dD
            midpointsdD = [x for x in data_dict["index"][i:currentend] if data_dict["dD"][x] >= middlevaluedD-5 and data_dict["dD"][x] <= middlevaluedD+5]
            nummidpointsdD = len(midpointsdD)
            if nummidpointsdD>0:
                midpointdD = np.ceil(np.mean(midpointsdD[:]))
            else:
                midvaluedD = np.max(np.abs(diffdD[i:currentend]))
                midpointsdD = [x for x in data_dict["index"][i:currentend] if np.abs(diffdD[x]) == midvaluedD]
                midpointdD = np.ceil(np.mean(midpointsdD[:]))
            smaveneadDindex = np.arange(len(smaveneadD))
            smaveneadDmidpoint = [x for x in smaveneadDindex if smaveneadD[x] >= 0.5 and smaveneadD[x] <= 0.55]
            smaveneadDmidpoint = smaveneadDmidpoint[0]
            for z in data_dict["index"][midpointdD:midpointdD+100]:
                neamemdD[z] = (valcomemdD[z]-previousvaluedD*(1-smaveneadD[z-midpoint+smaveneadDmidpoint]))/(smaveneadD[z-midpointdD+smaveneadDmidpoint])
                data_dict["flag1_dD"][z] = 'n'
        neamemdexcess = neamemdD-8*neamemd18o
        
    if data_dict["epoch"][0] > 1325376000:                                           ### 2130       
        print 'processing 2130 data'
        for i in meltchange: # change to diff min or max?
            previousbegin = i-100
            previousend = i
            currentbegin = i+350 #was 450
            currentend = i+450 #was 550
            previousvalued18o = np.mean(data_dict["d18o"][previousbegin:previousend])
            previousvaluedD = np.mean(data_dict["dD"][previousbegin:previousend])
            currentvalued18o = np.mean(data_dict["d18o"][currentbegin:currentend])
            currentvaluedD = np.mean(data_dict["dD"][currentbegin:currentend])
            stepsized18o = (currentvalued18o-previousvalued18o)
            stepsizedD = (currentvaluedD-previousvaluedD)
            middlevalued18o = (previousvalued18o+currentvalued18o)/2
            middlevaluedD = (previousvaluedD+currentvaluedD)/2
            midpointsd18o = [x for x in data_dict["index"][i:currentend] if data_dict["d18o"][x] >= middlevalued18o-1 and data_dict["d18o"][x] <= middlevalued18o+1]
            nummidpointsd18o = len(midpointsd18o)
            if nummidpointsd18o>0:
                midpointd18o = np.ceil(np.mean(midpointsd18o[:]))
            else:
                midvalued18o = np.min(diffd18o[i:currentend]) #np.max(np.abs(diffd18o[i:currentend])) was there for if the push was less than the ice core, but not useful here
                midpointsd18o = [x for x in data_dict["index"][i:currentend] if np.abs(diffd18o[x]) == midvalued18o]
                midpointd18o = np.ceil(np.mean(midpointsd18o[:]))
            smavenead18oindex = np.arange(len(smavenead18o))
            smavenead18omidpoint = [x for x in smavenead18oindex if smavenead18o[x] >= 0.5 and smavenead18o[x] <= 0.55]
            smavenead18omidpoint = smavenead18omidpoint[0]
            for z in data_dict["index"][midpointd18o:midpointd18o+200]: #temporariliy changed to 198 from 200 for 201607, 179 for 20160723, 180 for 20191129
                neamemd18o[z] = (valcomemd18o[z]-previousvalued18o*(1-smavenead18o[z-midpointd18o+smavenead18omidpoint]))/(smavenead18o[z-midpointd18o+smavenead18omidpoint])
                data_dict["flag1_d18o"][z] = 'n'
                
            #dD
            midpointsdD = [x for x in data_dict["index"][i:currentend] if data_dict["dD"][x] >= middlevaluedD-4 and data_dict["dD"][x] <= middlevaluedD+4]
            nummidpointsdD = len(midpointsdD)
            if nummidpointsdD>0:
                midpointdD = np.ceil(np.mean(midpointsdD[:]))
            else:
                midvaluedD = np.min(diffdD[i:currentend]) #np.max(np.abs(diffdD[i:currentend]))
                midpointsdD = [x for x in data_dict["index"][i:currentend] if np.abs(diffdD[x]) == midvaluedD]
                midpointdD = np.ceil(np.mean(midpointsdD[:])) 
            smaveneadDindex = np.arange(len(smaveneadD))
            smaveneadDmidpoint = [x for x in smaveneadDindex if smaveneadD[x] >= 0.5 and smaveneadD[x] <= 0.55]
            smaveneadDmidpoint = smaveneadDmidpoint[0]
            for z in data_dict["index"][midpointdD:midpointdD+175]:
                neamemdD[z] = (valcomemdD[z]-previousvaluedD*(1-smaveneadD[z-midpointdD+smaveneadDmidpoint]))/(smaveneadD[z-midpointdD+smaveneadDmidpoint])
                data_dict["flag1_dD"][z] = 'n'
            
            binstart = [0,20,40,60,80,100,120,140,160,180,200]
            for x in binstart:
                #collect stats for error of memory correction
                #valueindexd18o = [midpointd18o+60:midpointd18o+80]
                rawmemvalued18o = np.mean(valcomemd18o[midpointd18o+x:midpointd18o+x+20])
                rawprecisiond18o = np.std(valcomemd18o[midpointd18o+x:midpointd18o+x+20])
                correctedmemvalued18o = np.mean(neamemd18o[midpointd18o+x:midpointd18o+x+20])
                correctedmemprecisiond18o = np.std(neamemd18o[midpointd18o+x:midpointd18o+x+20])
                memcorrectionsized18o = correctedmemvalued18o-rawmemvalued18o
                memacuracyd18o = correctedmemvalued18o-currentvalued18o
                #valueindexdD = [midpointdD+60:midpointdD+80]
                rawmemvaluedD = np.mean(valcomemdD[midpointdD+x:midpointdD+x+20])
                rawprecisiondD = np.std(valcomemdD[midpointdD+x:midpointdD+x+20])
                correctedmemvaluedD = np.mean(neamemdD[midpointdD+x:midpointdD+x+20])
                correctedmemprecisiondD = np.std(neamemdD[midpointdD+x:midpointdD+x+20])
                memcorrectionsizedD = correctedmemvaluedD-rawmemvaluedD
                memacuracydD = correctedmemvaluedD-currentvaluedD
                #neafile = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/NeaPerformanceFile"
                neafile = "/Users/frio/EGRIP_2023/Data/NeaPerformanceFile"
                file = open(neafile, "a")
                neadata = np.transpose(np.vstack((x, x+20, filename, previousvalued18o, currentvalued18o, middlevalued18o, rawmemvalued18o, \
                    rawprecisiond18o, correctedmemvalued18o, correctedmemprecisiond18o, memcorrectionsized18o, memacuracyd18o, \
                        previousvaluedD, currentvaluedD, middlevaluedD, rawmemvaluedD, rawprecisiondD, \
                            correctedmemvaluedD, correctedmemprecisiondD, memcorrectionsizedD, memacuracydD)))
                np.savetxt(file, neadata, delimiter = "\t", fmt = ("%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s"))
                file.close()
            
        neamemdexcess = neamemdD-8*neamemd18o
        
        if corename == 'EGRIP' and d17oflag == 1:
            for i in meltchange: # change to diff min or max?
                previousbegin = i-100
                previousend = i
                currentbegin = i+350 #was 450
                currentend = i+450 #was 550
                previousvalue2140d18o = np.mean(data_dict["2140d18o"][previousbegin:previousend])
                previousvalue2140dD = np.mean(data_dict["2140dD"][previousbegin:previousend])
                previousvalue2140d17o = np.mean(data_dict["2140d17o"][previousbegin:previousend])
                currentvalue2140d18o = np.mean(data_dict["2140d18o"][currentbegin:currentend])
                currentvalue2140dD = np.mean(data_dict["2140dD"][currentbegin:currentend])
                currentvalue2140d17o = np.mean(data_dict["2140d17o"][currentbegin:currentend])
                stepsize2140d18o = (currentvalue2140d18o-previousvalue2140d18o)
                stepsize2140dD = (currentvalue2140dD-previousvalue2140dD)
                stepsize2140d17o = (currentvalue2140d17o-previousvalue2140d17o)
                middlevalue2140d18o = (previousvalue2140d18o+currentvalue2140d18o)/2
                middlevalue2140dD = (previousvalue2140dD+currentvalue2140dD)/2
                middlevalue2140d17o = (previousvalue2140d17o+currentvalue2140d17o)/2
                midpoints2140d18o = [x for x in data_dict["index"][i:currentend] if data_dict["2140d18o"][x] >= middlevalue2140d18o-.5 and data_dict["2140d18o"][x] <= middlevalue2140d18o+.5]
                nummidpoints2140d18o = len(midpoints2140d18o)
                if nummidpoints2140d18o>0:
                    midpoint2140d18o = np.ceil(np.mean(midpoints2140d18o[:]))
                else:
                    midvalue2140d18o = np.min(diff2140d18o[i:currentend]) #np.max(np.abs(diff2140d18o[i:currentend]))
                    midpoints2140d18o = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140d18o[x]) == midvalue2140d18o]
                    midpoint2140d18o = np.ceil(np.mean(midpoints2140d18o[:]))
                smavenea2140d18oindex = np.arange(len(smavenea2140d18o))
                smavenea2140d18omidpoint = [x for x in smavenea2140d18oindex if smavenea2140d18o[x] >= 0.5 and smavenea2140d18o[x] <= 0.55]
                smavenea2140d18omidpoint = smavenea2140d18omidpoint[0]
                for z in data_dict["index"][midpoint2140d18o:midpoint2140d18o+200]:
                    neamem2140d18o[z] = (valcomem2140d18o[z]-previousvalue2140d18o*(1-smavenea2140d18o[z-midpoint2140d18o+smavenea2140d18omidpoint]))/(smavenea2140d18o[z-midpoint2140d18o+smavenea2140d18omidpoint])
                    data_dict["flag7_2140_d18o"][z] = 'n'
#                #valueindex2140d18o = [midpoint2140d18o+60:midpoint2140d18o+80]
#                rawmemvalue2140d18o = np.mean(valcomem2140d18o[midpointd18o+60:midpointd18o+80])
#                rawprecision2140d18o = np.std(valcomem2140d18o[midpointd18o+60:midpointd18o+80])
#                correctedmemvalue2140d18o = np.mean(neamem2140d18o[midpointd18o+60:midpointd18o+80])
#                correctedmemprecision2140d18o = np.std(neamem2140d18o[midpointd18o+60:midpointd18o+80])
#                memcorrectionsize2140d18o = correctedmemvalue2140d18o-rawmemvalue2140d18o
#                memacuracy2140d18o = correctedmemvalue2140d18o-currentvalue2140d18o
                    
                #dD
                midpoints2140dD = [x for x in data_dict["index"][i:currentend] if data_dict["2140dD"][x] >= middlevalue2140dD-2 and data_dict["2140dD"][x] <= middlevalue2140dD+2]
                nummidpoints2140dD = len(midpoints2140dD)
                if nummidpoints2140dD>0:
                    midpoint2140dD = np.ceil(np.mean(midpoints2140dD[:]))
                else:
                    midvalue2140dD = np.min(diff2140dD[i:currentend]) #np.max(np.abs(diff2140dD[i:currentend]))
                    midpoints2140dD = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140dD[x]) == midvalue2140dD]
                    midpoint2140dD = np.ceil(np.mean(midpoints2140dD[:])) 
                smavenea2140dDindex = np.arange(len(smavenea2140dD))
                smavenea2140dDmidpoint = [x for x in smavenea2140dDindex if smavenea2140dD[x] >= 0.5 and smavenea2140dD[x] <= 0.55]
                smavenea2140dDmidpoint = smavenea2140dDmidpoint[0]
                for z in data_dict["index"][midpoint2140dD:midpoint2140dD+200]:
                    neamem2140dD[z] = (valcomem2140dD[z]-previousvalue2140dD*(1-smavenea2140dD[z-midpoint2140dD+smavenea2140dDmidpoint]))/(smavenea2140dD[z-midpoint2140dD+smavenea2140dDmidpoint])
                    data_dict["flag7_2140_dD"][z] = 'n'
#                #valueindex2140dD = [midpoint2140dD+60:midpoint2140dD+80]
#                rawmemvalue2140dD = np.mean(valcomem2140dD[midpointdD+60:midpointdD+80])
#                rawprecision2140dD = np.std(valcomem2140dD[midpointdD+60:midpointdD+80])
#                correctedmemvalue2140dD = np.mean(neamem2140dD[midpointdD+60:midpointdD+80])
#                correctedmemprecision2140dD = np.std(neamem2140dD[midpointdD+60:midpointdD+80])
#                memcorrectionsize2140dD = correctedmemvalue2140dD-rawmemvalue2140dD
#                memacuracy2140dD = correctedmemvalue2140dD-currentvalue2140dD
                    
                #d17o
                midpoints2140d17o = [x for x in data_dict["index"][i:currentend] if smooth2140d17o[x] >= middlevalue2140d17o-0.3 and smooth2140d17o[x] <= middlevalue2140d17o+0.3]
                nummidpoints2140d17o = len(midpoints2140d17o)
                if nummidpoints2140d17o>0:
                    midpoint2140d17o = np.ceil(np.mean(midpoints2140d17o[:]))
                else:
                    midvalue2140d17o = np.min(diff2140d17o[i:currentend])#np.max(np.abs(diff2140d17o[i:currentend]))
                    midpoints2140d17o = [x for x in data_dict["index"][i:currentend] if np.abs(diff2140d17o[x]) == midvalue2140d17o]
                    midpoint2140d17o = np.ceil(np.mean(midpoints2140d17o[:]))
                smavenea2140d17oindex = np.arange(len(smavenea2140d17o))
                smavenea2140d17omidpoint = [x for x in smavenea2140d17oindex if smavenea2140d17o[x] >= 0.5 and smavenea2140d17o[x] <= 0.55]
                smavenea2140d17omidpoint = smavenea2140d17omidpoint[0]
                for z in data_dict["index"][midpoint2140d17o:midpoint2140d17o+200]:
                    neamem2140d17o[z] = (valcomem2140d17o[z]-previousvalue2140d17o*(1-smavenea2140d17o[z-midpoint2140d17o+smavenea2140d17omidpoint]))/(smavenea2140d17o[z-midpoint2140d17o+smavenea2140d17omidpoint])
                    data_dict["flag7_2140_d17o"][z] = 'n'    
#                #valueindex2140d17o = [midpointd17o+60:midpointd17o+80]
#                rawmemvalue2140d17o = np.mean(valcomem2140d17o[midpointd17o+60:midpointd17o+80])
#                rawprecision2140d17o = np.std(valcomem2140d17o[midpoint2140d17o+60:midpointd17o+80])
#                correctedmemvalue2140d17o = np.mean(neamem2140d17o[midpointd17o+60:midpointd17o+80])
#                correctedmemprecision2140d17o = np.std(neamem2140d17o[midpointd17o+60:midpointd17o+80])
#                memcorrectionsize2140d17o = correctedmemvalue2140d17o-rawmemvalue2140d17o
#                memacuracy2140d17o = correctedmemvalue2140d17o-currentvalue2140d17o
                
            neamem2140dexcess = neamem2140dD-8*neamem2140d18o
            
            fig24_ax1.plot(data_dict["index"], neamem2140d18o, "c-")
            fig24_ax2.plot(data_dict["index"], neamem2140dD, "k-")
            fig24_ax3.plot(data_dict["index"], neamem2140d17o, "y-")
        
    #### plot memory corrected values onto original graph
    fig21_ax1.plot(data_dict["index"], neamemd18o, "c-")
    fig21_ax2.plot(data_dict["index"], neamemdD, "k-")
    fig21_ax3.plot(data_dict["index"], neamemdexcess, "y-") 

    ##### ISOTOPE CALIBRATION - linear fit, with slope intercept and appy slope intercept to data
    def linearfunc (m, x, b):
        return m*x + b
    
    print "made it to isotope calibration"
    
    corrd18o = deepcopy(neamemd18o)
    corrdD = deepcopy(neamemdD)
    corr2140d18o = deepcopy(neamem2140d18o)
    corr2140dD = deepcopy(neamem2140dD)
    corr2140d17o = deepcopy(neamem2140d17o)
    
    rawstdd18o = [0,0,0]
    stdevrawstdd18o = [0,0,0]
    rawstddD = [0,0,0]
    stdevrawstddD = [0,0,0]
    
    #   STANDARDS  KBW     KAW     (KGW)     KPW    for WAIS06A and EGRIP
    #   position   0,1     2,3      (4,5)    6,7
    # knownd18o = [-14.19, -30.35, (-38.09), -45.43]
    # knowndD   = [-111.8, -239.3, (-298.7), -355.6]
    
    #   STANDARDS  kaw     kgw     (kpw)     vw1f   for SPIceCore 2015
    #   position   0,1     2,3      (4,5)    6,7
    # knownd18o = [-30.35, -38.09, (-45.43),  -56]
    # knowndD   = [-239.3, -298.7, (-355.6), -438]
    
    rawstdd18o[0] = valcofunctd18o[0]
    stdevrawstdd18o[0] = valcofunctd18o[1]
    rawstdd18o[1] = valcofunctd18o[2]
    stdevrawstdd18o[1] = valcofunctd18o[3]
    rawstdd18o[2] = valcofunctd18o[6]
    stdevrawstdd18o[2] = valcofunctd18o[7]
    rawtrapd18o = valcofunctd18o[4]
    stdevrawtrapd18o = valcofunctd18o[5]
    
    rawstddD[0] = valcofunctdD[0]
    stdevrawstddD[0] = valcofunctdD[1] 
    rawstddD[1] = valcofunctdD[2]
    stdevrawstddD[1] = valcofunctdD[3]
    rawstddD[2] = valcofunctdD[6]
    stdevrawstddD[2] = valcofunctdD[7]
    rawtrapdD = valcofunctdD[4]
    stdevrawtrapdD = valcofunctdD[5]
    
    d18oslope, d18ointercept, d18or_value, d18op_value, d18ostd_err = stats.linregress(rawstdd18o,knownd18o)
    dDslope, dDintercept, dDr_value, dDp_value, dDstd_err = stats.linregress(rawstddD,knowndD)
    
    if verbose ==1:
        print " "
        print "d18O Stats"
        print "knownd18o", knownd18o
        print "rawstdd18o", rawstdd18o
        print "stdevrawstdd18o", stdevrawstdd18o
        print "d18oslope", d18oslope
        print "d18ointercept", d18ointercept
        print " "
        print "dD Stats"
        print "knowndD", knowndD
        print "rawstddD", rawstddD
        print "stdevrawstddD", stdevrawstddD
        print "dDslope", dDslope
        print "dDintercept", dDintercept

    corrd18o = neamemd18o * d18oslope + d18ointercept
    corrdD = neamemdD * dDslope + dDintercept
        
    #   STANDARDS   kaw       uwww     (kpw)      vw1f   for SPIceCore 2016
    #   position    0,1       2,3      (4,5)      6,7
    # knownd18o = [-30.30,  -33.82,  (-45.41),  -56.59]
    # knowndD   = [-239.13, -268.30, (-355.18), -438.43]
    
    if corename == 'EGRIP' and d17oflag == 1:
        rawstd2140d18o = [0,0,0]
        stdevrawstd2140d18o = [0,0,0]
        rawstd2140dD = [0,0,0]
        stdevrawstd2140dD = [0,0,0]
        rawstd2140d17o = [0,0,0]
        stdevrawstd2140d17o = [0,0,0]
        
        rawstd2140d18o[0] = valcofunct2140d18o[0]
        stdevrawstd2140d18o[0] = valcofunct2140d18o[1]
        rawstd2140d18o[1] = valcofunct2140d18o[2]
        stdevrawstd2140d18o[1] = valcofunct2140d18o[3]
        rawstd2140d18o[2] = valcofunct2140d18o[6]
        stdevrawstd2140d18o[2] = valcofunct2140d18o[7]
        rawtrap2140d18o = valcofunct2140d18o[4]
        stdevrawtrap2140d18o = valcofunct2140d18o[5]
        
        rawstd2140dD[0] = valcofunct2140dD[0]
        stdevrawstd2140dD[0] = valcofunct2140dD[1] 
        rawstd2140dD[1] = valcofunct2140dD[2]
        stdevrawstd2140dD[1] = valcofunct2140dD[3]
        rawstd2140dD[2] = valcofunct2140dD[6]
        stdevrawstd2140dD[2] = valcofunct2140dD[7]
        rawtrap2140dD = valcofunct2140dD[4]
        stdevrawtrap2140dD = valcofunct2140dD[5]

        rawstd2140d17o[0] = valcofunct2140d17o[0]
        stdevrawstd2140d17o[0] = valcofunct2140d17o[1]
        rawstd2140d17o[1] = valcofunct2140d17o[2]
        stdevrawstd2140d17o[1] = valcofunct2140d17o[3]
        rawstd2140d17o[2] = valcofunct2140d17o[6]
        stdevrawstd2140d17o[2] = valcofunct2140d17o[7]
        rawtrap2140d17o = valcofunct2140d17o[4]
        stdevrawtrap2140d17o = valcofunct2140d17o[5]
        
        slope2140d18o, intercept2140d18o, r_value2140d18o, p_value2140d18o, std_err2140d18o = stats.linregress(rawstd2140d18o,knownd18o)
        slope2140dD, intercept2140dD, r_value2140dD, p_value2140dD, std_err2140dD = stats.linregress(rawstd2140dD,knowndD)
        slope2140d17o, intercept2140d17o, r_value2140d17o, p_value2140d17o, std_err2140d17o = stats.linregress(rawstd2140d17o,knownd17o)

        if verbose ==1:
            print " "
            print "2140 d18O Stats"
            print "knownd18o", knownd18o
            print "rawstdd18o", rawstd2140d18o
            print "stdevrawstdd18o", stdevrawstd2140d18o
            print "d18oslope", slope2140d18o
            print "d18ointercept", intercept2140d18o
            print " "
            print "2140 dD Stats"
            print "knowndD", knowndD
            print "rawstddD", rawstd2140dD
            print "stdevrawstddD", stdevrawstd2140dD
            print "dDslope", slope2140dD
            print "dDintercept", intercept2140dD
            print " "
            print "2140 d17O Stats"
            print "knownd17o", knownd17o
            print "rawstdd17o", rawstd2140d17o
            print "stdevrawstdd17o", stdevrawstd2140d17o
            print "d17oslope", slope2140d17o
            print "d17ointercept", intercept2140d17o

        corr2140d18o = neamem2140d18o * slope2140d18o + intercept2140d18o
        corr2140dD = neamem2140dD * slope2140dD + intercept2140dD
        corr2140d17o = neamem2140d17o * slope2140d17o + intercept2140d17o

    #### plot slope corrected values onto original graphs
    fig21_ax1.plot(data_dict["index"], corrd18o, "b-")
    fig21_ax2.plot(data_dict["index"], corrdD, "r-") 
    if corename == 'EGRIP' and d17oflag == 1:
        fig24_ax1.plot(data_dict["index"], corr2140d18o, "b-")
        fig24_ax2.plot(data_dict["index"], corr2140dD, "r-")
        fig24_ax3.plot(data_dict["index"], corr2140d17o, "r-")

    ##### CALCULATE corrected VALUES FOR STANDARDS IN VALCO
    corrstdd18o = []
    stdevcorrstdd18o = []
    corrstddD = []
    stdevcorrstddD = []
    corrstd2140d18o = []
    stdevcorrstd2140d18o = []
    corrstd2140dD = []
    stdevcorrstd2140dD = []
    corrstd2140d17o = []
    stdevcorrstd2140d17o = []

    for i in amtransbegin2130d18o:
        begin = i-70
        end = i
        corrstdd18o.append(np.mean(corrd18o[begin:end]))                            
        stdevcorrstdd18o.append(np.std(corrd18o[begin:end]))
        corrstddD.append(np.mean(corrdD[begin:end]))
        stdevcorrstddD.append(np.std(corrdD[begin:end]))
    for i in pmtransbegin2130d18o:
        begin = i-70
        end = i
        corrstdd18o.append(np.mean(corrd18o[begin:end]))                            
        stdevcorrstdd18o.append(np.std(corrd18o[begin:end]))                            
        corrstddD.append(np.mean(corrdD[begin:end]))                            
        stdevcorrstddD.append(np.std(corrdD[begin:end]))                            
    
    if corename == 'EGRIP' and d17oflag == 1:                        
        for i in amtransbegin2140d18o:
            begin = i-70
            end = i
            corrstd2140d18o.append(np.mean(corr2140d18o[begin:end]))                            
            stdevcorrstd2140d18o.append(np.std(corr2140d18o[begin:end]))
            corrstd2140dD.append(np.mean(corr2140dD[begin:end]))
            stdevcorrstd2140dD.append(np.std(corr2140dD[begin:end]))
            corrstd2140d17o.append(np.mean(corr2140d17o[begin:end]))                            
            stdevcorrstd2140d17o.append(np.std(corr2140d17o[begin:end]))
        for i in pmtransbegin2140d18o:
            begin = i-70
            end = i
            corrstd2140d18o.append(np.mean(corr2140d18o[begin:end]))                            
            stdevcorrstd2140d18o.append(np.std(corr2140d18o[begin:end]))
            corrstd2140dD.append(np.mean(corr2140dD[begin:end]))
            stdevcorrstd2140dD.append(np.std(corr2140dD[begin:end]))
            corrstd2140d17o.append(np.mean(corr2140d17o[begin:end]))                            
            stdevcorrstd2140d17o.append(np.std(corr2140d17o[begin:end]))

    ##### ASSIGN corrstd VALUES ####################################
    diffam2pmd18o = np.arange(4)
    diffam2pmdD = np.arange(4)
    
    if corename == 'WAIS06AData' or corename == 'EGRIP': 
        trapd18o = kgwd18o
        trapdD = kgwdD
        trapd17o = kgwd17o
    #    # KBW, KAW, KGW(trap), KPW
    #    knownd18o = [kbwd18o, kawd18o, kpwd18o]       
    #    knowndD   = [kbwdD, kawdD, kpwdD]         

    if corename == 'DYE3': 
        trapd18o = kpwd18o
        trapdD = kpwdD
        trapd17o = kpwd17o
    #    # KAW, KGW, KPW(trap), VW1F
    #    knownd18o = [kawd18o, kgwd18o, vw1fd18o]       
    #    knowndD   = [kawdD, kgwdD, vw1fdD] 

    #if corename == 'DYE3' and d17oflag == 1: 
    #    # KAW, UWWW, KPW(trap), VW1F
    #    knownd18o = [kawd18o, uwww18o, vw1fd18o]       
    #    knowndD   = [kawdD, uwwwdD, vw1fdD]
    #    knownd17o = [kawd17o, uwwwd17o, vw1fd17o]
    #    knownD17o = [kawD17o, uwwwD17o, vw1fD17o]

    ##### ASSIGN corr VALUES ############################################    
    ## Assign first standard measured/corr d18o values
    corrfirstd18o = [corrstdd18o[1],corrstdd18o[7],corrstdd18o[9],corrstdd18o[15]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrfirstd18o = [corrstdd18o[1],corrstdd18o[7],corrstdd18o[9]]
    meancorrfirstd18o = np.mean(corrfirstd18o)
    stdevcorrfirstd18o = np.std(corrfirstd18o)
    diffcorrfirstd18o = knownd18o[0] - meancorrfirstd18o
    diffam2pmd18o[0] = np.mean(corrfirstd18o[0:1])-np.mean(corrfirstd18o[2:3])  
#    extrapfirstd18o = [extrapolatedd18o[0],extrapolatedd18o[6],extrapolatedd18o[8],extrapolatedd18o[14]]
#    meanextrapfirstd18o = np.mean(extrapfirstd18o)
#    stdevextrapfirstd18o = np.std(extrapfirstd18o)
    ## Assign first standard measured/corr dD values
    corrfirstdD = [corrstddD[1],corrstddD[7],corrstddD[9],corrstddD[15]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrfirstdD = [corrstddD[1],corrstddD[7],corrstddD[9]]
    meancorrfirstdD = np.mean(corrfirstdD)
    stdevcorrfirstdD = np.std(corrfirstdD)
    diffcorrfirstdD = knowndD[0] - meancorrfirstdD
    diffam2pmdD[0] = np.mean(corrfirstdD[0:1])-np.mean(corrfirstdD[2:3])  
#    extrapfirstdD = [extrapolateddD[0],extrapolateddD[6],extrapolateddD[8],extrapolateddD[14]]
#    meanextrapfirstdD = np.mean(extrapfirstdD)
#    stdevextrapfirstdD = np.std(extrapfirstdD)

    ##### Assign second measured d18o values
    corrsecondd18o = [corrstdd18o[2],corrstdd18o[6],corrstdd18o[10],corrstdd18o[14]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrsecondd18o = [corrstdd18o[2],corrstdd18o[6],corrstdd18o[10]]
    meancorrsecondd18o = np.mean(corrsecondd18o)
    stdevcorrsecondd18o = np.std(corrsecondd18o)
    diffcorrsecondd18o = knownd18o[1] - meancorrsecondd18o
    diffam2pmd18o[1] = np.mean(corrsecondd18o[0:1])-np.mean(corrsecondd18o[2:3])
#    extrapsecondd18o = [extrapolatedd18o[1],extrapolatedd18o[5],extrapolatedd18o[9],extrapolatedd18o[13]]
#    meanextrapsecondd18o = np.mean(extrapsecondd18o)
#    stdevextrapsecondd18o = np.std(extrapsecondd18o)
    ##### Assign second measured dD values
    corrseconddD = [corrstddD[2],corrstddD[6],corrstddD[10],corrstddD[14]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrseconddD = [corrstddD[2],corrstddD[6],corrstddD[10]]
    meancorrseconddD = np.mean(corrseconddD)
    stdevcorrseconddD = np.std(corrseconddD)
    diffcorrseconddD = knowndD[1] - meancorrseconddD
    diffam2pmdD[1] = np.mean(corrseconddD[0:1])-np.mean(corrseconddD[2:3])
#    extrapseconddD = [extrapolateddD[1],extrapolateddD[5],extrapolateddD[9],extrapolateddD[13]]
#    meanextrapseconddD = np.mean(extrapseconddD)
#    stdevextrapseconddD = np.std(extrapseconddD)

    ## Assign third measured d18o values
    corrthirdd18o = [corrstdd18o[3],corrstdd18o[5],corrstdd18o[11],corrstdd18o[13]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrthirdd18o = [corrstdd18o[3],corrstdd18o[5],corrstdd18o[11]]
    meancorrthirdd18o = np.mean(corrthirdd18o)
    stdevcorrthirdd18o = np.std(corrthirdd18o)
    diffcorrthirdd18o = trapd18o - meancorrthirdd18o
    diffam2pmd18o[2] = np.mean(corrthirdd18o[0:1])-np.mean(corrthirdd18o[2:3])
#    extrapthirdd18o = [extrapolatedd18o[2],extrapolatedd18o[4],extrapolatedd18o[10],extrapolatedd18o[12]]
#    meanextrapthirdd18o = np.mean(extrapthirdd18o)
#    stdevextrapthirdd18o = np.std(extrapthirdd18o)
    ## Assign third measured dD values
    corrthirddD = [corrstddD[3],corrstddD[5],corrstddD[11],corrstddD[13]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrthirddD = [corrstddD[3],corrstddD[5],corrstddD[11]]
    meancorrthirddD = np.mean(corrthirddD)
    stdevcorrthirddD = np.std(corrthirddD)
    diffcorrthirddD = trapdD - meancorrthirddD
    diffam2pmdD[2] = np.mean(corrthirddD[0:1])-np.mean(corrthirddD[2:3])
#    extrapthirddD = [extrapolateddD[2],extrapolateddD[4],extrapolateddD[10],extrapolateddD[12]]
#    meanextrapthirddD = np.mean(extrapthirddD)
#    stdevextrapthirddD = np.std(extrapthirddD)

    ##### Assign fourth measured d18o values
    corrfourthd18o = [corrstdd18o[4],corrstdd18o[12]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrfourthd18o = [corrstdd18o[4],corrstdd18o[4]]
    meancorrfourthd18o = np.mean(corrfourthd18o)
    stdevcorrfourthd18o = np.std(corrfourthd18o)
    diffcorrfourthd18o = knownd18o[2] - meancorrfourthd18o
    diffam2pmd18o[3] = corrfourthd18o[0]-corrfourthd18o[1]
#    extrapfourthd18o = [extrapolatedd18o[3],extrapolatedd18o[11]]
#    meanextrapfourthd18o = np.mean(extrapfourthd18o)
#    stdevextrapfourthd18o = np.std(extrapfourthd18o)
    ##### Assign fourth measured dD values
    corrfourthdD = [corrstddD[4],corrstddD[12]]
    if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
        corrfourthdD = [corrstddD[4],corrstddD[4]]
    meancorrfourthdD = np.mean(corrfourthdD)
    stdevcorrfourthdD = np.std(corrfourthdD)
    diffcorrfourthdD = knowndD[2] - meancorrfourthdD
    diffam2pmdD[3] = corrfourthdD[0]-corrfourthdD[1]
#    extrapfourthdD = [extrapolateddD[3],extrapolateddD[11]]
#    meanextrapfourthdD = np.mean(extrapfourthdD)
#    stdevextrapfourthdD = np.std(extrapfourthdD)
    
    if verbose ==1:
        print "first standard corr d18o valco values"
        print "known values KAW -30.30, -239.13"
        print corrfirstd18o
        print "average", meancorrfirstd18o, "standard deviation", stdevcorrfirstd18o
#        print meanextrapfirstd18o, stdevextrapfirstd18o
        print corrfirstdD
        print "average", meancorrfirstdD, "standard deviation", stdevcorrfirstdD
#        print meanextrapfirstdD, stdevextrapfirstdD
        print "second standard corr d18o valco values"
        print "known values UW-WW-33.82, -268.30"
        print corrsecondd18o
        print "average", meancorrsecondd18o, "standard deviation", stdevcorrsecondd18o
#        print meanextrapsecondd18o, stdevextrapsecondd18o
        print corrseconddD
        print "average", meancorrseconddD, "standard deviation", stdevcorrseconddD
#        print meanextrapseconddD, stdevextrapseconddD
        print "third standard corr d18o valco values"
        print "known values KPW -45.41, -355.18"
        print corrthirdd18o
        print "average", meancorrthirdd18o, "standard deviation", stdevcorrthirdd18o
#        print meanextrapthirdd18o, stdevextrapthirdd18o
        print corrthirddD
        print "average", meancorrthirddD, "standard deviation", stdevcorrthirddD
#        print meanextrapthirddD, stdevextrapthirddD
        print "fourth corr d18o valco values"
        print "known values VW1-F -56.59, -438.43"
        print corrfourthd18o
        print "average", meancorrfourthd18o, "standard deviation", stdevcorrfourthd18o
#        print meanextrapfourthd18o, stdevextrapfourthd18o
        print corrfourthdD
        print "average", meancorrfourthdD, "standard deviation", stdevcorrfourthdD
#        print meanextrapfourthdD, stdevextrapfourthdD
    
    if corename == 'EGRIP' and d17oflag == 1:
        diffam2pm2140d18o = np.arange(4)
        diffam2pm2140dD = np.arange(4)
        diffam2pm2140d17o = np.arange(4)
        
        ## Assign first standard measured/corr 2140d18o values
        corrfirst2140d18o = [corrstd2140d18o[1],corrstd2140d18o[7],corrstd2140d18o[9],corrstd2140d18o[15]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfirst2140d18o = [corrstd2140d18o[1],corrstd2140d18o[7],corrstd2140d18o[9]]
        meancorrfirst2140d18o = np.mean(corrfirst2140d18o)
        stdevcorrfirst2140d18o = np.std(corrfirst2140d18o)
        diffcorrfirst2140d18o = knownd18o[0] - meancorrfirst2140d18o
        diffam2pm2140d18o[0] = np.mean(corrfirst2140d18o[0:1])-np.mean(corrfirst2140d18o[2:3])  
        ## Assign first standard measured/corr 2140dD values
        corrfirst2140dD = [corrstd2140dD[1],corrstd2140dD[7],corrstd2140dD[9],corrstd2140dD[15]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfirst2140dD = [corrstd2140dD[1],corrstd2140dD[7],corrstd2140dD[9]]
        meancorrfirst2140dD = np.mean(corrfirst2140dD)
        stdevcorrfirst2140dD = np.std(corrfirst2140dD)
        diffcorrfirst2140dD = knowndD[0] - meancorrfirst2140dD
        diffam2pm2140dD[0] = np.mean(corrfirst2140dD[0:1])-np.mean(corrfirst2140dD[2:3])  
        ## Assign first standard measured/corr 2140d17o values
        corrfirst2140d17o = [corrstd2140d17o[1],corrstd2140d17o[7],corrstd2140d17o[9],corrstd2140d17o[15]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfirst2140d17o = [corrstd2140d17o[1],corrstd2140d17o[7],corrstd2140d17o[9]]
        meancorrfirst2140d17o = np.mean(corrfirst2140d17o)
        stdevcorrfirst2140d17o = np.std(corrfirst2140d17o)
        diffcorrfirst2140d17o = knownd17o[0] - meancorrfirst2140d17o
        diffam2pm2140d17o[0] = np.mean(corrfirst2140d17o[0:1])-np.mean(corrfirst2140d17o[2:3])  
        
        ##### Assign second measured 2140d18o values
        corrsecond2140d18o = [corrstd2140d18o[2],corrstd2140d18o[6],corrstd2140d18o[10],corrstd2140d18o[14]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrsecond2140d18o = [corrstd2140d18o[2],corrstd2140d18o[6],corrstd2140d18o[10]]
        meancorrsecond2140d18o = np.mean(corrsecond2140d18o)
        stdevcorrsecond2140d18o = np.std(corrsecond2140d18o)
        diffcorrsecond2140d18o = knownd18o[1] - meancorrsecond2140d18o
        diffam2pm2140d18o[1] = np.mean(corrsecond2140d18o[0:1])-np.mean(corrsecond2140d18o[2:3])
        ##### Assign second measured 2140dD values
        corrsecond2140dD = [corrstd2140dD[2],corrstd2140dD[6],corrstd2140dD[10],corrstd2140dD[14]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrsecond2140dD = [corrstd2140dD[2],corrstd2140dD[6],corrstd2140dD[10]]
        meancorrsecond2140dD = np.mean(corrsecond2140dD)
        stdevcorrsecond2140dD = np.std(corrsecond2140dD)
        diffcorrsecond2140dD = knowndD[1] - meancorrsecond2140dD
        diffam2pm2140dD[1] = np.mean(corrsecond2140dD[0:1])-np.mean(corrsecond2140dD[2:3])
        ##### Assign second measured 2140d17o values
        corrsecond2140d17o = [corrstd2140d17o[2],corrstd2140d17o[6],corrstd2140d17o[10],corrstd2140d17o[14]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrsecond2140d17o = [corrstd2140d17o[2],corrstd2140d17o[6],corrstd2140d17o[10]]
        meancorrsecond2140d17o = np.mean(corrsecond2140d17o)
        stdevcorrsecond2140d17o = np.std(corrsecond2140d17o)
        diffcorrsecond2140d17o = knownd17o[1] - meancorrsecond2140d17o
        diffam2pm2140d17o[1] = np.mean(corrsecond2140d17o[0:1])-np.mean(corrsecond2140d17o[2:3])
        
        ## Assign third measured 2140d18o values
        corrthird2140d18o = [corrstd2140d18o[3],corrstd2140d18o[5],corrstd2140d18o[11],corrstd2140d18o[13]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrthird2140d18o = [corrstd2140d18o[3],corrstd2140d18o[5],corrstd2140d18o[11]]
        meancorrthird2140d18o = np.mean(corrthird2140d18o)
        stdevcorrthird2140d18o = np.std(corrthird2140d18o)
        diffcorrthird2140d18o = trapd18o - meancorrthird2140d18o
        diffam2pm2140d18o[2] = np.mean(corrthird2140d18o[0:1])-np.mean(corrthird2140d18o[2:3])
        ## Assign third measured 2140dD values
        corrthird2140dD = [corrstd2140dD[3],corrstd2140dD[5],corrstd2140dD[11],corrstd2140dD[13]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrthird2140dD = [corrstd2140dD[3],corrstd2140dD[5],corrstd2140dD[11]]
        meancorrthird2140dD = np.mean(corrthird2140dD)
        stdevcorrthird2140dD = np.std(corrthird2140dD)
        diffcorrthird2140dD = trapdD - meancorrthird2140dD
        diffam2pm2140dD[2] = np.mean(corrthird2140dD[0:1])-np.mean(corrthird2140dD[2:3])
        ## Assign third measured 2140d17o values
        corrthird2140d17o = [corrstd2140d17o[3],corrstd2140d17o[5],corrstd2140d17o[11],corrstd2140d17o[13]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrthird2140d17o = [corrstd2140d17o[3],corrstd2140d17o[5],corrstd2140d17o[11]]
        meancorrthird2140d17o = np.mean(corrthird2140d17o)
        stdevcorrthird2140d17o = np.std(corrthird2140d17o)
        diffcorrthird2140d17o = trapd17o - meancorrthird2140d17o
        diffam2pm2140d17o[2] = np.mean(corrthird2140d17o[0:1])-np.mean(corrthird2140d17o[2:3])

        ##### Assign fourth measured 2140d18o values
        corrfourth2140d18o = [corrstd2140d18o[4],corrstd2140d18o[12]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfourth2140d18o = [corrstd2140d18o[4],corrstd2140d18o[4]]
        meancorrfourth2140d18o = np.mean(corrfourth2140d18o)
        stdevcorrfourth2140d18o = np.std(corrfourth2140d18o)
        diffcorrfourth2140d18o = knownd18o[2] - meancorrfourth2140d18o
        diffam2pm2140d18o[3] = corrfourth2140d18o[0]-corrfourth2140d18o[1]
        ##### Assign fourth measured 2140dD values
        corrfourth2140dD = [corrstd2140dD[4],corrstd2140dD[12]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfourth2140dD = [corrstd2140dD[4],corrstd2140dD[4]]
        meancorrfourth2140dD = np.mean(corrfourth2140dD)
        stdevcorrfourth2140dD = np.std(corrfourth2140dD)
        diffcorrfourth2140dD = knowndD[2] - meancorrfourth2140dD
        diffam2pm2140dD[3] = corrfourth2140dD[0]-corrfourth2140dD[1]
        ##### Assign fourth measured 2140d17o values
        corrfourth2140d17o = [corrstd2140d17o[4],corrstd2140d17o[12]]
        if filename == "rawHIDS2038-20160729-000006Z-DataLog_User.dat":
            corrfourth2140d17o = [corrstd2140d17o[4],corrstd2140d17o[4]]
        meancorrfourth2140d17o = np.mean(corrfourth2140d17o)
        stdevcorrfourth2140d17o = np.std(corrfourth2140d17o)
        diffcorrfourth2140d17o = knownd18o[2] - meancorrfourth2140d17o
        diffam2pm2140d17o[3] = corrfourth2140d17o[0]-corrfourth2140d17o[1]

        if verbose ==1:
            print "first standard corr 2140 valco values"
            print "known values KAW -30.30, -239.13, 17, -16.11"
            print corrfirst2140d18o
            print "average", meancorrfirst2140d18o, "standard deviation", stdevcorrfirst2140d18o
            print corrfirst2140dD
            print "average", meancorrfirst2140dD, "standard deviation", stdevcorrfirst2140dD
            print corrfirst2140d17o
            print "average", meancorrfirst2140d17o, "standard deviation", stdevcorrfirst2140d17o
            print "second standard corr 2140 valco values"
            print "known values UW-WW -33.82, -268.30, 27, -18.00"
            print corrsecond2140d18o
            print "average", meancorrsecond2140d18o, "standard deviation", stdevcorrsecond2140d18o
            print corrsecond2140dD
            print "average", meancorrsecond2140dD, "standard deviation", stdevcorrsecond2140dD
            print corrsecond2140d17o
            print "average", meancorrsecond2140d17o, "standard deviation", stdevcorrsecond2140d17o
            print "third standard corr 2140o valco values"
            print "known values KPW -45.41, -355.18, 6, -24.24"
            print corrthird2140d18o
            print "average", meancorrthird2140d18o, "standard deviation", stdevcorrthird2140d18o
            print corrthird2140dD
            print "average", meancorrthird2140dD, "standard deviation", stdevcorrthird2140dD
            print corrthird2140d17o
            print "average", meancorrthird2140d17o, "standard deviation", stdevcorrthird2140d17o       
            print "fourth corr 2140 valco values"
            print "known values VW1-F -56.59, -438.43, 7, -30.29"
            print corrfourth2140d18o
            print "average", meancorrfourth2140d18o, "standard deviation", stdevcorrfourth2140d18o
            print corrfourth2140dD
            print "average", meancorrfourth2140dD, "standard deviation", stdevcorrfourth2140dD
            print corrfourth2140d17o
            print "average", meancorrfourth2140d17o, "standard deviation", stdevcorrfourth2140d17o
           
    ##### Assign KBW measured isotope values #######################################    
    if corename == "WAIS06AData":
        corrstdkbwd18o = corrfirstd18o
        corrstdkbwdD = corrfirstdD                                       
        meankbwd18o = meancorrfirstd18o
        stdkbwd18o = stdevcorrfirstd18o
        diffkbwd18o = kbwd18o - meankbwd18o
        meankbwdD = meancorrfirstdD
        stdkbwdD = stdevcorrfirstdD
        diffkbwdD = kbwdD - meankbwdD

    ##### Assign KAW measured isotope values #######################################    
    if corename == "WAIS06AData" or corename == 'DYE3':
        corrstdkawd18o = corrsecondd18o
        corrstdkawdD = corrseconddD
    if corename == "EGRIP":
        corrstdkawd18o = corrfirstd18o
        corrstdkawdD = corrfirstdD                                      
    meankawd18o = np.mean(corrstdkawd18o)
    stdkawd18o = np.std(corrstdkawd18o)
    diffkawd18o = kawd18o - meankawd18o
    meankawdD = np.mean(corrstdkawdD)
    stdkawdD = np.std(corrstdkawdD)
    diffkawdD = kawdD - meankawdD
    
    ##### Assign kgw or uwww measured isotope values #######################################    
    if corename == "WAIS06AData" or corename == 'DYE3':
        corrstdkgwd18o = corrthirdd18o
        corrstdkgwdD = corrthirddD
        meankgwd18o = np.mean(corrstdkgwd18o)
        stdkgwd18o = np.std(corrstdkgwd18o)
        diffkgwd18o = kgwd18o - meankgwd18o
        meankgwdD = np.mean(corrstdkgwdD)
        stdkgwdD = np.std(corrstdkgwdD)
        diffkgwdD = kgwdD - meankgwdD
    if corename == "EGRIP" and d17oflag ==0:
        corrstdkgwd18o = corrsecondd18o
        corrstdkgwdD = corrseconddD
        meankgwd18o = np.mean(corrstdkgwd18o)
        stdkgwd18o = np.std(corrstdkgwd18o)
        diffkgwd18o = kgwd18o - meankgwd18o
        meankgwdD = np.mean(corrstdkgwdD)
        stdkgwdD = np.std(corrstdkgwdD)
        diffkgwdD = kgwdD - meankgwdD
    if corename == "EGRIP" and d17oflag ==1:  
        corrstduwwwd18o = corrsecondd18o
        corrstduwwwdD = corrseconddD
        meanuwwwd18o = np.mean(corrstduwwwd18o)
        stduwwwd18o = np.std(corrstduwwwd18o)
        diffuwwwd18o = uwwwd18o - meanuwwwd18o
        meanuwwwdD = np.mean(corrstduwwwdD)
        stduwwwdD = np.std(corrstduwwwdD)
        diffuwwwdD = uwwwdD - meanuwwwdD

    ## Assign kpw measured isotope values ########################################## 
    if corename == "WAIS06AData" or corename == 'DYE3':
        corrstdkpwd18o = corrfourthd18o
        corrstdkpwdD = corrfourthdD
        meankpwd18o = np.mean(corrstdkpwd18o)
        stdkpwd18o = np.std(corrstdkpwd18o)
        diffkpwd18o = kpwd18o - meankpwd18o
        meankpwdD = np.mean(corrstdkpwdD)
        stdkpwdD = np.std(corrstdkpwdD)
        diffkpwdD = kpwdD - meankpwdD
    if corename == "EGRIP":
        corrstdkpwd18o = corrthirdd18o
        corrstdkpwdD = corrthirddD
        meankpwd18o = np.mean(corrstdkpwd18o)
        stdkpwd18o = np.std(corrstdkpwd18o)
        diffkpwd18o = kpwd18o - meankpwd18o
        meankpwdD = np.mean(corrstdkpwdD)
        stdkpwdD = np.std(corrstdkpwdD)
        diffkpwdD = kpwdD - meankpwdD

    ##### Assign vw1f measured isotope values ####################################### 
    if corename == "EGRIP":
        corrstdvw1fd18o = corrfourthd18o
        corrstdvw1fdD = corrfourthdD
        meanvw1fd18o = np.mean(corrstdvw1fd18o)
        stdvw1fd18o = np.std(corrstdvw1fd18o)
        diffvw1fd18o = vw1fd18o - meanvw1fd18o
        meanvw1fdD = np.mean(corrstdvw1fdD)
        stdvw1fdD = np.std(corrstdvw1fdD)
        diffvw1fdD = vw1fdD - meanvw1fdD
        
    #### Collect drift values from comparing am to pm valcos
    avedriftstdd18o = np.mean(diffam2pmd18o)
    stdevdriftstdd18o = np.std(diffam2pmd18o)
    avedriftstddD = np.mean(diffam2pmdD)
    stdevdriftstddD = np.std(diffam2pmdD)
    if corename == 'DYE3' and d17oflag == 1:
        avedriftstd2140d18o = np.mean(diffam2pm2140d18o)
        stdevdriftstd2140d18o = np.std(diffam2pm2140d18o)
        avedriftstd2140dD = np.mean(diffam2pm2140dD)
        stdevdriftstd2140dD = np.std(diffam2pm2140dD)
        avedriftstd2140d17o = np.mean(diffam2pm2140d17o)
        stdevdriftstd2140d17o = np.std(diffam2pm2140d17o)

    ##### ICE CORE IDENTIFICATION ##################################################
    ## Look at comments to decided on number of cores, location of cores, length of cores
    beginmelt = [x for x in data_dict["index"][1:] if data_dict["comments"][x-1]!=175 and data_dict["comments"][x]==175]
    begincores = [x for x in data_dict["index"][1:-20] if data_dict["comments"][x+20]==175 and \
        data_dict["start_depth"][x-1]!=data_dict["start_depth"][x]] #comment out last conditional for 20120215
    coreindex = np.arange(len(begincores))
    for x in coreindex:
        begincores[x] = begincores[x] + 5
    startdepth = data_dict["start_depth"][begincores]
    enddepth = data_dict["end_depth"][begincores]
    coredatalength = np.arange(len(begincores))
    for i in coreindex:
        begincores[i] = begincores[i] - 7
        coredataindex = [x for x in data_dict["index"][1:] if data_dict["start_depth"][x]==startdepth[i] and data_dict["comments"][x]==175]
        coredatalength[i] = len(coredataindex)
    startepoch = data_dict["epoch"][begincores]
    numbeginmelt = len(beginmelt)
    endmelt = [x for x in data_dict["index"][1:] if data_dict["comments"][x-1]==175 and data_dict["comments"][x]!=175]
    meltend = []
    for i in endmelt: meltend.append(i-10)
    numendmelt = len(endmelt)
    if verbose ==1:
        print "Beginning of melts indices ", beginmelt
        print "Start depths ", startdepth
        print "Start time in epoch ", startepoch
        print "Number of melts begun ", numbeginmelt
        print "Ending of melts indices ", endmelt
        print "End depths ", enddepth
        print "Number of melts ended ", numendmelt
        
    meltlength = map(lambda x,y: x-y, endmelt, beginmelt)
    meltlengthplus = []
    for i in meltlength: meltlengthplus.append(i+500)

    if verbose ==1:
        print "index length of each melt ", meltlength
        
    ## Write 2130 raw data dictionary to file for future reprocessing
    #dataout_file = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/raw_dictionaries/raw" + filename
    dataout_file = "/Users/frio/EGRIP_2023/Data/raw_dictionaries/raw" + filename
    file = open(dataout_file, "w")   
    pickle.dump(data_dict, file)
    file.close()  
    ## Write 2140 raw data dictionary to file for future reprocessing
    if corename == 'EGRIP' and d17oflag == 1:
        #d17odataout_file = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/2140_raw_dictionaries/rawd17o" + filename
        d17odataout_file = "/Users/frio/EGRIP_2023/Data/2140_raw_dictionaries/rawd17o" + filename
        d17ofile = open(d17odataout_file, "w")   
        pickle.dump(d17odata_dict, d17ofile)
        d17ofile.close()  

    ##### ASSIGN DEPTH, EC TO ISOTOPES AND SAVE FILES #######################    
    ##### Cut out and assign isotopes, ec, and depth to each core ###########
    meltindex = np.arange(len(beginmelt))
    ice_dict ={}
    ice_dict["filepath"] = np.array(()).astype("S")
    ice_dict["time"] = np.array(())
    ice_dict["d18o"] = np.array(())
    ice_dict["errorpred18o"] = np.array(())
    ice_dict["erroraccd18o"] = np.array(())
    ice_dict["rawd18o"] = np.array(())
    ice_dict["dD"] = np.array(())
    ice_dict["errorpredD"] = np.array(())
    ice_dict["erroraccdD"] = np.array(())
    ice_dict["rawdD"] = np.array(())
    ice_dict["2140d18o"] = np.array(())
    ice_dict["errorpre2140d18o"] = np.array(())
    ice_dict["erroracc2140d18o"] = np.array(())
    ice_dict["raw2140d18o"] = np.array(())
    ice_dict["2140dD"] = np.array(())
    ice_dict["errorpre2140dD"] = np.array(())
    ice_dict["erroracc2140dD"] = np.array(())
    ice_dict["raw2140dD"] = np.array(())
    ice_dict["2140d17o"] = np.array(())
    ice_dict["errorpre2140d17o"] = np.array(())
    ice_dict["erroracc2140d17o"] = np.array(())
    ice_dict["raw2140d17o"] = np.array(())
    ice_dict["d_excess"] =np.array(())
    ice_dict["water"] = np.array(())
    ice_dict["ec"] = np.array(())
    ice_dict["depth"] = np.array(())
    ice_dict["meltrate"] = np.array(())
    ice_dict["flag1_d18o"] = np.array(()).astype("S")
    ice_dict["flag1_dD"] = np.array(()).astype("S")
    ice_dict["flag1_2140_d18o"] = np.array(()).astype("S")
    ice_dict["flag1_2140_dD"] = np.array(()).astype("S")
    ice_dict["flag1_2140_d17o"] = np.array(()).astype("S")
    ice_dict["flag2"] = np.array(()).astype("S")
    ice_dict["flag3"] = np.array(()).astype("S")
    ice_dict["flag4_d18o"] = np.array(()).astype("S")
    ice_dict["flag4_dD"] = np.array(()).astype("S")
    ice_dict["flag4_2140_d18o"] = np.array(()).astype("S") 
    ice_dict["flag4_2140_dD"] = np.array(()).astype("S")
    ice_dict["flag4_2140_d17o"] = np.array(()).astype("S")
    ice_dict["flag5"] = np.array(()).astype("S")
    ice_dict["flag6"] = np.array(()).astype("S")
    ice_dict["flag7_d18o"] = np.array(()).astype("S")
    ice_dict["flag7_dD"] = np.array(()).astype("S")
    ice_dict["flag7_2140_d18o"] = np.array(()).astype("S")
    ice_dict["flag7_2140_dD"] = np.array(()).astype("S")
    ice_dict["flag7_2140_d17o"] = np.array(()).astype("S")
    ice_dict["flag8"] = np.array(()).astype("S")
    ice_dict["ave_valco_normsigma_d18o"] = np.array(())
    ice_dict["ave_valco_normsigma_dD"] = np.array(())
    ice_dict["ave_valco_normsigma_2140d18o"] = np.array(())
    ice_dict["ave_valco_normsigma_2140dD"] = np.array(())
    ice_dict["ave_valco_normsigma_2140d17o"] = np.array(())
    ice_dict["ave_valco_skewsigma_d18o"] = np.array(())
    ice_dict["ave_valco_skewsigma_dD"] = np.array(())
    ice_dict["ave_valco_skewsigma_2140d18o"] = np.array(())
    ice_dict["ave_valco_skewsigma_2140dD"] = np.array(())
    ice_dict["ave_valco_skewsigma_2140d17o"] = np.array(())
    ice_dict["ave_nea_normsigma_d18o"] = np.array(())
    ice_dict["ave_nea_normsigma_dD"] = np.array(())
    ice_dict["ave_nea_normsigma_2140d18o"] = np.array(())
    ice_dict["ave_nea_normsigma_2140dD"] = np.array(())
    ice_dict["ave_nea_normsigma_2140d17o"] = np.array(())
    ice_dict["ave_nea_skewsigma_d18o"] = np.array(())
    ice_dict["ave_nea_skewsigma_dD"] = np.array(())
    ice_dict["ave_nea_skewsigma_2140d18o"] = np.array(())
    ice_dict["ave_nea_skewsigma_2140dD"] = np.array(())
    ice_dict["ave_nea_skewsigma_2140d17o"] = np.array(())
    ice_dict["prodate"] = np.array(())
    ice_dict["crunchversion"] = np.array(())

    
    ##### Create memory arrays before loop
    memcorrd18o = deepcopy(corrd18o)
    memcorrdD = deepcopy(corrdD)
    memcorr2140d18o = deepcopy(corr2140d18o)
    memcorr2140dD = deepcopy(corr2140dD)
    memcorr2140d17o = deepcopy(corr2140d17o)

    for i in meltindex:
        startmelt = beginmelt[i]+5
        endmelt = startmelt + meltlength[i]-10
        if endmelt >= data_dict["index"][-1]:
            endmelt = data_dict["index"][-1]
        endmeltplus = startmelt + meltlengthplus[i]
        if endmeltplus >= data_dict["index"][-1]:
            endmeltplus = data_dict["index"][-1]  #put in for cut files that had binary shifts half way through the day
        core = data_dict["index"][startmelt:endmelt]
        
        ##### FIND THE MINIUMS, AND ASSIGN BEGINNING AND END OF CORES ##############
        startisod18o = np.min(diffd18o[startmelt:startmelt+550])
        endisod18o = np.max(diffd18o[endmelt:endmeltplus])
        startisoindexd18o = [x for x in data_dict["index"][startmelt:endmelt] if diffd18o[x]==startisod18o] # by first derivative, could be from mid point if was saved in an array...
        startisoindexd18o = startisoindexd18o[0]
        if filepath =='/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180730-000005Z-DataLog_User.dat' and i==3:
        	startisoindexd18o = 83312
        endisoindexd18o = [x for x in data_dict["index"][startmelt:endmeltplus] if diffd18o[x]==endisod18o]
        endisoindexd18o = endisoindexd18o[0]
        
        startisodD = np.min(diffdD[startmelt:startmelt+550])
        endisodD = np.max(diffdD[endmelt:endmeltplus]) #    endiso = np.max(diffdD[startmelt:endmeltplus])
        startisoindexdD = [x for x in data_dict["index"][startmelt:endmelt] if diffdD[x]==startisodD] # by first derivative, could be from mid point if was saved in an array...
        startisoindexdD = startisoindexdD[0]
        if filepath =='/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180730-000005Z-DataLog_User.dat' and i==3:
        	startisoindexdD = 83312
        endisoindexdD = [x for x in data_dict["index"][startmelt:endmeltplus] if diffdD[x]==endisodD]
        endisoindexdD = endisoindexdD[0]

        startiso2140d18o = np.min(diff2140d18o[startmelt:startmelt+550])
        endiso2140d18o = np.max(diff2140d18o[endmelt:endmeltplus])
        startisoindex2140d18o = [x for x in data_dict["index"][startmelt:endmelt] if diff2140d18o[x]==startiso2140d18o] # by first derivative, could be from mid point if was saved in an array...
        startisoindex2140d18o = startisoindex2140d18o[0]
        if filepath =='/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180730-000005Z-DataLog_User.dat' and i==3:
        	startisoindex2140d18o = 83312
        endisoindex2140d18o = [x for x in data_dict["index"][startmelt:endmeltplus] if diff2140d18o[x]==endiso2140d18o]
        endisoindex2140d18o = endisoindex2140d18o[0]
        
        startiso2140dD = np.min(diff2140dD[startmelt:startmelt+550])
        endiso2140dD = np.max(diff2140dD[endmelt:endmeltplus])
        startisoindex2140dD = [x for x in data_dict["index"][startmelt:endmelt] if diff2140dD[x]==startiso2140dD] # by first derivative, could be from mid point if was saved in an array...
        startisoindex2140dD = startisoindex2140dD[0]
        if filepath =='/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180730-000005Z-DataLog_User.dat' and i==3:
        	startisoindex2140dD = 83312
        endisoindex2140dD = [x for x in data_dict["index"][startmelt:endmeltplus] if diff2140dD[x]==endiso2140dD]
        endisoindex2140dD = endisoindex2140dD[0]
        
        startiso2140d17o = np.min(diff2140d17o[startmelt:startmelt+550])
        endiso2140d17o = np.max(diff2140d17o[endmelt:endmeltplus])
        startisoindex2140d17o = [x for x in data_dict["index"][startmelt:endmelt] if diff2140d17o[x]==startiso2140d17o] # by first derivative, could be from mid point if was saved in an array...
        startisoindex2140d17o = startisoindex2140d17o[0]
        if filepath =='/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180730-000005Z-DataLog_User.dat' and i==3:
        	startisoindex2140d17o = 83312
        endisoindex2140d17o = [x for x in data_dict["index"][startmelt:endmeltplus] if diff2140d17o[x]==endiso2140d17o]
        endisoindex2140d17o = endisoindex2140d17o[0]
        
        if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
        	print "fixing startiso issue!"
        	startisoindexd18o = 56317
        	startisoindexdD = 56317
        	startisoindex2140d18o = 56301
        	startisoindex2140dD = 56301
        	startisoindex2140d17o = 56301


        print "start isotopic indexes", startisoindexd18o, startisoindexdD, startisoindex2140d18o, startisoindex2140dD, startisoindex2140d17o
        print "end isotopic indexes", endisoindexd18o, endisoindexdD, endisoindex2140d18o, endisoindex2140dD, endisoindex2140d17o
        
        #Special cases for WAIS06A where the isotopes need to be identified manually
#        if filepath == '/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/raw_dictionaries/rawHBDS92-20110831-0814-Data.dat':
#            startisoall = [2816,5057,7128,9171,11440,13741]
#            endisoall = [4597,6653,8666,10763,13004,15745]
#            startisoindexd18o = startisoall[i]
#            startisoindexdD = startisoall[i]
#            endisoindexd18o = endisoall[i]
#            endisoindexdD = endisoall[i]
        ## if ever for 2140 data, need to rememer to also specify for those 3 isotopes
        
        #### Find 50% of transition for the midpoint versus the startisoindex from the first derivative
        previousd18o = np.mean(corrd18o[startisoindexd18o-250:startisoindexd18o-150])
        postd18o = np.mean(corrd18o[startisoindexd18o+150:startisoindexd18o+250])
        midd18o = (previousd18o+postd18o)/2
        midisopointsd18o = [x for x in data_dict["index"][startisoindexd18o-100:startisoindexd18o+200] if corrd18o[x] >= midd18o-0.5 and corrd18o[x] <= midd18o+0.5]
        if filepath == '/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180708-003343Z-DataLog_User.dat' and i==0:
        	midisopointsd18o = [47419]
        if len(midisopointsd18o) >= 1:
            midisoindexd18o = midisopointsd18o[-1]
        else:
            midisopointsd18o = [x for x in data_dict["index"][startisoindexd18o-100:startisoindexd18o+200] if corrd18o[x+1] <= midd18o and corrd18o[x-1] >= midd18o]
            midisoindexd18o = midisopointsd18o[-1]
        if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
        	midisoindexd18o = startisoindexd18o
        	    
        previousdD = np.mean(corrdD[startisoindexdD-250:startisoindexdD-150])
        postdD = np.mean(corrdD[startisoindexdD+150:startisoindexdD+250])
        middD = (previousdD+postdD)/2
        midisopointsdD = [x for x in data_dict["index"][startisoindexdD-100:startisoindexdD+200] if corrdD[x] >= middD-2 and corrdD[x] <= middD+2]
        if filepath == '/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180708-003343Z-DataLog_User.dat' and i==0:
        	midisopointsdD = [47419]
        if len(midisopointsdD) >= 1:
            midisoindexdD = midisopointsdD[-1]
        else:
            midisopointsdD = [x for x in data_dict["index"][startisoindexdD-100:startisoindexdD+200] if corrdD[x+1] <= middD and corrdD[x-1] >= middD]
            midisoindexdD = midisopointsdD[-1]
        if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
        	midisoindexdD = startisoindexdD
        
            
        print "mid isotopic indexes before matching, d18o then dD", midisoindexd18o, midisoindexdD
        midisoindexd18o = midisoindexdD #to make match for d-excess
        print "mid isotopic indexes", midisoindexd18o, midisoindexdD
        
        if corename == 'EGRIP' and d17oflag == 1:
            previous2140d18o = np.mean(corr2140d18o[startisoindex2140d18o-250:startisoindex2140d18o-150])
            post2140d18o = np.mean(corr2140d18o[startisoindex2140d18o+150:startisoindex2140d18o+250])
            mid2140d18o = (previous2140d18o+post2140d18o)/2
            midisopoints2140d18o = [x for x in data_dict["index"][startisoindex2140d18o-200:startisoindex2140d18o+200] if corr2140d18o[x] >= mid2140d18o-0.5 and corr2140d18o[x] <= mid2140d18o+0.5]
            if filepath == '/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180708-003343Z-DataLog_User.dat' and i==0:
        		midisopoints2140d18o = [47419]
            if len(midisopoints2140d18o) >= 1:
                midisoindex2140d18o = midisopoints2140d18o[-1]
            else:
                midisopoints2140d18o = [x for x in data_dict["index"][startisoindex2140d18o-200:startisoindex2140d18o+200] if corr2140d18o[x+1] <= mid2140d18o and corr2140d18o[x-1] >= mid2140d18o]
                midisoindex2140d18o = midisopoints2140d18o[-1]
            if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
        		midisoindex2140d18o = startisoindex2140d18o
        		
            previous2140dD = np.mean(corr2140dD[startisoindex2140dD-250:startisoindex2140dD-150])
            post2140dD = np.mean(corr2140dD[startisoindex2140dD+150:startisoindex2140dD+250])
            mid2140dD = (previous2140dD+post2140dD)/2
            midisopoints2140dD = [x for x in data_dict["index"][startisoindex2140dD-200:startisoindex2140dD+200] if corr2140dD[x] >= mid2140dD-2 and corr2140dD[x] <= mid2140dD+2]
            if filepath == '/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180708-003343Z-DataLog_User.dat' and i==0:
        		midisopoints2140dD = [47419]
            if len(midisopoints2140dD) >= 1:
                midisoindex2140dD = midisopoints2140dD[-1]
            else:
                midisopoints2140dD = [x for x in data_dict["index"][startisoindex2140dD-200:startisoindex2140dD+200] if corr2140dD[x+1] <= mid2140dD and corr2140dD[x-1] >= mid2140dD]
                midisoindex2140dD = midisopoints2140dD[-1]
            if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
        		midisoindex2140dD = startisoindex2140dD
        		
            previous2140d17o = np.mean(corr2140d17o[startisoindex2140d17o-250:startisoindex2140d17o-150])
            post2140d17o = np.mean(corr2140d17o[startisoindex2140d17o+150:startisoindex2140d17o+250])
            mid2140d17o = (previous2140d17o+post2140d17o)/2
            midisopoints2140d17o = [x for x in data_dict["index"][startisoindex2140d17o-200:startisoindex2140d17o+200] if corr2140d17o[x] >= mid2140d17o-0.3 and corr2140d17o[x] <= mid2140d17o+0.3]
            if filepath == '/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180708-003343Z-DataLog_User.dat' and i==0:
        		midisopoints2140d17o = [47419]
            if len(midisopoints2140d17o) >= 1:
                midisoindex2140d17o = midisopoints2140d17o[-1]
            else:
                midisopoints2140d17o = [x for x in data_dict["index"][startisoindex2140d17o-200:startisoindex2140d17o+200] if corr2140d17o[x+1] <= mid2140d17o and corr2140d17o[x-1] >= mid2140d17o]
                midisoindex2140d17o = midisopoints2140d17o[-1]
            
            if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180606-000007Z-DataLog_User.dat" and i==0:
            	midisoindex2140d17o = startisoindex2140d17o
            if filepath=="/Users/frio/EGRIP_2023/Data/raw_dictionaries/rawHIDS2143-20180607-000007Z-DataLog_User.dat":
            	midisoindex2140d18o = midisoindex2140dD
            	midisoindex2140d17o = midisoindex2140dD
        	
        	print "2140 mid isotopic indexes", midisoindex2140d18o, midisoindex2140dD, midisoindex2140d17o   
            midisoindex2140d17o = midisoindex2140d18o #to make match for D17o   
              
        
        #### Memory application of nea curve to each ice core only when push ice was used inbetween ice cores, valco correction already applied
        #### changed all startisoindex's to midisoindex's on 20170119
        if data_dict["epoch"][0]>=1364947200 and filepath != "/Users/frio/Dropbox/WaterWorld/EGRIP/raw_dictionaries/rawHIDS2038-20160819-000007Z-DataLog_User.dat":
            previousvalued18o = np.mean(corrd18o[startisoindexd18o-350:startisoindexd18o-225])
            previousvaluedD = np.mean(corrdD[startisoindexdD-350:startisoindexdD-225])
            previousvalue2140d18o = np.mean(corr2140d18o[startisoindex2140d18o-350:startisoindex2140d18o-225])
            previousvalue2140dD = np.mean(corr2140dD[startisoindex2140dD-350:startisoindex2140dD-225])
            previousvalue2140d17o = np.mean(corr2140d17o[startisoindex2140d17o-350:startisoindex2140d17o-225])
            for z in data_dict["index"][midisoindexdD:midisoindexdD+200]: #IF bomb out here, check post setting for index2
                memcorrd18o[z] = (corrd18o[z]-previousvalued18o*(1-smavenead18o[z-midisoindexd18o+smavenead18omidpoint]))/(smavenead18o[z-midisoindexd18o+smavenead18omidpoint])
                data_dict["flag7_d18o"][z] = 'n'
                memcorrdD[z] = (corrdD[z]-previousvaluedD*(1-smaveneadD[z-midisoindexdD+smaveneadDmidpoint]))/(smaveneadD[z-midisoindexdD+smaveneadDmidpoint])
                data_dict["flag7_dD"][z] = 'n'
            if corename == 'EGRIP' and d17oflag == 1:
                for z in data_dict["index"][midisoindex2140dD:midisoindex2140dD+200]:
                    memcorr2140d18o[z] = (corr2140d18o[z]-previousvalue2140d18o*(1-smavenea2140d18o[z-midisoindex2140d18o+smavenea2140d18omidpoint]))/(smavenea2140d18o[z-midisoindex2140d18o+smavenea2140d18omidpoint]) 
                    data_dict["flag7_2140_d18o"][z] = 'n'
                    memcorr2140dD[z] = (corr2140dD[z]-previousvalue2140dD*(1-smavenea2140dD[z-midisoindex2140dD+smavenea2140dDmidpoint]))/(smavenea2140dD[z-midisoindex2140dD+smavenea2140dDmidpoint]) 
                    data_dict["flag7_2140_dD"][z] = 'n'
                    memcorr2140d17o[z] = (corr2140d17o[z]-previousvalue2140d17o*(1-smavenea2140d17o[z-midisoindex2140d17o+smavenea2140d17omidpoint]))/(smavenea2140d17o[z-midisoindex2140d17o+smavenea2140d17omidpoint]) 
                    data_dict["flag7_2140_d17o"][z] = 'n'
        memcorrdexcess = memcorrdD-8*memcorrd18o

        ##### DEPTH, FILTER OUT OUTLIERS AND LASER ERRORS, STILL MAY NEED SOME MANUAL EDITING OF RAW FILE TO MAKE WORK DEPENDING ON ERRORS
                
        ## find transition from ice core to ice core
        iceidx = [x for x in data_dict["index"][startmelt-10:endmelt+10] if diff_start_depth[x] != 0]
        depthidx = deepcopy(iceidx)
        for o in np.arange(len(depthidx)):
            depthidx[o] = depthidx[o] + 1
        epoch_marker_icec = data_dict["epoch"][iceidx]
        start_depth_marker_icec = data_dict["start_depth"][depthidx]
        end_depth_marker_icec = data_dict["end_depth"][depthidx]
        diff_laser_marker_icec = diff_laser[iceidx] 
        print end_depth_marker_icec

        ## isolate depth profiles for each ice stick
        lng = len(epoch_marker_icec)
        estop = [x for x in data_dict["index"][startmelt-10:endmelt+10] if data_dict["comments"][x]==175 and data_dict["comments"][x+1]==0]
        if len(estop)==1 and endmelt>= iceidx[-1]: #emergency stop and start and enddepths were not changed
            os.system('say "enter end depth"')
            epoch_marker_icec = np.append(epoch_marker_icec,data_dict["epoch"][endmelt])
            start_depth_marker_icec = np.append(start_depth_marker_icec,data_dict["start_depth"][endmelt])
            end_depth_marker_icec = np.append(end_depth_marker_icec,data_dict["end_depth"][endmelt])
            diff_laser_marker_icec = np.append(diff_laser_marker_icec,diff_laser[endmelt])
            lng = len(epoch_marker_icec)
            ### ask to type in new depth data
            wrong_depth = data_dict["end_depth"][endmelt]
            print "wrong depth", data_dict["end_depth"][endmelt] 
            right_depth = input("Please type in right depth since the melt ended early...")
            print endmelt
            fixtime = data_dict["epoch"][endmelt]
            print fixtime
            fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                data_dict["epoch"][x]<=fixtime]
            for x in fixdepth:
                data_dict["end_depth"][x] = right_depth
            correction = wrong_depth-right_depth
            end_depth_marker_icec = data_dict["end_depth"][depthidx]
            print end_depth_marker_icec
        coredepth = []
        for q in np.arange(lng-1):
            idx_ice = [x for x in data_dict["index"] if data_dict["epoch"][x] >= epoch_marker_icec[q] and \
                  data_dict["epoch"][x] <= epoch_marker_icec[q+1]]
            epoch_icec_idv = data_dict["epoch"][idx_ice]
            laser_distance_icec_idv = data_dict["laser_distance"][idx_ice]
            loadflag = data_dict["loadflag"][idx_ice]
            diff_laser_icec_idv = np.diff(laser_distance_icec_idv)
            
            #check end depth is greater than start depth
            if end_depth_marker_icec[q] <= start_depth_marker_icec[q]:
                wrong_depth = end_depth_marker_icec[q]
                print "wrong depth", end_depth_marker_icec[q]
                print "start depth", start_depth_marker_icec[q]
                right_depth = input("Please type in right end depth since it is lower than start depth...")
                print endmelt
                fixtime = data_dict["epoch"][idx_ice[-1]]
                print fixtime
                fixdepth = [x for x in data_dict["index"][:] if data_dict["end_depth"][x]==wrong_depth and \
                    data_dict["epoch"][x]<=fixtime]
                for x in fixdepth:
                    data_dict["end_depth"][x] = right_depth
                end_depth_marker_icec[q] = right_depth
            
            # make laser depth reading outliers = nan
            epoch_icec_idv_laserfix = epoch_icec_idv
            laser_distance_icec_idv_laserfix = deepcopy(laser_distance_icec_idv)
            diff_laser_icec_idv = np.append(0, diff_laser_icec_idv) # add 0 to front
            idx_icec_idv = np.arange(len(diff_laser_icec_idv))
            idx = [x for x in idx_icec_idv if diff_laser_icec_idv[x] <= 1 or \
                diff_laser_icec_idv[x] >= 2]
            laser_distance_icec_idv_laserfix[idx] = 'NaN'
            if len(idx)==len(diff_laser_icec_idv):
                laser_distance_icec_idv_laserfix = epoch_icec_idv_laserfix
            
            # remove values at beginning and end, which tend to be problematic
            rvs = 4   #number start points to remove at start
            laser_distance_icec_idv_laserfix[0:rvs] = 'NaN'
            rve = 2   #number end points to remove at end
            laser_distance_icec_idv_laserfix[-rve:-1] = 'NaN'
            
            #end flatline fix
            rvf = 10 #excluding rve, remove this many points at end
            laser_distance_icec_idv_laserfix[-rvf:-rve] = laser_distance_icec_idv[-rvf:-rve]
            
            # replace all core stacking with 'NaN'
            idx = [x for x in idx_icec_idv if loadflag[x] == 1]
            laser_distance_icec_idv_laserfix[idx] = 'NaN'

            # interp beginning to end to replace NaN
            z = epoch_icec_idv_laserfix
            y = laser_distance_icec_idv_laserfix
            idx = np.isnan(y)
            notnan = np.where(idx==False)[0]
            v = z[notnan]
            w = y[notnan]
            endz = v[-20:]
            endy = w[-20:]
            fend = stats.linregress(endz,endy)
            u = z[notnan[-1]:]*fend[0]+fend[1]
            t = z[notnan[-1]:]
            v = np.append(v,t)
            w = np.append(w,u)
            beginz = v[0:20]
            beginy = w[0:20]
            fbegin = stats.linregress(beginz,beginy)
            s = z[0:notnan[0]]*fbegin[0]+fbegin[1]
            r = z[0:notnan[0]]
            v = np.append(r,v)
            w = np.append(s,w)
            f = sp.interpolate.interp1d(v,w)
            laser_distance_interp = f(z)

            A = laser_distance_interp/1000    #mm to m
            inmin = min(A)
            inmax = max(A)
            l = start_depth_marker_icec[q]
            u = end_depth_marker_icec[q]
            true_depth2 = l + (A-inmin)/(inmax-inmin)*(u-l) #forces both the start and end depth

            # repeat interpolation             
            z2 = epoch_icec_idv_laserfix
            y2 = laser_distance_icec_idv_laserfix
            idx = np.isnan(y2)
            notnan = np.where(idx==False)[0]
            v2 = z2[notnan]
            w2 = y2[notnan]
            endz = v2[-20:]
            endy = w2[-20:]
            fend = stats.linregress(endz,endy)
            u2 = z2[notnan[-1]:]*fend[0]+fend[1]
            t2 = z2[notnan[-1]:]
            v2 = np.append(v2,t2)
            w2 = np.append(w2,u2)
            beginz = v2[0:20]
            beginy = w2[0:20]
            fbegin = stats.linregress(beginz,beginy)
            s2 = z2[0:notnan[0]]*fbegin[0]+fbegin[1]
            r2 = z2[0:notnan[0]]
            v2 = np.append(r2,v2)
            w2 = np.append(s2,w2)
            f2 = sp.interpolate.interp1d(v2,w2)
            laser_distance_interp2 = f2(z)

            A = laser_distance_interp2/1000    #mm to m
            inmin = min(A)
            inmax = max(A)
            l = start_depth_marker_icec[q]
            u = end_depth_marker_icec[q]
            true_depth2 = l + (A-inmin)/(inmax-inmin)*(u-l) #forces both the start and end depth
#            true_depth2 = l + (A-inmin) #forces only the start depth
            if l==15.9525:
            	u=16.1455
            print filepath
            print q
            print l
            print u
            if l == 873.95:
            	u = 874.5
            if l == 872.91:
            	u = 873.388
            if l == 1458.583:
            	l = 1458.6
            if l == 1459.137:
            	l = 1459.15
            if l == 2501.95:
            	u = 2502.497
            if l == 2615.25:
            	u = 2615.8
            if l == 85.265:
            	l = 85.25
            	u = 85.785
            if l ==878.364:
            	l = 878.375
            if  l==967.436:
            	l = 967.45
            	u = 967.605
            if l==967.591:
            	l = 967.605
            	u = 968.002
            if l==696.3:
            	u = 696.85
            depthissues = [6.05,13.2,15.9525,17.6025,20.9,40.7,55.55,66.55,69.3,81.395,\
            	82.5,85.25,91.85,94.05,99,101.2,101.71,110.555,110.725,116.05,132.55,\
            	133.1,134.75,138.05,149.6,151.25,151.8,155.1,157.85,165,168.3,169.95,\
            	171.05,184.25,190.3,190.572,204.05,210.65,261.25,265.65,275.55,277.2,278.06,289.85,\
            	292.05,317.35,317.77,326.15,348.7,353.65,354.2,360.8,361.35,361.9,367.95,\
            	384.45,390.5,398.75,406.45,407.55,410.3,414.7,424.05,429.55,430.65,459.8,465.85,466.4,\
            	466.617,470.8,475.2,476.3,481.8,488.95,507.65,509.3,509.85,510.4,514.8,\
            	530.2,545.05,554.4,558.8,564.3,574.75,579.7,580.25,580.804,589.6,590.15,\
            	592.35,598.334,605.55,606.1,607.75,608.3,624.25,628.1,635.8,637.45,642.4,\
				644.022,644.6,645.934,646.25,646.8,647.35,647.9,650.65,680.9,691.35,694.095,696.3,696.85,\
				697.288,701.8,706.2,710.05,714.45,716.65,723.28,723.8,741.4,745.8,746.337,\
				762.85,766.7,767.815,771.65,775.49,780.45,781.0,787.05,787.6,809.048,\
				809.441,814.0,815.1,829.95,831.585,846.475,847.0,848.1,852.505,853.05,855.25,856.9,863.499,\
				867.376,870.1,872.91,873.95,878.375,878.9,879.45,893.204,893.747,895.42,\
				902.55,910.25,916.85,929.503,938.302,938.85,960.3,963.05,965.801,966.352,\
				966.905,967.45,967.605,972.95,973.52,980.65,985.6,991.653,1049.398,1061.5,\
				1062.022,1063.15,1068.6365,1069.2025,1069.7495,1078.0,1079.1,1080.2,\
				1081.3,1081.851,1084.049,1085.138,1087.351,1092.85,1096.7,1097.25,1103.85,\
				1109.9,1133.55,1141.25,1149.5,1183.6,1195.15,1229.765,1243,1243.55,1260.599,1260.91,\
				1260.95,1261.15,1277.65,1301.3,1315.599,1319.45,1355.75,1361.25,1361.8,1367.3,1367.44,\
				1382.7,1390.4,1412.95,1413.5,1416.25,1417.3535,1420.65,1426.15,1440.45,\
				1455.301,1458.6,1459.15,1469.05,1472.9,1478.4075,1524.0505,1532.302,\
				1540.0005,1540.5495,1541.1,1541.645,1588.4005,1609.85,1610.4025,1667.6,\
				1674.201,1675.8505,1676.401,1681.3505,1681.9,1684.1,1685.2005,1697.851,\
				1730.85,1734.7,1753.402,1817.752,1843.052,1844.154,1844.7,1890.349,\
				1913.45,1914.002,1920.05,1996.502,2051.501,2138.951,2179.1,2186.8,2196.150,\
				2196.700,2197.250,2205.5,2212.300,2226.95,2235.198,2246.75,2250.600,\
				2270.95,2274.25,2274.8,2279.2,2288.000,2288.55,2296.25,2313.3,2325.95,\
				2387.55,2401.3,2405.149,2428.8,2434.3,2459.6,2465.101,2487.1,2501.95,\
				2536.047,2615.25,2622.950,2623.5,2633.95,2634.5,2638.351]
            if l in depthissues:
            	print "interpolating full core"
                newline = stats.linregress([epoch_icec_idv_laserfix[0],epoch_icec_idv_laserfix[-1]],[l,u])
                for p in np.arange(len(idx_ice)):
                    true_depth2[p] = epoch_icec_idv_laserfix[p]*newline[0]+newline[1]
            reversedepthissues = [317.77,317.9]
            if l in reversedepthissues:
            	print "interpolating reverse full core" # for 20170621
                newline = stats.linregress([epoch_icec_idv_laserfix[0],epoch_icec_idv_laserfix[-1]],[u,l])
                for p in np.arange(len(idx_ice)):
                    true_depth2[p] = epoch_icec_idv_laserfix[p]*newline[0]+newline[1]

			
            
            #Save data back to data_dict
            coredepth = np.append(coredepth, true_depth2)
            data_dict["true_depth"][idx_ice] = true_depth2
            fig23_ax4.plot(idx_ice, true_depth2, "b-")        
                            
        ##### EC
        ecdelay = np.mean(ecdelays)
        startecindex = midisoindexdD+ecdelay # ecdelays calculated during valcochange section, from midpoints
        
        ##### COLLECT AND WRITE DATA TO MELER DATA FILE ############################
        coreindex = np.arange(len(core))   
        coreflag8 = data_dict["flag8"][core] 
        if data_dict["epoch"][0] > 1325376000:                                                  #### 2130, with if statement for 2140 incorporated data
            for j in coreindex[:]: 
                ice_dict["filepath"] = np.hstack([ice_dict["filepath"],filepath])
                ice_dict["time"] = np.hstack([ice_dict["time"],data_dict["j_days"][midisoindexdD + j]])
                ice_dict["d18o"] = np.hstack([ice_dict["d18o"],memcorrd18o[midisoindexd18o + j]])
                ice_dict["errorpred18o"] = np.hstack([ice_dict["errorpred18o"],stdevcorrthirdd18o])
                ice_dict["erroraccd18o"] = np.hstack([ice_dict["erroraccd18o"],diffcorrthirdd18o])
                ice_dict["rawd18o"] = np.hstack([ice_dict["rawd18o"],data_dict["d18o"][midisoindexd18o + j]])
                ice_dict["dD"] = np.hstack([ice_dict["dD"],memcorrdD[midisoindexdD + j]])
                ice_dict["errorpredD"] = np.hstack([ice_dict["errorpredD"],stdevcorrthirddD])
                ice_dict["erroraccdD"] = np.hstack([ice_dict["erroraccdD"],diffcorrthirddD])
                ice_dict["rawdD"] = np.hstack([ice_dict["rawdD"],data_dict["dD"][midisoindexdD + j]])
                ice_dict["d_excess"] = np.hstack([ice_dict["d_excess"],memcorrdD[midisoindexdD + j]-8*memcorrd18o[midisoindexd18o + j]])   
                ice_dict["water"] = np.hstack([ice_dict["water"],data_dict["water_ppm"][midisoindexdD + j]])
                if filepath == "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/raw_dictionaries/rawHIDS2038-20120213-175121Z-DataLog_User.dat":
                    startdepthdata = startdepth[i]
                    enddepthdata = enddepth[i]
                    depthslope = (enddepthdata-startdepthdata)/(endmelt-startmelt)
                    if j == 35:
                        print "INTERPOLATED DEPTH, INDEX, START, END, SLOPE:", i, startdepthdata, enddepthdata, depthslope
                    ice_dict["depth"] = np.hstack([ice_dict["depth"],startdepthdata + depthslope*j])
                    coreflag8[j] = coreflag8[j]+'D'
                else:
                    ice_dict["depth"] = np.hstack([ice_dict["depth"],coredepth[j]])                       ###laserdepth[j]])                  
                ice_dict["ec"] = np.hstack([ice_dict["ec"],data_dict["ec_value"][startecindex + j]])
                ice_dict["flag1_d18o"] = np.hstack([ice_dict["flag1_d18o"],data_dict["flag1_d18o"][midisoindexd18o + j]])
                ice_dict["flag1_dD"] = np.hstack([ice_dict["flag1_dD"],data_dict["flag1_dD"][midisoindexdD + j]])
                ice_dict["flag2"] = np.hstack([ice_dict["flag2"],data_dict["flag2"][midisoindexd18o + j]])
                ice_dict["flag3"] = np.hstack([ice_dict["flag3"],data_dict["flag3"][midisoindexd18o + j]])
                ice_dict["flag4_d18o"] = np.hstack([ice_dict["flag4_d18o"],data_dict["flag4_d18o"][midisoindexd18o + j]])
                ice_dict["flag4_dD"] = np.hstack([ice_dict["flag4_dD"],data_dict["flag4_dD"][midisoindexdD + j]])
                ice_dict["flag5"] = np.hstack([ice_dict["flag5"],data_dict["flag5"][0]])
                ice_dict["flag6"] = np.hstack([ice_dict["flag6"],data_dict["flag6"][0]])
                ice_dict["flag7_d18o"] = np.hstack([ice_dict["flag7_d18o"],data_dict["flag7_d18o"][midisoindexd18o + j]])
                ice_dict["flag7_dD"] = np.hstack([ice_dict["flag7_dD"],data_dict["flag7_dD"][midisoindexdD + j]])
                ice_dict["flag8"] = np.hstack([ice_dict["flag8"],coreflag8[j]])
                if j <= 10:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P1'
                if j <= 20:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P2'
                if j <= 30:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P3'
                if j <= 40:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P4'
                if j <= 50:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P5'
                if j <= 60:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P6'
                if j <= 70:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P7'
                if j <= 80:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P8'
                if j <= 90:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P9'
                if j <= 100:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'P0'
                if j >= coreindex[-1]-25:
                    ice_dict["flag2"][-1] = ice_dict["flag2"][-1]+'Pe'
                now = datetime.date.today()
                ice_dict["ave_valco_normsigma_d18o"] = np.hstack([ice_dict["ave_valco_normsigma_d18o"],ave_valco_normsigma_d18o])
                ice_dict["ave_valco_normsigma_dD"] = np.hstack([ice_dict["ave_valco_normsigma_dD"],ave_valco_normsigma_dD])
                ice_dict["ave_valco_skewsigma_d18o"] = np.hstack([ice_dict["ave_valco_skewsigma_d18o"],ave_valco_skewsigma_d18o])
                ice_dict["ave_valco_skewsigma_dD"] = np.hstack([ice_dict["ave_valco_skewsigma_dD"],ave_valco_skewsigma_dD])
                if corename == 'EGRIP' and d17oflag ==1:
                    ice_dict["2140d18o"] = np.hstack([ice_dict["2140d18o"],memcorr2140d18o[midisoindex2140d18o + j]])
                    ice_dict["errorpre2140d18o"] = np.hstack([ice_dict["errorpre2140d18o"],stdevcorrthird2140d18o])
                    ice_dict["erroracc2140d18o"] = np.hstack([ice_dict["erroracc2140d18o"],diffcorrthird2140d18o])
                    ice_dict["raw2140d18o"] = np.hstack([ice_dict["raw2140d18o"],data_dict["2140d18o"][midisoindex2140d18o + j]])
                    ice_dict["flag1_2140_d18o"] = np.hstack([ice_dict["flag1_2140_d18o"],data_dict["flag1_2140_d18o"][midisoindex2140d18o + j]])
                    ice_dict["flag1_2140_dD"] = np.hstack([ice_dict["flag1_2140_dD"],data_dict["flag1_2140_dD"][midisoindex2140dD + j]])
                    ice_dict["flag1_2140_d17o"] = np.hstack([ice_dict["flag1_2140_d17o"],data_dict["flag1_2140_d17o"][midisoindex2140d17o + j]])
                    ice_dict["flag4_2140_d18o"] = np.hstack([ice_dict["flag4_2140_d18o"],data_dict["flag4_2140_d18o"][midisoindex2140d18o + j]])
                    ice_dict["flag4_2140_dD"] = np.hstack([ice_dict["flag4_2140_dD"],data_dict["flag4_2140_dD"][midisoindex2140dD + j]])
                    ice_dict["flag4_2140_d17o"] = np.hstack([ice_dict["flag4_2140_d17o"],data_dict["flag4_2140_d17o"][midisoindex2140d17o + j]])
                    ice_dict["flag7_2140_d18o"] = np.hstack([ice_dict["flag7_2140_d18o"],data_dict["flag7_2140_d18o"][midisoindex2140d18o + j]])
                    ice_dict["flag7_2140_dD"] = np.hstack([ice_dict["flag7_2140_dD"],data_dict["flag7_2140_dD"][midisoindex2140dD + j]])
                    ice_dict["flag7_2140_d17o"] = np.hstack([ice_dict["flag7_2140_d17o"],data_dict["flag7_2140_d17o"][midisoindex2140d17o + j]])
                    ice_dict["2140dD"] = np.hstack([ice_dict["2140dD"],memcorr2140dD[midisoindex2140dD + j]])
                    ice_dict["errorpre2140dD"] = np.hstack([ice_dict["errorpre2140dD"],stdevcorrthird2140dD])
                    ice_dict["erroracc2140dD"] = np.hstack([ice_dict["erroracc2140dD"],diffcorrthird2140dD])
                    ice_dict["raw2140dD"] = np.hstack([ice_dict["raw2140dD"],data_dict["2140dD"][midisoindex2140dD + j]])
                    ice_dict["2140d17o"] = np.hstack([ice_dict["2140d17o"],memcorr2140d17o[midisoindex2140d17o + j]])
                    ice_dict["errorpre2140d17o"] = np.hstack([ice_dict["errorpre2140d17o"],stdevcorrthird2140d17o])
                    ice_dict["erroracc2140d17o"] = np.hstack([ice_dict["erroracc2140d17o"],diffcorrthird2140d17o])
                    ice_dict["raw2140d17o"] = np.hstack([ice_dict["raw2140d17o"],data_dict["2140d17o"][midisoindex2140d17o + j]])
                    ice_dict["ave_valco_normsigma_2140d18o"] = np.hstack([ice_dict["ave_valco_normsigma_2140d18o"],ave_valco_normsigma_2140d18o])
                    ice_dict["ave_valco_normsigma_2140dD"] = np.hstack([ice_dict["ave_valco_normsigma_2140dD"],ave_valco_normsigma_2140dD])
                    ice_dict["ave_valco_normsigma_2140d17o"] = np.hstack([ice_dict["ave_valco_normsigma_2140d17o"],ave_valco_normsigma_2140d17o])
                    ice_dict["ave_valco_skewsigma_2140d18o"] = np.hstack([ice_dict["ave_valco_skewsigma_2140d18o"],ave_valco_skewsigma_2140d18o])
                    ice_dict["ave_valco_skewsigma_2140dD"] = np.hstack([ice_dict["ave_valco_skewsigma_2140dD"],ave_valco_skewsigma_2140dD])
                    ice_dict["ave_valco_skewsigma_2140d17o"] = np.hstack([ice_dict["ave_valco_skewsigma_2140d17o"],ave_valco_skewsigma_2140d17o])
                    ice_dict["ave_nea_normsigma_d18o"] = np.hstack([ice_dict["ave_nea_normsigma_d18o"],ave_nea_normsigma_d18o])
                    ice_dict["ave_nea_normsigma_dD"] = np.hstack([ice_dict["ave_nea_normsigma_dD"],ave_nea_normsigma_dD])
                    ice_dict["ave_nea_normsigma_2140d18o"] = np.hstack([ice_dict["ave_nea_normsigma_2140d18o"],ave_nea_normsigma_2140d18o])
                    ice_dict["ave_nea_normsigma_2140dD"] = np.hstack([ice_dict["ave_nea_normsigma_2140dD"],ave_nea_normsigma_2140dD])
                    ice_dict["ave_nea_normsigma_2140d17o"] = np.hstack([ice_dict["ave_nea_normsigma_2140d17o"],ave_nea_normsigma_2140d17o])
                    ice_dict["ave_nea_skewsigma_d18o"] = np.hstack([ice_dict["ave_nea_skewsigma_d18o"],ave_nea_skewsigma_d18o])
                    ice_dict["ave_nea_skewsigma_dD"] = np.hstack([ice_dict["ave_nea_skewsigma_dD"],ave_nea_skewsigma_dD])
                    ice_dict["ave_nea_skewsigma_2140d18o"] = np.hstack([ice_dict["ave_nea_skewsigma_2140d18o"],ave_nea_skewsigma_2140d18o])
                    ice_dict["ave_nea_skewsigma_2140dD"] = np.hstack([ice_dict["ave_nea_skewsigma_2140dD"],ave_nea_skewsigma_2140dD])
                    ice_dict["ave_nea_skewsigma_2140d17o"] = np.hstack([ice_dict["ave_nea_skewsigma_2140d17o"],ave_nea_skewsigma_2140d17o])
                ice_dict["prodate"] = np.hstack([ice_dict["prodate"],now.strftime("%Y%m%d")])
                ice_dict["crunchversion"] = np.hstack([ice_dict["crunchversion"],crunchversion])
        
        # get melt rate from smoothed time and depth data, and put smoothed data into end file
        depth_cm = ice_dict["depth"]*100  # depth in meters*100 = depth in cm
        smooth_depth = smooth(depth_cm)
        minutes = ice_dict["time"]/60  # time in sec/60 = time in minutes
        smooth_time =smooth(minutes)
        diff_depth = np.diff(smooth_depth)
        diff_time = np.diff(smooth_time)
        ice_dict["meltrate"] = diff_depth/diff_time
        smooth_meltrate = smooth(meltrate)   
        
    #### plot memory corrected values onto original graph
    fig21_ax1.plot(data_dict["index"], memcorrd18o, "y-")
    fig21_ax2.plot(data_dict["index"], memcorrdD, "y-")
    fig21_ax3.plot(data_dict["index"], memcorrdexcess, "m-")  
    if corename == 'EGRIP' and d17oflag ==1:
        fig24_ax1.plot(data_dict["index"], memcorr2140d18o, "y-")
        fig24_ax2.plot(data_dict["index"], memcorr2140dD, "y-")
        fig24_ax3.plot(data_dict["index"], memcorr2140d17o, "m-")  
    
    #### WRITE TO SEPARATE FILE TO RECORD START AND END DEPTH, START AND END TIME, AVERAGE MELT RATE AND FILENAME
    #dataout_file = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/AveMeltrateFile"
    dataout_file = "/Users/frio/EGRIP_2023/Data/AveMeltrateFile"
    file = open(dataout_file, "a")
    begincoreindex= np.arange(len(begincores))
    for i in begincoreindex:
        starttime = data_dict["j_days"][begincores[i]] # in seconds
        endtime = data_dict["j_days"][begincores[i]+coredatalength[i]]
        avemeltrate = ((enddepth[i]-startdepth[i])*100)/((endtime-starttime)/60) # in cm/min
        data = np.transpose(np.vstack((filename, starttime, endtime, startdepth[i], enddepth[i], avemeltrate, crunchversion)))
        np.savetxt(file, data, delimiter = "\t", fmt = ("%s", "%s", "%s", "%s", "%s", "%s", "%s"))
    file.close()

    ##### READ IN ICE DATA FILE FROM MELTER FOR QA/QC AND PRUNNING #######################################

    ice_dict["index"] = np.arange(len(ice_dict["d18o"]))
    flagfilter_d18o = [x for x in ice_dict["index"] if ice_dict["flag1_d18o"][x]=="." and ice_dict["flag2"][x]=="."]
    flagfilter_dD = [x for x in ice_dict["index"] if ice_dict["flag1_dD"][x]=="." and ice_dict["flag2"][x]=="."]
       
    fig321 = plt.figure(321)
    clear = plt.clf()
    fig321_ax1 = fig321.add_subplot(511)
    fig321_ax1.plot(ice_dict["index"], ice_dict["d18o"], "g-",ice_dict["index"][flagfilter_d18o], ice_dict["d18o"][flagfilter_d18o], "b-")
    fig321_ax1.set_ylabel("d18o")
    fig321_ax2 = fig321.add_subplot(512)
    fig321_ax2.plot(ice_dict["index"], ice_dict["dD"], "m-", ice_dict["index"][flagfilter_dD], ice_dict["dD"][flagfilter_dD], "r-")
    fig321_ax2.set_ylabel("dD")
    fig321_ax3 = fig321.add_subplot(513)
    fig321_ax3.plot(ice_dict["index"], ice_dict["ec"], "k-", ice_dict["index"][flagfilter_dD], ice_dict["ec"][flagfilter_dD], "g-")
    fig321_ax3.set_ylabel("EC")
    fig321_ax4 = fig321.add_subplot(514)
    fig321_ax4.plot(ice_dict["index"], ice_dict["depth"], "k-", ice_dict["index"][flagfilter_dD], ice_dict["depth"][flagfilter_dD], "g-")
    fig321_ax4.set_ylabel("depth")
    fig321_ax5 = fig321.add_subplot(515)
    fig321_ax5.plot(ice_dict["index"], ice_dict["d_excess"], "g-", ice_dict["index"][flagfilter_dD], ice_dict["d_excess"][flagfilter_dD], "k-")
    fig321_ax5.set_ylabel("d-excess")
    fig321_ax5.set_xlabel("Index")
    fig321_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    if corename == 'EGRIP' and d17oflag == 1:
        flagfilter_2140_d18o = [x for x in ice_dict["index"] if ice_dict["flag1_2140_d18o"][x]=="." and ice_dict["flag2"][x]=="."]
        flagfilter_2140_dD = [x for x in ice_dict["index"] if ice_dict["flag1_2140_dD"][x]=="." and ice_dict["flag2"][x]=="."]
        flagfilter_2140_d17o = [x for x in ice_dict["index"] if ice_dict["flag1_2140_d17o"][x]=="." and ice_dict["flag2"][x]=="."]
        ice_D17O = (sp.log(ice_dict["2140d17o"]+1)-0.528*sp.log(ice_dict["d18o"]+1))
        fig321_ax3.plot(ice_dict["index"], ice_dict["2140d17o"], "k-", ice_dict["index"][flagfilter_2140_d17o], ice_dict["2140d17o"][flagfilter_2140_d17o], "g-")
        fig321_ax3.set_ylabel("2140d17o")
        ice_D17O = (sp.log(ice_dict["2140d17o"]+1)-0.528*sp.log(ice_dict["d18o"]+1))
        fig322 = plt.figure(322)
        clear = plt.clf()
        fig322_ax1 = fig322.add_subplot(511)
        fig322_ax1.plot(ice_dict["index"], ice_dict["2140d18o"], "g-",ice_dict["index"][flagfilter_2140_d18o], ice_dict["2140d18o"][flagfilter_2140_d18o], "b-")
        fig322_ax1.set_ylabel("2140d18o")
        fig322_ax2 = fig322.add_subplot(512)
        fig322_ax2.plot(ice_dict["index"], ice_dict["2140dD"], "m-", ice_dict["index"][flagfilter_2140_dD], ice_dict["2140dD"][flagfilter_2140_dD], "r-")
        fig322_ax2.set_ylabel("2140dD")
        fig322_ax3 = fig322.add_subplot(513)
        fig322_ax3.plot(ice_dict["index"], ice_dict["2140d17o"], "k-", ice_dict["index"][flagfilter_2140_d17o], ice_dict["2140d17o"][flagfilter_2140_d17o], "m-")
        fig322_ax3.set_ylabel("2140d17o")
        fig322_ax4 = fig322.add_subplot(514)
        fig322_ax4.plot(ice_dict["index"], ice_dict["depth"], "k-", ice_dict["index"][flagfilter_dD], ice_dict["depth"][flagfilter_dD], "g-")
        fig322_ax4.set_ylabel("depth")
        fig322_ax5 = fig322.add_subplot(515)
        fig322_ax5.plot(ice_dict["index"], ice_D17O, "g-", ice_dict["index"][flagfilter_2140_d17o], ice_D17O[flagfilter_2140_d17o], "k-")
        fig322_ax5.set_ylabel("D17O")
        fig322_ax5.set_xlabel("Index")
        fig322_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    #plot       
    fig522 = plt.figure(522)                                                                         
    fig522_ax1 = fig522.add_subplot(211)
    fig522_ax1.plot(data_dict["index"], data_dict["true_depth"], "b-")
    fig522_ax1.set_ylabel("true depth")
    fig522_ax2 = fig522.add_subplot(212)
    fig522_ax2.plot(data_dict["index"], data_dict["laser_distance"], "r.")
    fig522_ax2.set_ylabel("laser_distance")
    fig522_ax2.set_xlabel("Index")
    fig522_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    #### Plot d18o for comparison 
    fig622 = plt.figure(622)                                                                         
    fig622_ax1 = fig622.add_subplot(211)
    fig622_ax1.plot(ice_dict["index"][flagfilter_dD], ice_dict["d18o"][flagfilter_dD], "b-")
    fig622_ax1.set_ylabel("d18o")
    fig622_ax2 = fig622.add_subplot(212)
    fig622_ax2.plot(ice_dict["index"][flagfilter_dD], ice_dict["water"][flagfilter_dD], "g.")
    fig622_ax2.set_ylabel("water conc")
    fig622_ax2.set_xlabel("Index")
    fig622_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
    
    os.system('say "your loop has finished"')    
    
    #### Manual Flags, to be applied to WAIS as a concatenated file, left here for SPIce    
    #flagfilepath = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/flag_files/" + filename + "flags.txt"
    flagfilepath = "/Users/frio/EGRIP_2023/Data/flag_files/" + filename + "flags.txt"
    if verbose == 0:  ##change to 0 from 1 to use existing flags in verbose mode 20170118
        manualflag = pickpoints(ice_dict["dD"][flagfilter_dD],ice_dict["index"][flagfilter_dD],flagfilepath)
        plt.show()
    
    #### Read in flag file and apply to ice_dict
    flagdata = np.loadtxt(flagfilepath, dtype = "S", delimiter = " ", ndmin=2)
    print flagdata
    flagindex = np.arange(len(flagdata))
    for f in flagindex:
        splitflagdata = flagdata[f,0].rpartition("-")        
        startflag = int(splitflagdata[0])
        endflag = int(splitflagdata[-1])
        print startflag
        print endflag
        print len(ice_dict["flag3"])
        if endflag >= len(ice_dict["flag3"])-40:
            endflag = len(ice_dict["flag3"])-1 #to go all the way to end
        flagtype = flagdata[f,1]
        print startflag
        print endflag
        print flagtype
        if flagtype == "P":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'p'
        if flagtype == "S":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'s'
        if flagtype == "N":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'n'
        if flagtype == "O":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'o'
        if flagtype == "V":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'v'
        if flagtype == "F":
            findex = np.arange(startflag,endflag+1)
            for l in findex:
                ice_dict["flag3"][l] = ice_dict["flag3"][l]+'s'
    
    flagfilter_d18o = [x for x in ice_dict["index"] if ice_dict["flag1_d18o"][x]=="." and ice_dict["flag2"][x]=="." and ice_dict["flag3"][x]=="."]
    flagfilter_dD = [x for x in ice_dict["index"] if ice_dict["flag1_dD"][x]=="." and ice_dict["flag2"][x]=="." and ice_dict["flag3"][x]=="."]
    

    fig321_ax1.plot(ice_dict["index"][flagfilter_d18o], ice_dict["d18o"][flagfilter_d18o], "y-")
    fig321_ax2.plot(ice_dict["index"][flagfilter_dD], ice_dict["dD"][flagfilter_dD], "y-")
    fig321_ax4.plot(ice_dict["index"][flagfilter_dD], ice_dict["depth"][flagfilter_dD], "y-")
    fig321_ax5.plot(ice_dict["index"][flagfilter_dD], ice_dict["d_excess"][flagfilter_dD], "y-")
    
    if d17oflag == 1:
        flagfilter_2140_d18o = [x for x in ice_dict["index"] if ice_dict["flag1_2140_d18o"][x]=="." and ice_dict["flag2"][x]=="." and ice_dict["flag3"][x]=="."]
        flagfilter_2140_dD = [x for x in ice_dict["index"] if ice_dict["flag1_2140_dD"][x]=="." and ice_dict["flag2"][x]=="." and ice_dict["flag3"][x]=="."]
        flagfilter_2140_d17o = [x for x in ice_dict["index"] if ice_dict["flag1_2140_d17o"][x]=="." and ice_dict["flag2"][x]=="." and ice_dict["flag3"][x]=="."]
        fig321_ax3.plot(ice_dict["index"][flagfilter_2140_d17o], ice_dict["2140d17o"][flagfilter_2140_d17o], "y-")    
    
    if verbose == 1:
    	plt.show()
                
    ## Write 2130 raw data dictionary to file for future reprocessing
    dataout_file = "/Users/frio/EGRIP_2023/Data/raw_dictionaries/raw" + filename
    file = open(dataout_file, "w")   
    pickle.dump(data_dict, file)
    file.close()  
    ## Write 2140 raw data dictionary to file for future reprocessing
    if d17oflag == 1:
        d17odataout_file = "/Users/frio/EGRIP_2023/Data/2140_raw_dictionaries/rawd17o" + filename
        d17ofile = open(d17odataout_file, "w")   
        pickle.dump(d17odata_dict, d17ofile)
        d17ofile.close() 
        
    ##### WRITE ALL STATS TO PERFORMANCE FILE AND SET All Analytical Flags ######################################
    #Allan - 10, 600, 3600 of d18o and dD
    #Valco - Memory?, raw std values
    #Drift - mean and stddev of d18o and dD
    #Scaling - slope and intercept, mean, stdev, and diff of each standard
    #Nea - ?? from transfer, Delays 
    iceindex = np.arange(len(ice_dict["d18o"]))
    #dataout_file = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/MelterPerformanceFile201710"
    dataout_file = "/Users/frio/EGRIP_2023/Data/MelterPerformanceFile2025"
    file = open(dataout_file, "a")
    if corename =="WAIS06AData" or corename == 'EGRIP':
        data = np.transpose(np.vstack((filename, amallanaved18o, amallanstdevd18o, amallan10secd18o, amallan60secd18o, \
            amallan600secd18o, amallan3600secd18o, amallanavedD, amallanstdevdD, amallan10secdD, \
            amallan60secdD, amallan600secdD, amallan3600secdD, amallanh2o, pmallanaved18o, \
                pmallanstdevd18o, pmallan10secd18o, pmallan60secd18o, pmallan600secd18o, pmallan3600secd18o, \
                pmallanavedD, pmallanstdevdD, pmallan10secdD, pmallan60secdD, pmallan600secdD, \
                    pmallan3600secdD, pmallanh2o, avedriftstdd18o, stdevdriftstdd18o, avedriftstddD, \
                    stdevdriftstddD, fullmelttime, d18oslope, d18ointercept, dDslope, \
                        dDintercept, rawstdd18o[0], meancorrfirstd18o, stdevcorrfirstd18o, diffcorrfirstd18o, \
                        rawstddD[0], meancorrfirstdD, stdevcorrfirstdD, diffcorrfirstdD, rawstdd18o[1], \
                            meancorrsecondd18o, stdevcorrsecondd18o, diffcorrsecondd18o, rawstddD[1], meancorrseconddD, \
                            stdevcorrseconddD, diffcorrseconddD, rawtrapd18o, meancorrthirdd18o, stdevcorrthirdd18o, \
                                diffcorrthirdd18o, rawtrapdD, meancorrthirddD, stdevcorrthirddD, diffcorrthirddD, \
                                rawstdd18o[2], meancorrfourthd18o, stdevcorrfourthd18o, diffcorrfourthd18o, rawstddD[1], \
                                    meancorrfourthdD, stdevcorrfourthdD, diffcorrfourthdD, ave_valco_skewsigma_d18o, ave_valco_normsigma_d18o, \
                                    ave_valco_skewsigma_dD, ave_valco_normsigma_dD, ave_nea_skewsigma_d18o, ave_nea_normsigma_d18o, ave_nea_skewsigma_dD, \
                                        ave_nea_normsigma_dD, ice_dict["prodate"][0], crunchversion)))
        np.savetxt(file, data, delimiter = "\t", fmt = ("%s", "%s", "%s", "%s", "%s", \
            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                        "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                        "%s", "%s", "%s"))
        file.close()

    avedriftstd2140d18o = avedriftstdd18o # short term fix    
    stdevdriftstd2140d18o = stdevdriftstdd18o
    avedriftstd2140dD = avedriftstddD # short term fix    
    stdevdriftstd2140dD = stdevdriftstddD
    
    if corename == "EGRIP1": #### add in more here for d17o and have zeros in file before 2016
        data = np.transpose(np.vstack((filename, amallanaved18o, amallanstdevd18o, amallan10secd18o, amallan60secd18o, \
            amallan600secd18o, amallan3600secd18o, amallanavedD, amallanstdevdD, amallan10secdD, \
            amallan60secdD, amallan600secdD, amallan3600secdD, amallanave2140d18o, amallanstdev2140d18o, \
                amallan10sec2140d18o, amallan60sec2140d18o, amallan600sec2140d18o, amallan3600sec2140d18o, amallanave2140dD, \
                amallanstdev2140dD, amallan10sec2140dD, amallan60sec2140dD, amallan600sec2140dD, amallan3600sec2140dD, \
                    amallanave2140d17o, amallanstdev2140d17o, amallan10sec2140d17o, amallan60sec2140d17o, amallan600sec2140d17o, \
                    amallan3600sec2140d17o, amallanh2o, pmallanaved18o, pmallanstdevd18o, pmallan10secd18o, \
                        pmallan60secd18o, pmallan600secd18o, pmallan3600secd18o, pmallanavedD, pmallanstdevdD, \
                        pmallan10secdD, pmallan60secdD, pmallan600secdD, pmallan3600secdD, pmallanave2140d18o, \
                            pmallanstdev2140d18o, pmallan10sec2140d18o, pmallan60sec2140d18o, pmallan600sec2140d18o, pmallan3600sec2140d18o, \
                            pmallanave2140dD, pmallanstdev2140dD, pmallan10sec2140dD, pmallan60sec2140dD, pmallan600sec2140dD, \
                                pmallan3600sec2140dD, pmallanave2140d17o, pmallanstdev2140d17o, pmallan10sec2140d17o, pmallan60sec2140d17o, \
                                pmallan600sec2140d17o, pmallan3600sec2140d17o, pmallanh2o, avedriftstdd18o, stdevdriftstdd18o, \
                                    avedriftstddD, stdevdriftstddD, avedriftstd2140d18o, stdevdriftstd2140d18o, avedriftstd2140dD, \
                                    stdevdriftstd2140dD, avedriftstd2140d17o, stdevdriftstd2140d17o, fullmelttime, d18oslope, \
                                        d18ointercept, dDslope, dDintercept, slope2140d18o, intercept2140d18o, \
                                        slope2140dD, intercept2140dD, slope2140d17o, intercept2140d17o, rawstdd18o[0], \
                                            meancorrfirstd18o, stdevcorrfirstd18o, diffcorrfirstd18o, rawstddD[0], meancorrfirstdD, \
                                            stdevcorrfirstdD, diffcorrfirstdD, rawstd2140d18o[0], meancorrfirst2140d18o, stdevcorrfirst2140d18o, \
                                                diffcorrfirst2140d18o, rawstd2140dD[0], meancorrfirst2140dD, stdevcorrfirst2140dD, diffcorrfirst2140dD, \
                                                rawstd2140d17o[0], meancorrfirst2140d17o, stdevcorrfirst2140d17o, diffcorrfirst2140d17o, rawstdd18o[1], \
                                                    meancorrsecondd18o, stdevcorrsecondd18o, diffcorrsecondd18o, rawstddD[1], meancorrseconddD, \
                                                    stdevcorrseconddD, diffcorrseconddD, rawstd2140d18o[1], meancorrsecond2140d18o, stdevcorrsecond2140d18o, \
                                                        diffcorrsecond2140d18o, rawstd2140dD[1], meancorrsecond2140dD, stdevcorrsecond2140dD, diffcorrsecond2140dD, \
                                                        rawstd2140d17o[1], meancorrsecond2140d17o, stdevcorrsecond2140d17o, diffcorrsecond2140d17o, rawtrapd18o, \
                                                            meancorrthirdd18o, stdevcorrthirdd18o, diffcorrthirdd18o, rawtrapdD, meancorrthirddD, \
                                                            stdevcorrthirddD, diffcorrthirddD, rawtrap2140d18o, meancorrthird2140d18o, stdevcorrthird2140d18o, \
                                                                diffcorrthird2140d18o, rawtrap2140dD, meancorrthird2140dD, stdevcorrthird2140dD, diffcorrthird2140dD, \
                                                                rawtrap2140d17o, meancorrthird2140d17o, stdevcorrthird2140d17o, diffcorrthird2140d17o, rawstdd18o[2], \
                                                                    meancorrfourthd18o, stdevcorrfourthd18o, diffcorrfourthd18o, rawstddD[2], meancorrfourthdD, \
                                                                    stdevcorrfourthdD, diffcorrfourthdD, rawstd2140d18o[2], meancorrfourth2140d18o, stdevcorrfourth2140d18o, \
                                                                        diffcorrfourth2140d18o, rawstd2140dD[2], meancorrfourth2140dD, stdevcorrfourth2140dD, diffcorrfourth2140dD, \
                                                                        rawstd2140d17o[2], meancorrfourth2140d17o, stdevcorrfourth2140d17o, diffcorrfourth2140d17o, ave_valco_skewsigma_d18o, \
                                                                            ave_valco_normsigma_d18o, ave_valco_skewsigma_dD, ave_valco_normsigma_dD, ave_valco_skewsigma_2140d18o, ave_valco_normsigma_2140d18o, \
                                                                            ave_valco_skewsigma_2140dD, ave_valco_normsigma_2140dD, ave_valco_skewsigma_2140d17o, ave_valco_normsigma_2140d17o, ave_nea_skewsigma_d18o, \
                                                                                ave_nea_normsigma_d18o, ave_nea_skewsigma_dD, ave_nea_normsigma_dD, ave_nea_skewsigma_2140d18o, ave_nea_normsigma_2140d18o, \
                                                                                ave_nea_skewsigma_2140dD, ave_nea_normsigma_2140dD, ave_nea_skewsigma_2140d17o, ave_nea_normsigma_2140d17o, ice_dict["prodate"][0], \
                                                                                    crunchversion)))
        np.savetxt(file, data, delimiter = "\t", fmt = ("%s", "%s", "%s", "%s", "%s", \
            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                        "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                        "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                        "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                    "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                        "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                            "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                                "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s", \
                                                                                    "%s"))

        file.close()
            
    # sA and sP flag - 1st position, if standards have a collective standard deviation or difference from know greater than 0.3 for or 3.0 for dD 
    # sa and sp flags - 4th position, if standards have a collective standard deviation or difference from know greater than 0.1 for d18O or 1.0 for dD (a1), or 0.2 and 2.0 (a2), 0.3 and 3.0 (a3)
    if stdevcorrfirstd18o > 0.1 or stdevcorrsecondd18o > 0.1 or stdevcorrfourthd18o > 0.1:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sp1'
    if diffcorrfirstd18o > 0.1 or diffcorrsecondd18o > 0.1 or diffcorrfourthd18o > 0.1:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sa1'
    if stdevcorrfirstdD > 1.0 or stdevcorrseconddD > 1.0 or stdevcorrfourthdD > 1.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sp1'     
    if diffcorrfirstdD > 1.0 or diffcorrseconddD > 1.0 or diffcorrfourthdD > 1.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sa1'  
                
    if stdevcorrfirstd18o > 0.2 or stdevcorrsecondd18o > 0.2 or stdevcorrfourthd18o > 0.2:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sp2'
    if diffcorrfirstd18o > 0.2 or diffcorrsecondd18o > 0.2 or diffcorrfourthd18o > 0.2:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sa2'
    if stdevcorrfirstdD > 2.0 or stdevcorrseconddD > 2.0 or stdevcorrfourthdD > 2.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sp2'     
    if diffcorrfirstdD > 2.0 or diffcorrseconddD > 2.0 or diffcorrfourthdD > 2.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sa2'
                  
    if stdevcorrfirstd18o > 0.3 or stdevcorrsecondd18o > 0.3 or stdevcorrfourthd18o > 0.3:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sp3'
    if diffcorrfirstd18o > 0.3 or diffcorrsecondd18o > 0.3 or diffcorrfourthd18o > 0.3:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sa3'
    if stdevcorrfirstdD > 3.0 or stdevcorrseconddD > 3.0 or stdevcorrfourthdD > 3.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sp3'     
    if diffcorrfirstdD > 3.0 or diffcorrseconddD > 3.0 or diffcorrfourthdD > 3.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sa3'
                    
    if stdevcorrfirstd18o > 0.4 or stdevcorrsecondd18o > 0.4 or stdevcorrfourthd18o > 0.4:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sp4'
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sP'
    if diffcorrfirstd18o > 0.4 or diffcorrsecondd18o > 0.4 or diffcorrfourthd18o > 0.4:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sa4'
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'sA'
    if stdevcorrfirstdD > 4.0 or stdevcorrseconddD > 4.0 or stdevcorrfourthdD > 4.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sp4'  
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sp'   
    if diffcorrfirstdD > 4.0 or diffcorrseconddD > 4.0 or diffcorrfourthdD > 4.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sa4'
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'sA'
                            
    # tA and tP flag - 1st position, if trap standard has a collective standard deviation or difference from know greater than 0.3 for or 3.0 for dD
    # ta and tp flags - 4th position, if trap standard have a collective standard deviation or difference from know greater than 0.1 for d18O or 1.0 for dD (t1), or 0.2 and 2.0 (t2), 0.3 and 3.0 (t3)
    if stdevcorrthirdd18o > 0.1:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tp1'
    if diffcorrthirdd18o > 0.1:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ta1'
    if stdevcorrthirddD > 1.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tp1'
    if diffcorrthirddD > 1.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ta1'        
            
    if stdevcorrthirdd18o > 0.2:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tp2'
    if diffcorrthirdd18o > 0.2:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ta2'        
    if stdevcorrthirddD > 2.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tp2'
    if diffcorrthirddD > 2.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ta2'        
            
    if stdevcorrthirdd18o > 0.3:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tp3'
    if diffcorrthirdd18o > 0.3:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ta3'        
    if stdevcorrthirddD > 3.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tp3'
    if diffcorrthirddD > 3.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ta3'        
            
    if stdevcorrthirdd18o > 0.4:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tp4'
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tP'
    if diffcorrthirdd18o > 0.4:
        for i in iceindex:
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ta4'
            ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'tA'        
    if stdevcorrthirddD > 4.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tp4'
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tP'
    if diffcorrthirddD > 4.0:
        for i in iceindex:
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ta4'
            ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'tA'        
            
    # ap flag - 4th position, if allan varience at __ seconds is greater than 0.1 for d18O or 1.0 for dD (ap1), or 0.2 and 2.0 (ap2), 0.3 and 3.0 (ap3) 
    # aP flag - 1st position,
    if amallanstdevd18o > 0.1 or amallan10secd18o > 0.1 or amallan60secd18o > 0.1 or amallan600secd18o > 0.1 or  amallan3600secd18o > 0.1 or \
        pmallanstdevd18o > 0.1 or pmallan10secd18o > 0.1 or pmallan60secd18o > 0.1 or pmallan600secd18o > 0.1 or pmallan3600secd18o > 0.1:
             for i in iceindex:
                ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] +'ap1'
    if amallanstdevdD > 1.0 or amallan10secdD > 1.0 or amallan60secdD > 1.0 or amallan600secdD > 1.0 or amallan3600secdD > 1.0 or \
        pmallanstdevdD > 1.0 or pmallan10secdD > 1.0 or pmallan60secdD > 1.0 or pmallan600secdD > 1.0 or pmallan3600secdD > 1.0:
            for i in iceindex:
                ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] +'ap1'
    if amallanstdevd18o > 0.2 or amallan10secd18o > 0.2 or amallan60secd18o > 0.2 or amallan600secd18o > 0.2 or  amallan3600secd18o > 0.2 or \
        pmallanstdevd18o > 0.2 or pmallan10secd18o > 0.2 or pmallan60secd18o > 0.2 or pmallan600secd18o > 0.2 or pmallan3600secd18o > 0.2:
            for i in iceindex:
                ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ap2'
    if amallanstdevdD > 2.0 or amallan10secdD > 2.0 or amallan60secdD > 2.0 or amallan600secdD > 2.0 or amallan3600secdD > 2.0 or \
        pmallanstdevdD > 2.0 or pmallan10secdD > 2.0 or pmallan60secdD > 2.0 or pmallan600secdD > 1.0 or pmallan3600secdD > 2.0:
            for i in iceindex:
                ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ap2'
    if amallanstdevd18o > 0.3 or amallan10secd18o > 0.3 or amallan60secd18o > 0.3 or amallan600secd18o > 0.3 or  amallan3600secd18o > 0.3 or \
        pmallanstdevd18o > 0.3 or pmallan10secd18o > 0.3 or pmallan60secd18o > 0.3 or pmallan600secd18o > 0.3 or pmallan3600secd18o > 0.3:
            for i in iceindex:
                ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ap3'
    if amallanstdevdD > 3.0 or amallan10secdD > 3.0 or amallan60secdD > 3.0 or amallan600secdD > 3.0 or amallan3600secdD > 3.0 or \
        pmallanstdevdD > 3.0 or pmallan10secdD > 3.0 or pmallan60secdD > 3.0 or pmallan600secdD > 3.0 or pmallan3600secdD > 3.0:
            for i in iceindex:
                ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ap3'
    if amallanstdevd18o > 0.4 or amallan10secd18o > 0.4 or amallan60secd18o > 0.4 or amallan600secd18o > 0.4 or  amallan3600secd18o > 0.4 or \
        pmallanstdevd18o > 0.4 or pmallan10secd18o > 0.4 or pmallan60secd18o > 0.4 or pmallan600secd18o > 0.4 or pmallan3600secd18o > 0.4:
            for i in iceindex:
                ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'ap4'
                ice_dict["flag4_d18o"][i] = ice_dict["flag4_d18o"][i] + 'aP'
    if amallanstdevdD > 4.0 or amallan10secdD > 4.0 or amallan60secdD > 4.0 or amallan600secdD > 4.0 or amallan3600secdD > 4.0 or \
        pmallanstdevdD > 4.0 or pmallan10secdD > 4.0 or pmallan60secdD > 4.0 or pmallan600secdD > 4.0 or pmallan3600secdD > 4.0:
            for i in iceindex:
                ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'ap4'
                ice_dict["flag4_dD"][i] = ice_dict["flag4_dD"][i] + 'aP'
                
    if corename == 'DYE3' and d17oflag == 1:
        # A flag - 1st position, if standards have a collective standard deviation or difference from know greater than 0.3 for or 3.0 for dD 
        # a flag - 4th position, if standards have a collective standard deviation or difference from know greater than 0.1 for d18O or 1.0 for dD (a1), or 0.2 and 2.0 (a2), 0.3 and 3.0 (a3)
        if stdevcorrfirst2140d18o > 0.1 or stdevcorrsecond2140d18o > 0.1 or stdevcorrfourth2140d18o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sp1'
        if diffcorrfirst2140d18o > 0.1 or diffcorrsecond2140d18o > 0.1 or diffcorrfourth2140d18o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sa1'
        if stdevcorrfirst2140dD > 1.0 or stdevcorrsecond2140dD > 1.0 or stdevcorrfourth2140dD > 1.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sp1'     
        if diffcorrfirst2140dD > 1.0 or diffcorrsecond2140dD > 1.0 or diffcorrfourth2140dD > 1.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sa1'  
        if stdevcorrfirst2140d17o > 0.1 or stdevcorrsecond2140d17o > 0.1 or stdevcorrfourth2140d17o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sp1'
        if diffcorrfirst2140d17o > 0.1 or diffcorrsecond2140d17o > 0.1 or diffcorrfourth2140d17o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sa1'
                    
        if stdevcorrfirst2140d18o > 0.2 or stdevcorrsecond2140d18o > 0.2 or stdevcorrfourth2140d18o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sp2'
        if diffcorrfirst2140d18o > 0.2 or diffcorrsecond2140d18o > 0.2 or diffcorrfourth2140d18o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sa2'
        if stdevcorrfirst2140dD > 2.0 or stdevcorrsecond2140dD > 2.0 or stdevcorrfourth2140dD > 2.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sp2'     
        if diffcorrfirst2140dD > 2.0 or diffcorrsecond2140dD > 2.0 or diffcorrfourth2140dD > 2.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sa2'  
        if stdevcorrfirst2140d17o > 0.2 or stdevcorrsecond2140d17o > 0.2 or stdevcorrfourth2140d17o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sp2'
        if diffcorrfirst2140d17o > 0.2 or diffcorrsecond2140d17o > 0.2 or diffcorrfourth2140d17o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sa2'
                      
        if stdevcorrfirst2140d18o > 0.3 or stdevcorrsecond2140d18o > 0.3 or stdevcorrfourth2140d18o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sp3'
        if diffcorrfirst2140d18o > 0.3 or diffcorrsecond2140d18o > 0.3 or diffcorrfourth2140d18o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sa3'
        if stdevcorrfirst2140dD > 3.0 or stdevcorrsecond2140dD > 3.0 or stdevcorrfourth2140dD > 3.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sp3'     
        if diffcorrfirst2140dD > 3.0 or diffcorrsecond2140dD > 3.0 or diffcorrfourth2140dD > 3.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sa3'  
        if stdevcorrfirst2140d17o > 0.3 or stdevcorrsecond2140d17o > 0.3 or stdevcorrfourth2140d17o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sp3'
        if diffcorrfirst2140d17o > 0.3 or diffcorrsecond2140d17o > 0.3 or diffcorrfourth2140d17o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sa3'
                        
        if stdevcorrfirst2140d18o > 0.4 or stdevcorrsecond2140d18o > 0.4 or stdevcorrfourth2140d18o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sp4'
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sP'
        if diffcorrfirst2140d18o > 0.4 or diffcorrsecond2140d18o > 0.4 or diffcorrfourth2140d18o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sa4'
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'sA'
        if stdevcorrfirst2140dD > 4.0 or stdevcorrsecond2140dD > 4.0 or stdevcorrfourth2140dD > 4.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sp4'  
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sp'   
        if diffcorrfirst2140dD > 4.0 or diffcorrsecond2140dD > 4.0 or diffcorrfourth2140dD > 4.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sa4'
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'sA'
        if stdevcorrfirst2140d17o > 0.4 or stdevcorrsecond2140d17o > 0.4 or stdevcorrfourth2140d17o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sp4'
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sP'
        if diffcorrfirst2140d17o > 0.4 or diffcorrsecond2140d17o > 0.4 or diffcorrfourth2140d17o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sa4'
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'sA'
                                
        # T flag - 1st position, if trap standard has a collective standard deviation or difference from know greater than 0.3 for or 3.0 for dD
        # t flag - 4th position, if trap standard have a collective standard deviation or difference from know greater than 0.1 for d18O or 1.0 for dD (t1), or 0.2 and 2.0 (t2), 0.3 and 3.0 (t3)
        if stdevcorrthird2140d18o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tp1'
        if diffcorrthird2140d18o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ta1'
        if stdevcorrthird2140dD > 1.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tp1'
        if diffcorrthird2140dD > 1.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ta1'  
        if stdevcorrthird2140d17o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tp1'
        if diffcorrthird2140d17o > 0.1:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ta1'
                
        if stdevcorrthird2140d18o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tp2'
        if diffcorrthird2140d18o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ta2'
        if stdevcorrthird2140dD > 2.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tp2'
        if diffcorrthird2140dD > 2.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ta2'  
        if stdevcorrthird2140d17o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tp2'
        if diffcorrthird2140d17o > 0.2:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ta2'        
                
        if stdevcorrthird2140d18o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tp3'
        if diffcorrthird2140d18o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ta3'
        if stdevcorrthird2140dD > 3.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tp3'
        if diffcorrthird2140dD > 3.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ta3'  
        if stdevcorrthird2140d17o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tp3'
        if diffcorrthird2140d17o > 0.3:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ta3'      
                
        if stdevcorrthird2140d18o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tp4'
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tP'
        if diffcorrthird2140d18o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ta4'
                ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'tA'        
        if stdevcorrthird2140dD > 4.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tp4'
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tP'
        if diffcorrthird2140dD > 4.0:
            for i in iceindex:
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ta4'
                ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'tA' 
        if stdevcorrthird2140d17o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tp4'
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tP'
        if diffcorrthird2140d17o > 0.4:
            for i in iceindex:
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ta4'
                ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'tA' 
                
        # ap flag - 4th position, if allan varience at __ seconds is greater than 0.1 for d18O or 1.0 for dD (ap1), or 0.2 and 2.0 (ap2), 0.3 and 3.0 (ap3) 
        # aP flag - 1st position,
        if amallanstdev2140d18o > 0.1 or amallan10sec2140d18o > 0.1 or amallan60sec2140d18o > 0.1 or amallan600sec2140d18o > 0.1 or  amallan3600sec2140d18o > 0.1 or \
            pmallanstdev2140d18o > 0.1 or pmallan10sec2140d18o > 0.1 or pmallan60sec2140d18o > 0.1 or pmallan600sec2140d18o > 0.1 or pmallan3600sec2140d18o > 0.1:
                 for i in iceindex:
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] +'ap1'
        if amallanstdev2140dD > 1.0 or amallan10sec2140dD > 1.0 or amallan60sec2140dD > 1.0 or amallan600sec2140dD > 1.0 or amallan3600sec2140dD > 1.0 or \
            pmallanstdev2140dD > 1.0 or pmallan10sec2140dD > 1.0 or pmallan60sec2140dD > 1.0 or pmallan600sec2140dD > 1.0 or pmallan3600sec2140dD > 1.0:
                for i in iceindex:
                    ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] +'ap1'
        if amallanstdev2140d17o > 0.1 or amallan10sec2140d17o > 0.1 or amallan60sec2140d17o > 0.1 or amallan600sec2140d17o > 0.1 or  amallan3600sec2140d17o > 0.1 or \
            pmallanstdev2140d17o > 0.1 or pmallan10sec2140d17o > 0.1 or pmallan60sec2140d17o > 0.1 or pmallan600sec2140d17o > 0.1 or pmallan3600sec2140d17o > 0.1:
                 for i in iceindex:
                    ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] +'ap1'
        
        if amallanstdev2140d18o > 0.2 or amallan10sec2140d18o > 0.2 or amallan60sec2140d18o > 0.2 or amallan600sec2140d18o > 0.2 or  amallan3600sec2140d18o > 0.2 or \
            pmallanstdev2140d18o > 0.2 or pmallan10sec2140d18o > 0.2 or pmallan60sec2140d18o > 0.2 or pmallan600sec2140d18o > 0.2 or pmallan3600sec2140d18o > 0.2:
                for i in iceindex:
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ap2'
        if amallanstdev2140dD > 2.0 or amallan10sec2140dD > 2.0 or amallan60sec2140dD > 2.0 or amallan600sec2140dD > 2.0 or amallan3600sec2140dD > 2.0 or \
            pmallanstdev2140dD > 2.0 or pmallan10sec2140dD > 2.0 or pmallan60sec2140dD > 2.0 or pmallan600sec2140dD > 1.0 or pmallan3600sec2140dD > 2.0:
                for i in iceindex:
                    ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ap2'
        if amallanstdev2140d17o > 0.2 or amallan10sec2140d17o > 0.2 or amallan60sec2140d17o > 0.2 or amallan600sec2140d17o > 0.2 or  amallan3600sec2140d17o > 0.2 or \
            pmallanstdev2140d17o > 0.2 or pmallan10sec2140d17o > 0.2 or pmallan60sec2140d17o > 0.2 or pmallan600sec2140d17o > 0.2 or pmallan3600sec2140d17o > 0.2:
                for i in iceindex:
                    ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ap2'
        
        if amallanstdev2140d18o > 0.3 or amallan10sec2140d18o > 0.3 or amallan60sec2140d18o > 0.3 or amallan600sec2140d18o > 0.3 or  amallan3600sec2140d18o > 0.3 or \
            pmallanstdev2140d18o > 0.3 or pmallan10sec2140d18o > 0.3 or pmallan60sec2140d18o > 0.3 or pmallan600sec2140d18o > 0.3 or pmallan3600sec2140d18o > 0.3:
                for i in iceindex:
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ap3'
        if amallanstdev2140dD > 3.0 or amallan10sec2140dD > 3.0 or amallan60sec2140dD > 3.0 or amallan600sec2140dD > 3.0 or amallan3600sec2140dD > 3.0 or \
            pmallanstdev2140dD > 3.0 or pmallan10sec2140dD > 3.0 or pmallan60sec2140dD > 3.0 or pmallan600sec2140dD > 3.0 or pmallan3600sec2140dD > 3.0:
                for i in iceindex:
                    ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ap3'
        if amallanstdev2140d17o > 0.3 or amallan10sec2140d17o > 0.3 or amallan60sec2140d17o > 0.3 or amallan600sec2140d17o > 0.3 or  amallan3600sec2140d17o > 0.3 or \
            pmallanstdev2140d17o > 0.3 or pmallan10sec2140d17o > 0.3 or pmallan60sec2140d17o > 0.3 or pmallan600sec2140d17o > 0.3 or pmallan3600sec2140d17o > 0.3:
                for i in iceindex:
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ap3'
        
        if amallanstdev2140d18o > 0.4 or amallan10sec2140d18o > 0.4 or amallan60sec2140d18o > 0.4 or amallan600sec2140d18o > 0.4 or  amallan3600sec2140d18o > 0.4 or \
            pmallanstdev2140d18o > 0.4 or pmallan10sec2140d18o > 0.4 or pmallan60sec2140d18o > 0.4 or pmallan600sec2140d18o > 0.4 or pmallan3600sec2140d18o > 0.4:
                for i in iceindex:
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'ap4'
                    ice_dict["flag4_2140_d18o"][i] = ice_dict["flag4_2140_d18o"][i] + 'aP'
        if amallanstdev2140dD > 4.0 or amallan10sec2140dD > 4.0 or amallan60sec2140dD > 4.0 or amallan600sec2140dD > 4.0 or amallan3600sec2140dD > 4.0 or \
            pmallanstdev2140dD > 4.0 or pmallan10sec2140dD > 4.0 or pmallan60sec2140dD > 4.0 or pmallan600sec2140dD > 4.0 or pmallan3600sec2140dD > 4.0:
                for i in iceindex:
                    ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'ap4'
                    ice_dict["flag4_2140_dD"][i] = ice_dict["flag4_2140_dD"][i] + 'aP'
        if amallanstdev2140d17o > 0.4 or amallan10sec2140d17o > 0.4 or amallan60sec2140d17o > 0.4 or amallan600sec2140d17o > 0.4 or  amallan3600sec2140d17o > 0.4 or \
            pmallanstdev2140d17o > 0.4 or pmallan10sec2140d17o > 0.4 or pmallan60sec2140d17o > 0.4 or pmallan600sec2140d17o > 0.4 or pmallan3600sec2140d17o > 0.4:
                for i in iceindex:
                    ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'ap4'
                    ice_dict["flag4_2140_d17o"][i] = ice_dict["flag4_2140_d17o"][i] + 'aP'

    #### Write file with all Flags
    #dataout_file = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/calc_dictionaries/calc" + filename
    dataout_file = "/Users/frio/EGRIP_2023/Data/calc_dictionaries/calcraw" + filename
    file = open(dataout_file, "w")                                                    # create bininary file to write to
    pickle.dump(ice_dict, file)                                                       # write dictionary (calc_dict) to file (file)
    file.close()
    
    if verbose == 1:     #For crunching individual files and pruning by indices #### Can change to 0 FOR PROBLEM FILES through batch crunch
        print filename
        print "hard reject flag1_d18o", ice_dict["flag1_d18o"]
        print "hard reject flag1_dD", ice_dict["flag1_dD"]
        print "hard reject flag1_2140_d18o", ice_dict["flag1_2140_d18o"]
        print "hard reject flag1_2140_dD", ice_dict["flag1_2140_dD"]
        print "hard reject flag1_2140_d17o", ice_dict["flag1_2140_d17o"]
        print "prune flag2", ice_dict["flag2"]
        print "manual flag3", ice_dict["flag3"]
        print "statistical flag4_d18o", ice_dict["flag4_d18o"]
        print "statistical flag4_dD", ice_dict["flag4_dD"]
        print "statistical flag4_2140_d18o", ice_dict["flag4_2140_d18o"]
        print "statistical flag4_2140_dD", ice_dict["flag4_2140_dD"]
        print "statistical flag4_2140_d17o", ice_dict["flag4_2140_d17o"]
        print "method flag5", ice_dict["flag5"]
        print "instrument flag6", ice_dict["flag6"]
        print "memory flag7_d18o", ice_dict["flag7_d18o"]
        print "memory flag7_dD", ice_dict["flag7_dD"]
        print "memory flag7_2140_d18o", ice_dict["flag7_2140_d18o"]
        print "memory flag7_2140_dD", ice_dict["flag7_2140_dD"]
        print "memory flag7_2140_d17o", ice_dict["flag7_2140_d17o"]
        print "depth interpolation flag8", ice_dict["flag8"]
        plt.show()                                                     # to pause and look at each run before continuing
        # checkflag = input("Check flags and graphs. Type number to continue?  ")       # to pause and look at each run before continuing

print "reading in all data"
    
#### Read in to Plot All Trimmed Data
trimmed_dict_long = {}
trimmed_dict_long["filepath"] = np.array(()).astype("S")
trimmed_dict_long["time"] = np.array(())
trimmed_dict_long["d18o"] = np.array(())
trimmed_dict_long["errorpred18o"] = np.array(())
trimmed_dict_long["erroraccd18o"] = np.array(())
trimmed_dict_long["rawd18o"] = np.array(())
trimmed_dict_long["dD"] = np.array(())
trimmed_dict_long["errorpredD"] = np.array(())
trimmed_dict_long["erroraccdD"] = np.array(())
trimmed_dict_long["rawdD"] = np.array(())
trimmed_dict_long["2140d18o"] = np.array(())
trimmed_dict_long["errorpre2140d18o"] = np.array(())
trimmed_dict_long["erroracc2140d18o"] = np.array(())
trimmed_dict_long["raw2140d18o"] = np.array(())
trimmed_dict_long["2140dD"] = np.array(())
trimmed_dict_long["errorpre2140dD"] = np.array(())
trimmed_dict_long["erroracc2140dD"] = np.array(())
trimmed_dict_long["raw2140dD"] = np.array(())
trimmed_dict_long["2140d17o"] = np.array(())
trimmed_dict_long["errorpre2140d17o"] = np.array(())
trimmed_dict_long["erroracc2140d17o"] = np.array(())
trimmed_dict_long["raw2140d17o"] = np.array(())
trimmed_dict_long["d_excess"] =np.array(())
trimmed_dict_long["water"] = np.array(())
trimmed_dict_long["ec"] = np.array(())
trimmed_dict_long["depth"] = np.array(())
trimmed_dict_long["meltrate"] = np.array(())
trimmed_dict_long["flag1_d18o"] = np.array(()).astype("S")
trimmed_dict_long["flag1_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag1_2140_d18o"] = np.array(()).astype("S")
trimmed_dict_long["flag1_2140_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag1_2140_d17o"] = np.array(()).astype("S")
trimmed_dict_long["flag2"] = np.array(()).astype("S")
trimmed_dict_long["flag3"] = np.array(()).astype("S")
trimmed_dict_long["flag4_d18o"] = np.array(()).astype("S")
trimmed_dict_long["flag4_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag4_2140_d18o"] = np.array(()).astype("S") 
trimmed_dict_long["flag4_2140_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag4_2140_d17o"] = np.array(()).astype("S")
trimmed_dict_long["flag5"] = np.array(()).astype("S")
trimmed_dict_long["flag6"] = np.array(()).astype("S")
trimmed_dict_long["flag7_d18o"] = np.array(()).astype("S")
trimmed_dict_long["flag7_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag7_2140_d18o"] = np.array(()).astype("S")
trimmed_dict_long["flag7_2140_dD"] = np.array(()).astype("S")
trimmed_dict_long["flag7_2140_d17o"] = np.array(()).astype("S")
trimmed_dict_long["flag8"] = np.array(()).astype("S")
trimmed_dict_long["ave_valco_normsigma_d18o"] = np.array(())
trimmed_dict_long["ave_valco_normsigma_dD"] = np.array(())
trimmed_dict_long["ave_valco_normsigma_2140d18o"] = np.array(())
trimmed_dict_long["ave_valco_normsigma_2140dD"] = np.array(())
trimmed_dict_long["ave_valco_normsigma_2140d17o"] = np.array(())
trimmed_dict_long["ave_valco_skewsigma_d18o"] = np.array(())
trimmed_dict_long["ave_valco_skewsigma_dD"] = np.array(())
trimmed_dict_long["ave_valco_skewsigma_2140d18o"] = np.array(())
trimmed_dict_long["ave_valco_skewsigma_2140dD"] = np.array(())
trimmed_dict_long["ave_valco_skewsigma_2140d17o"] = np.array(())
trimmed_dict_long["ave_nea_normsigma_d18o"] = np.array(())
trimmed_dict_long["ave_nea_normsigma_dD"] = np.array(())
trimmed_dict_long["ave_nea_normsigma_2140d18o"] = np.array(())
trimmed_dict_long["ave_nea_normsigma_2140dD"] = np.array(())
trimmed_dict_long["ave_nea_normsigma_2140d17o"] = np.array(())
trimmed_dict_long["ave_nea_skewsigma_d18o"] = np.array(())
trimmed_dict_long["ave_nea_skewsigma_dD"] = np.array(())
trimmed_dict_long["ave_nea_skewsigma_2140d18o"] = np.array(())
trimmed_dict_long["ave_nea_skewsigma_2140dD"] = np.array(())
trimmed_dict_long["ave_nea_skewsigma_2140d17o"] = np.array(())
trimmed_dict_long["prodate"] = np.array(())
trimmed_dict_long["rundate"] = np.array(())
trimmed_dict_long["crunchversion"] = np.array(())
trimmed_dict_long["index"] = np.array(())
trimmed_dict_long["ec_value"] = np.array(())

#for root, dirs, files in os.walk('/Users/frio/Dropbox/WaterWorld/'+corename+'/Data/calc_dictionaries'):
for root, dirs, files in os.walk('/Users/frio/EGRIP_2023/Data/calc_dictionaries'):
	files = sorted(files)
	if verbose ==1:
		files = sorted(files)
		print files
    

if files[0] == '.DS_Store':
    files = files[1:]
if files[0] == 'Icon\r':
    files = files[1:]
if files[-1] == 'Icon\r':
    files = files[:-1]

for file in files[:]:   
    #filepath = "/Users/frio/Dropbox/WaterWorld/"+corename+"/Data/calc_dictionaries/" + file
    filepath = "/Users/frio/EGRIP_2023/Data/calc_dictionaries/" + file
    trimdata = open(filepath, "r")                                   # open bininary file to read
    trimmed_data_dict = pickle.load(trimdata)
    trimmed_data_dict["meltrate"] = np.append(trimmed_data_dict["meltrate"],trimmed_data_dict["meltrate"][-1])
    if len(trimmed_data_dict["2140d17o"])==0:
		print "no 2140 data?"
		trimmed_data_dict["2140d17o"] = np.arange(len(trimmed_data_dict["d18o"])) - np.arange(len(trimmed_data_dict["d18o"]))
    if trimmed_data_dict.has_key('ec')==False:
		trimmed_data_dict["ec"] = np.arange(len(trimmed_data_dict["d18o"])) - np.arange(len(trimmed_data_dict["d18o"]))
    print filepath, len(trimmed_data_dict["d18o"]),len(trimmed_data_dict["dD"]),len(trimmed_data_dict["2140d17o"]),len(trimmed_data_dict["ec"]),len(trimmed_data_dict["meltrate"])
    for i in trimmed_data_dict.keys():
        trimmed_dict_long[i] = np.concatenate([trimmed_dict_long[i], trimmed_data_dict[i][1:]])
	
trimmed_dict_long["d_excess"] = trimmed_dict_long["dD"]-8*trimmed_dict_long["d18o"]
trimmed_dict_long["fullindex"] = np.arange(len(trimmed_dict_long["dD"]))
for x in trimmed_dict_long["fullindex"]:
	if trimmed_dict_long["depth"][x]>=2120.8:
		trimmed_dict_long["depth"][x]=(trimmed_dict_long["depth"][x]-0.55)

print "full length or d18O, dD and d17O",len(trimmed_dict_long["d18o"]),len(trimmed_dict_long["dD"]),len(trimmed_dict_long["2140d17o"])

##### Flag filter
trimindex = np.arange(len(trimmed_dict_long["d18o"]))
filterd18o = [x for x in trimindex if trimmed_dict_long["flag1_d18o"][x] == '.' and trimmed_dict_long["flag2"][x] == '.' and trimmed_dict_long["flag3"][x] == '.']
filterdD = [x for x in trimindex if trimmed_dict_long["flag1_dD"][x] == '.' and trimmed_dict_long["flag2"][x] == '.' and trimmed_dict_long["flag3"][x] == '.']
filterd17o = [x for x in trimindex if trimmed_dict_long["flag1_2140_d17o"][x] == '.' and trimmed_dict_long["flag2"][x] == '.' and trimmed_dict_long["flag3"][x] == '.']
filteredd18o = trimmed_dict_long["d18o"][filterd18o]
filteredd18odepth = trimmed_dict_long["depth"][filterd18o]
filtereddD = trimmed_dict_long["dD"][filterdD]
filtereddDdepth = trimmed_dict_long["depth"][filterdD]
filtereddexcess = trimmed_dict_long["d_excess"][filterdD]
filteredd17o = trimmed_dict_long["2140d17o"][filterd17o]
filteredd17odepth = trimmed_dict_long["depth"][filterd17o]
filteredec = trimmed_dict_long["ec"][filterdD]
filteredmeltrate = trimmed_dict_long["meltrate"][filterdD]
filteredwater = trimmed_dict_long["water"][filterdD]

graphinteration = np.arange(760)
print graphinteration
graphinteration = graphinteration*3.5+455
print graphinteration

for g in graphinteration:
	fig441 = plt.figure(441)                    
	fig441_ax1 = fig441.add_subplot(311)
	fig441_ax1.plot(trimmed_dict_long["depth"], trimmed_dict_long["d18o"], "g.", filteredd18odepth, filteredd18o, "b.")#, uwdepth, uwd18o, "g.", msdepth, msd18o, "y.", \
	fig441_ax1.axis([g,g+4,-55,-20])
	fig441_ax1.set_ylabel("d18o")
	fig441_ax2 = fig441.add_subplot(312)
	fig441_ax2.plot(trimmed_dict_long["depth"], trimmed_dict_long["dD"], "m.", filtereddDdepth, filtereddD, "r.")#, uwdepth, uwdD, "g.", msdepth, msdD, "y.", \
	fig441_ax2.axis([g,g+4,-450,-150])
	fig441_ax2.set_ylabel("dD")
	fig441_ax3 = fig441.add_subplot(313)
	fig441_ax3.plot(trimmed_dict_long["depth"], trimmed_dict_long["d_excess"], "k.", filtereddDdepth, filtereddexcess, "y.")#, uwdepth, uwdexcess, "g.", msdepth, msdexcess, "y.", \
	fig441_ax3.axis([g,g+4,-20,30])
	fig441_ax3.set_ylabel("d-excess")
	fig441_ax3.set_xlabel("Depth")
	fig441_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))

	fig442 = plt.figure(442)                    
	fig442_ax1 = fig442.add_subplot(311)
	fig442_ax1.plot(trimmed_dict_long["depth"], trimmed_dict_long["2140d17o"], "g.", filteredd17odepth, filteredd17o, "m.")#, uwdepth, uwd18o, "g.", msdepth, msd18o, "y.", \
	fig442_ax1.axis([g,g+4,-30,-10])
	fig442_ax1.set_ylabel("d17o")
	fig442_ax2 = fig442.add_subplot(312)
	fig442_ax2.plot(trimmed_dict_long["depth"], trimmed_dict_long["water"], "b.", filtereddDdepth, filteredwater, "g.")#, uwdepth, uwdD, "g.", msdepth, msdD, "y.", \
	fig442_ax2.axis([g,g+4,10000,40000])
	fig442_ax2.set_ylabel("water")
	fig442_ax3 = fig442.add_subplot(313)
	fig442_ax3.plot(trimmed_dict_long["depth"], trimmed_dict_long["meltrate"], "y.", filtereddDdepth, filteredmeltrate, "b.")#, uwdepth, uwdexcess, "g.", msdepth, msdexcess, "y.", \
	fig442_ax3.axis([g,g+4,0,10])
	fig442_ax3.set_ylabel("meltrate")
	fig442_ax3.set_xlabel("Depth")
	fig442_ax1.set_title("%s" %(os.path.splitext(filepath)[0]))
	
	plt.show()

print "full depth", len(trimmed_dict_long["depth"])
print "filtereddDdepth",len(filtereddDdepth)
print "filteredd18odepth",len(filteredd18odepth)
print "filteredd17odepth",len(filteredd17odepth)

plt.show()

################################################################################
# END OF FULL PROGRAM ##########################################################
################################################################################
