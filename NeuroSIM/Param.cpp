/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "math.h"
#include "Param.h"
#include "constant.h"

using namespace std;

Param::Param() {
	/***************************************** user defined design options and parameters *****************************************/
	memcelltype = 1;        	// 1: cell.memCellType = Type::SRAM
								// 2: cell.memCellType = Type::RRAM
								// 3: cell.memCellType = Type::FeFET
								// 4: cell.memCellType = Type::STTMRAM
								// 5: cell.memCellType = Type::SOTMRAM
								// 6: cell.memCellType = Type::eDRAM
	
	transistortype = 1;     	// 1: inputParameter.transistorType = conventional
	
	deviceroadmap = 2;      	// 1: inputParameter.deviceRoadmap = HP
								// 2: inputParameter.deviceRoadmap = LSTP
								
	validated = true;			// false: no calibration factors
								// true: validated by silicon data (wiring area in layout, gate switching activity, post-layout performance drop...)

	/*** conventional hardware design options ***/
	clkFreq = 1e9;                      // Clock frequency
	temp = 300;                         // Temperature (K)
	// technode: 130	 --> wireWidth: 175
	// technode: 90		 --> wireWidth: 110
	// technode: 65      --> wireWidth: 105
	// technode: 45      --> wireWidth: 80
	// technode: 32      --> wireWidth: 56
	// technode: 22      --> wireWidth: 40
	// technode: 14      --> wireWidth: 25
	// technode: 10, 7   --> wireWidth: 18

	// 1.4 update: special layout
	speciallayout=1;

	// 1.4 update
	sizingfactor_WLdecoder=1; // sizing for the final driver of WLdecoder in rowdecoder.cpp (important for technology scaling)

	// 1.4 update: switchmatrix parameter tuning
	newswitchmatrixsizeratio=6;
	switchmatrixsizeratio=0.1;

	// 1.4 update: buffer parameters : change it if you have to insert buffers
	buffernumber=0;
	buffersizeratio=0;										 
// For Linux Automation (test for 1.4 update)
const int 
tech=8
;

	// 1.4 update: new technology node added

	switch (tech){
		case 0: technode = 130; break;  
		case 1:technode =  90; break;  
		case 2: technode =  65; break;  
		case 3: technode =  45; break;  
		case 4: technode =  32; break;  
		case 5: technode =  22; break; 
		case 6:technode =  14; break;  
		case 7: technode =  10; break;  
		case 8: technode = 7; break;  
		case 9: technode =  5; break;  
		case 10: technode =  3;  break; 
		case 11: technode =  2;  break;  
		case 12: technode =  1; break; 
	} 

// 1.4 update: new wire width, barrierthickness

	switch (technode){
		case 130: 	Metal0 = 175; Metal1 = 175; wireWidth = 175; barrierthickness = 10.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 90: 	Metal0 = 110; Metal1 = 110; wireWidth=110; barrierthickness = 10.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 65:	Metal0 = 105; Metal1 = 105; wireWidth=105; barrierthickness = 7.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 45:	Metal0 = 80; Metal1 = 80; wireWidth=80; barrierthickness = 5.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 32:	Metal0 = 56; Metal1 = 56; wireWidth=56; barrierthickness = 4.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 22:	Metal0 = 40; Metal1 = 40; wireWidth=40; barrierthickness = 2.5e-9 ; featuresize = wireWidth*1e-9; break; 
		case 14:	Metal0 = 32; Metal1 = 39; wireWidth=32; barrierthickness = 2.5e-9 ; featuresize = wireWidth*1e-9; break;  
		case 10:	Metal0 = 22; Metal1 = 32; wireWidth=22; barrierthickness = 2.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 7:		Metal0 = 20; Metal1 = 28.5; wireWidth=20; barrierthickness = 2.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 5:		Metal0 = 15; Metal1 = 17; wireWidth=15; barrierthickness = 2.0e-9 ; featuresize = wireWidth*1e-9; break;  
		case 3:		Metal0 = 12; Metal1 = 16; wireWidth=12; barrierthickness = 1.5e-9 ; featuresize = wireWidth*1e-9; break; 
		case 2:		Metal0 = 10; Metal1 = 11.5; wireWidth=10; barrierthickness = 0.5e-9 ; featuresize = wireWidth*1e-9; break;  
		case 1:		Metal0 = 8; Metal1 = 10; wireWidth=8; barrierthickness = 0.2e-9 ; featuresize = wireWidth*1e-9; break;    
		case -1:	break;	
		default:	exit(-1); puts("Wire width out of range"); 
	}
	
	numRowSubArray = 256;               // # of rows in single subArray
	numColSubArray = 512;               // # of columns in single subArray


	/*** parameters for SRAM ***/
	// due the scaling, suggested SRAM cell size above 22nm: 160F^2
	// SRAM cell size at 14nm: 300F^2
	// SRAM cell size at 10nm: 400F^2
	// SRAM cell size at 7nm: 600F^2
	heightInFeatureSizeSRAM = 10;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 28;        // SRAM Cell width in feature size  
	widthSRAMCellNMOS = 2;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;

	// 1.4 update : SRAM size update

	if (technode>14){
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 10;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 28;        // SRAM Cell width in feature size  
	}
	else if (technode==14){ // Samsung 14 nm 
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 10.6;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 30.8;        // SRAM Cell width in feature size  
	}
	else if (technode==10){ // TSMC 10 nm 
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 12.8;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 31.25;        // SRAM Cell width in feature size  
	}
	else if (technode==7){ // TSMC IEDM 2016
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 16;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 34.43;        // SRAM Cell width in feature size  
	}
	else if (technode==5){ // IRDS
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 19.2;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 43.75;        // SRAM Cell width in feature size  
	}
	else if (technode==3){ // IRDS
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 30;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 68.26;        // SRAM Cell width in feature size  
	}
	else if (technode==2){ // IRDS
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 42;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 120;// 111.42;        // SRAM Cell width in feature size  
	}
	else if (technode==1){ // IRDS
	widthSRAMCellNMOS = 1;                            
	widthSRAMCellPMOS = 1;
	widthAccessCMOS = 1;
	heightInFeatureSizeSRAM = 80;        // SRAM Cell height in feature size  
	widthInFeatureSizeSRAM = 144;        // SRAM Cell width in feature size  
	}


	minSenseVoltage = 0.1;
	
	/*** parameters for analog synaptic devices ***/
	heightInFeatureSize1T1R = 4;        // 1T1R Cell height in feature size
	widthInFeatureSize1T1R = 12;         // 1T1R Cell width in feature size
	heightInFeatureSizeCrossbar = 2;    // Crossbar Cell height in feature size
	widthInFeatureSizeCrossbar = 2;     // Crossbar Cell width in feature size
	
	resistanceOn = 6e3;               // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	resistanceOff = 6e3*150;           // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	maxConductance = (double) 1/resistanceOn;
	minConductance = (double) 1/resistanceOff;
	
	readVoltage = 0.5;	                // On-chip read voltage for memory cell
	readPulseWidth = 10e-9;             // read pulse width in sec
	accessVoltage = 1.1;                // Gate voltage for the transistor in 1T1R
	resistanceAccess = resistanceOn*IR_DROP_TOLERANCE;            // resistance of access CMOS in 1T1R
	
	gateCapFeFET = 2.1717e-18;	        // Gate capacitance of FeFET (F)
	polarization = 20;                  // polarization of FeFET (uC/cm^2)
	writeVoltage = 2;					// Enable level shifer if writeVoltage > 1.5V
	writePulseWidth = 50e-9;
	
	/*** Calibration parameters ***/
	if(validated){
		alpha = 1.44;	// wiring area of level shifter
		beta = 1.4;  	// latency factor of sensing cycle
		gamma = 0.5; 	// switching activity of DFF in shifter-add and accumulator
		delta = 0.15; 	// switching activity of adder 
		epsilon = 0.05; // switching activity of control circuits
		zeta = 1.22; 	// post-layout energy increase
	}		
	
	/***************************************** user defined design options and parameters *****************************************/
	
	
	
	/***************************************** Initialization of parameters NO need to modify *****************************************/
	
	if (memcelltype == 1) {
		cellBit = 1;             // force cellBit = 1 for all SRAM cases
	} 
	

	
	/*** Initialize interconnect wires ***/

	// 1.4 update: wirewidth
	if (wireWidth >= 175) {
		AR = 1.6; 
		Rho = 2.01*1e-8;
	} else if ((110 <= wireWidth) &&  (wireWidth < 175)) {
		AR = 1.6; 
		Rho = 2.20*1e-8;
	} else if ((105 <= wireWidth) &&  (wireWidth < 110)) {
		AR = 1.7; 
		Rho = 2.21*1e-8;
	} else if ((80 <= wireWidth) &&  (wireWidth < 105)){
		AR = 1.7; 
		Rho = 2.37*1e-8;
	} else if ((56 <= wireWidth) &&  (wireWidth < 80)){
		AR = 1.8; 
		Rho = 2.63*1e-8;
	} else if ((40 <= wireWidth) &&  (wireWidth < 56)) {
		AR = 1.9; 
		Rho = 2.97*1e-8;
	} else if ((32 <= wireWidth) &&  (wireWidth < 40)) {
		AR = 2.0; 
		Rho = 3.25*1e-8;
	} else if ((22 <= wireWidth) &&  (wireWidth < 32)){
		AR = 2.00; Rho = 3.95*1e-8;
	} else if ((20 <= wireWidth) &&  (wireWidth < 22)){
		AR = 2.00; Rho = 4.17*1e-8; 
	} else if ((15 <= wireWidth) &&  (wireWidth < 20)){
		AR = 2.00; Rho = 4.98*1e-8; 
	} else if ((12 <= wireWidth) &&  (wireWidth < 15)){
		AR = 2.00; Rho = 5.8*1e-8; 
	} else if ((10 <= wireWidth) &&  (wireWidth < 12)){
		// AR = 3.00; Rho = 6.65*1e-8; 
		AR = 2.00; Rho = 6.61*1e-8; 
	} else if ((8 <= wireWidth) &&  (wireWidth < 10)){
		AR = 3.00; Rho = 7.87*1e-8; 
	} else {
		exit(-1); puts("Wire width out of range"); 
	}
	
	Rho = Rho * 1 / (1- ( (2*AR*wireWidth + wireWidth)*barrierthickness / (AR*pow(wireWidth,2) ) ));
	
	// 1.4 update: Metal0
	if (Metal0 >= 175) {
		AR_Metal0 = 1.6; 
		Rho_Metal0 = 2.01*1e-8;
	} else if ((110 <= Metal0) &&  (Metal0< 175)) {
		AR_Metal0 = 1.6; 
		Rho_Metal0 = 2.20*1e-8;
	} else if ((105 <= Metal0) &&  (Metal0< 110)){
		AR_Metal0 = 1.7; 
		Rho_Metal0 = 2.21*1e-8;
	} else if ((80 <= Metal0) &&  (Metal0< 105)) {
		AR_Metal0 = 1.7; 
		Rho_Metal0 = 2.37*1e-8;
	} else if ((56 <= Metal0) &&  (Metal0< 80)){
		AR_Metal0 = 1.8; 
		Rho_Metal0 = 2.63*1e-8;
	} else if ((40 <= Metal0) &&  (Metal0< 56)) {
		AR_Metal0 = 1.9; 
		Rho_Metal0 = 2.97*1e-8;
	} else if ((32 <= Metal0) &&  (Metal0< 40)) {
		AR_Metal0 = 2.0; 
		Rho_Metal0 = 3.25*1e-8;
	} else if ((22 <= Metal0) &&  (Metal0< 32)){
		AR_Metal0 = 2.00; Rho_Metal0 = 3.95*1e-8;
	} else if ((20 <= Metal0) &&  (Metal0< 22)){
		AR_Metal0 = 2.00; Rho_Metal0 = 4.17*1e-8; 
	} else if ((15 <= Metal0) &&  (Metal0< 20)){
		AR_Metal0 = 2.00; Rho_Metal0 = 4.98*1e-8; 
	} else if ((12 <= Metal0) &&  (Metal0< 15)){
		AR_Metal0 = 2.00; Rho_Metal0 = 5.8*1e-8; 
	} else if ((10 <= Metal0) &&  (Metal0< 12)){
		// AR_Metal0 = 3.00; Rho_Metal0 = 6.65*1e-8; 
		AR_Metal0 = 2.00; Rho_Metal0 = 6.61*1e-8; 
	} else if ((8 <= Metal0) &&  (Metal0< 10)){
		AR_Metal0 = 3.00; Rho_Metal0 = 7.87*1e-8; 
	} else {
		exit(-1); puts("Wire width out of range"); 
	}
	
	Rho_Metal0 = Rho_Metal0 * 1 / (1- ( (2*AR_Metal0*Metal0 + Metal0)*barrierthickness / (AR_Metal0*pow(Metal0,2) ) ));
	
	
	// 1.4 update: Metal1
	if (Metal1 >= 175) {
		AR_Metal1 = 1.6; 
		Rho_Metal1 = 2.01*1e-8;
	} else if ((110 <= Metal1) &&  (Metal1 < 175)) {
		AR_Metal1 = 1.6; 
		Rho_Metal1 = 2.20*1e-8;
	} else if ((105 <= Metal1) &&  (Metal1 < 110)) {
		AR_Metal1 = 1.7; 
		Rho_Metal1 = 2.21*1e-8;
	} else if ((80 <= Metal1) &&  (Metal1 <105)) {
		AR_Metal1 = 1.7; 
		Rho_Metal1 = 2.37*1e-8;
	} else if ((56 <= Metal1) &&  (Metal1 < 80)) {
		AR_Metal1 = 1.8; 
		Rho_Metal1 = 2.63*1e-8;
	} else if ((40 <= Metal1) &&  (Metal1 < 56)){
		AR_Metal1 = 1.9; 
		Rho_Metal1 = 2.97*1e-8;
	} else if ((32 <= Metal1) &&  (Metal1 < 40)) {
		AR_Metal1 = 2.0; 
		Rho_Metal1 = 3.25*1e-8;
	} else if ((22 <= Metal1) &&  (Metal1 < 32)){
		AR_Metal1 = 2.00; Rho_Metal1 = 3.95*1e-8;
	} else if ((20 <= Metal1) &&  (Metal1 < 22)){
		AR_Metal1 = 2.00; Rho_Metal1 = 4.17*1e-8; 
	} else if ((15 <= Metal1) &&  (Metal1 < 20)){
		AR_Metal1 = 2.00; Rho_Metal1 = 4.98*1e-8; 
	} else if ((12 <= Metal1) &&  (Metal1 < 15)){
		AR_Metal1 = 2.00; Rho_Metal1 = 5.8*1e-8; 
	} else if ((10 <= Metal1) &&  (Metal1 < 12)){
		// AR_Metal1 = 3.00; Rho_Metal1 = 6.65*1e-8; 
		AR_Metal1 = 2.00; Rho_Metal1 = 6.61*1e-8;
	} else if ((8 <= Metal1) &&  (Metal1 < 10)){
		AR_Metal1 = 3.00; Rho_Metal1 = 7.87*1e-8; 
	} else {
		exit(-1); puts("Wire width out of range"); 
	}

	Rho_Metal1 = Rho_Metal1 * 1 / (1- ( (2*AR_Metal1*Metal1 + Metal1)*barrierthickness / (AR_Metal1*pow(Metal1,2) ) ));
	
	Metal0_unitwireresis =  Rho_Metal0 / ( Metal0*1e-9 * Metal0*1e-9 * AR_Metal0 );
	Metal1_unitwireresis =  Rho_Metal1 / ( Metal1*1e-9 * Metal1*1e-9 * AR_Metal1 );
	


	if (memcelltype == 1) {
		wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeSRAM;
		wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeSRAM;
	} else {
		if (accesstype == 1) {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSize1T1R;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSize1T1R;
		} else {
			wireLengthRow = wireWidth * 1e-9 * heightInFeatureSizeCrossbar;
			wireLengthCol = wireWidth * 1e-9 * widthInFeatureSizeCrossbar;
		}
	}
	Rho *= (1+0.00451*abs(temp-300));
	if (wireWidth == -1) {
		unitLengthWireResistance = 1.0;	// Use a small number to prevent numerical error for NeuroSim
		wireResistanceRow = 0;
		wireResistanceCol = 0;
	} else {
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
		wireResistanceRow = unitLengthWireResistance * wireLengthRow;
		wireResistanceCol = unitLengthWireResistance * wireLengthCol;
	}
	/***************************************** Initialization of parameters NO need to modify *****************************************/
}

