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

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "math.h"
#include "Param.h"

Param::Param() {
	/***************************************** user defined design options and parameters *****************************************/
	operationmode = 2;     		// 1: conventionalSequential (Use several multi-bit RRAM as one synapse)
								// 2: conventionalParallel (Use several multi-bit RRAM as one synapse)
	
	memcelltype = 2;        	// 1: cell.memCellType = Type::SRAM
								// 2: cell.memCellType = Type::RRAM
								// 3: cell.memCellType = Type::FeFET
	
	accesstype = 1;         	// 1: cell.accessType = CMOS_access
								// 2: cell.accessType = BJT_access
								// 3: cell.accessType = diode_access
								// 4: cell.accessType = none_access (Crossbar Array)
	
	transistortype = 1;     	// 1: inputParameter.transistorType = conventional
								// 2: inputParameter.transistorType = FET_2D
								// 3: inputParameter.transistorType = TFET
	
	deviceroadmap = 2;      	// 1: inputParameter.deviceRoadmap = HP
								// 2: inputParameter.deviceRoadmap = LSTP
								
	globalBufferType = false;    // false: register file
								// true: SRAM
								
	tileBufferType = false;      // false: register file
								// true: SRAM
								
	peBufferType = false;        // false: register file
								// true: SRAM
	
	chipActivation = true;      // false: activation (reLu/sigmoid) inside Tile
								// true: activation outside Tile
								
	reLu = true;                // false: sigmoid
								// true: reLu
								
	novelMapping = true;        // false: conventional mapping
								// true: novel mapping
	
	/*** algorithm weight range, the default wrapper (based on WAGE) has fixed weight range of (-1, 1) ***/
	algoWeightMax = 1;
	algoWeightMin = -1;
	
	/*** conventional hardware design options ***/
	clkFreq = 1e9;                      // Clock frequency
	featuresize = 40e-9;                // Wire width for subArray simulation
	temp = 301;                         // Temperature (K)
	technode = 32;                      // Technology
	wireWidth = 40;                     // wireWidth of the cell for Accuracy calculation
	globalBusDelayTolerance = 0.1;      // to relax bus delay for global H-Tree (chip level: communication among tiles), if tolerance is 0.1, the latency will be relax to (1+0.1)*optimalLatency (trade-off with energy)
	localBusDelayTolerance = 0.1;       // to relax bus delay for global H-Tree (tile level: communication among PEs), if tolerance is 0.1, the latency will be relax to (1+0.1)*optimalLatency (trade-off with energy)
	treeFoldedRatio = 4;                // the H-Tree is assumed to be able to folding in layout (save area)
	maxGlobalBusWidth = 2048;           // the max buswidth allowed on chip level (just a upper_bound, the actual bus width is defined according to the auto floorplan)

	numRowSubArray = 128;               // # of rows in single subArray
	numColSubArray = 128;               // # of columns in single subArray
	
	/*** option to relax subArray layout ***/
	relaxArrayCellHeight = 0;           // relax ArrayCellHeight or not
	relaxArrayCellWidth = 0;            // relax ArrayCellWidth or not
	
	numColMuxed = 8;                    // How many columns share 1 ADC (for eNVM and FeFET) or parallel SRAM
	levelOutput = 32;                   // # of levels of the multilevelSenseAmp output, should be in 2^N forms; e.g. 32 levels --> 5-bit ADC
	cellBit = 4;                        // precision of memory device 
	
	/*** parameters for SRAM ***/
	heightInFeatureSizeSRAM = 8;        // SRAM Cell height in feature size
	widthInFeatureSizeSRAM = 20;        // SRAM Cell width in feature size
	widthSRAMCellNMOS = 2.08;                              
	widthSRAMCellPMOS = 1.23;
	widthAccessCMOS = 1.31;
	minSenseVoltage = 0.1;
	
	/*** parameters for analog synaptic devices ***/
	heightInFeatureSize1T1R = 4;        // 1T1R Cell height in feature size
	widthInFeatureSize1T1R = 4;         // 1T1R Cell width in feature size
	heightInFeatureSizeCrossbar = 2;    // Crossbar Cell height in feature size
	widthInFeatureSizeCrossbar = 2;     // Crossbar Cell width in feature size
	
	resistanceOn = 100e3;               // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	resistanceOff = 100e3*10;           // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	maxConductance = (double) 1/resistanceOn;
	minConductance = (double) 1/resistanceOff;
	
	readVoltage = 0.5;	                // On-chip read voltage for memory cell
	readPulseWidth = 10e-9;             // read pulse width in sec
	accessVoltage = 1.1;                // Gate voltage for the transistor in 1T1R
	resistanceAccess = 15e3;            // resistance of access CMOS in 1T1R
	
	/***************************************** user defined design options and parameters *****************************************/
	
	
	
	/***************************************** Initialization of parameters NO need to modify *****************************************/
	
	if (memcelltype == 1) {
		cellBit = 1;             // force cellBit = 1 for all SRAM cases
	} 
	

	/*** initialize operationMode as default ***/
	conventionalParallel = 0;
	conventionalSequential = 0;
	BNNparallelMode = 0;                // parallel BNN
	BNNsequentialMode = 0;              // sequential BNN
	XNORparallelMode = 0;           	// parallel XNOR
	XNORsequentialMode = 0;         	// sequential XNOR
	switch(operationmode) {
		case 6:	    XNORparallelMode = 1;               break;     
		case 5:	    XNORsequentialMode = 1;             break;     
		case 4:	    BNNparallelMode = 1;                break;     
		case 3:	    BNNsequentialMode = 1;              break;    
		case 2:	    conventionalParallel = 1;           break;     
		case 1:	    conventionalSequential = 1;         break;    
		case -1:	break;
		default:	exit(-1);
	}
	
	
	/*** parallel read ***/
	parallelRead = 0;
	if(conventionalParallel || BNNparallelMode || XNORparallelMode) {
		parallelRead = 1;
	} else {
		parallelRead = 0;
	}
	
	/*** Initialize interconnect wires ***/
	switch(wireWidth) {
		case 200: 	AR = 2.10; Rho = 2.42e-8; break;
		case 100:	AR = 2.30; Rho = 2.73e-8; break;
		case 50:	AR = 2.34; Rho = 3.91e-8; break;
		case 40:	AR = 1.90; Rho = 4.03e-8; break;
		case 32:	AR = 1.90; Rho = 4.51e-8; break;
		case 22:	AR = 2.00; Rho = 5.41e-8; break;
		case 14:	AR = 2.10; Rho = 7.43e-8; break;
		case -1:	break;	// Ignore wire resistance or user define
		default:	exit(-1); puts("Wire width out of range"); 
	}
	
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

