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
*   Pai-Yu Chen	    Email: pchen72 at asu dot edu 
*                    
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cmath>
#include <iostream>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Sigmoid.h"

using namespace std;

extern Param *param;

Sigmoid::Sigmoid(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), mux(_inputParameter, _tech, _cell), muxDecoder(_inputParameter, _tech, _cell), wlDecoder(_inputParameter, _tech, _cell), colDecoder(_inputParameter, _tech, _cell), senseAmp(_inputParameter, _tech, _cell), colDecoderDriver(_inputParameter, _tech, _cell), voltageSenseAmp(_inputParameter, _tech, _cell), precharger(_inputParameter, _tech, _cell), sramWriteDriver(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}


void Sigmoid::Initialize(bool _SRAM, int _numYbit, int _numEntry, int _numFunction, double _clkFreq) {
	if (initialized)
		cout << "[Sigmoid] Warning: Already initialized!" << endl;
	
	SRAM = _SRAM;
	numYbit = _numYbit;               // # of y bit
	numEntry = _numEntry;             // # of (x,y) entry
	numFunction = _numFunction;       // # of sigmoid functions that can be processed in parallel
	clkFreq = _clkFreq;
	numCell = numYbit * numEntry;     // # of memory cell in each single sigmoid function
	
	// INV
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	double hInv, wInv;
	// INV
	CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);

	if (SRAM) { // calculate each memory cell size
		hUnit = hInv + cell.heightInFeatureSize * tech.featureSize;
		// 1.4 update?
		// hUnit = cell.heightInFeatureSize * tech.featureSize;
		wUnit = MAX(wInv * 3, cell.widthInFeatureSize * tech.featureSize) * numYbit;
	} else {	// RRAM
		hUnit = cell.heightInFeatureSize * tech.featureSize;
		wUnit = cell.widthInFeatureSize * tech.featureSize * numYbit;
	}

	if (SRAM) { // initialize peripheral ckt for sigmoid function
		// 1.4 update
		numCol = numYbit;
		numRow = numEntry;

		capRow1 = wUnit * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
		capCol =  hUnit * numEntry * 0.2e-15/1e-6;

		// 1.4 update: SRAM parameters
		
		resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
		capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech);
		capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateDrainCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, PMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateGateCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech);
		capRow1 += 2*CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
		if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;
		else capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;	
		
		resRow = wUnit * param->Metal0_unitwireresis ;
		resCol =  hUnit * numEntry * param->Metal1_unitwireresis;

		// 1.4 update: consider overlap capacitance for FinFET
		if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;	

		// 1.4 update
		// SRAM write driver
		// Precharger 
		precharger.Initialize(numYbit, hUnit * numEntry * param->Metal1_unitwireresis, 1, numYbit, numYbit);
		sramWriteDriver.Initialize(numYbit, 1, numYbit);


		wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)numEntry)), false, false);      // wlDecoder to give x values to sigmoid function
		senseAmp.Initialize(numYbit, false, cell.minSenseVoltage, wUnit/numYbit, clkFreq, 1);    // just assign one S/A
	} else {

		// 1.4 update
		numCol = numYbit;
		numRow = numEntry;
		resRow = wUnit * param->Metal0_unitwireresis ;

		wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)numEntry)), false, false);      // wlDecoder to give x values to sigmoid function
		voltageSenseAmp.Initialize(numYbit, clkFreq);
	}

	initialized = true;
}


void Sigmoid::CalculateUnitArea(AreaModify _option) {      // firstly calculate single sigmoid unit area
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {

		areaUnit = 0;
		
		wlDecoder.CalculateArea(NULL, NULL, NONE);
		
		if (SRAM) { // initialize peripheral ckt for sigmoid function
			senseAmp.CalculateArea(NULL, NULL, NONE);
			// 1.4 update
			// precharger.CalculateArea(NULL, NULL, NONE);
			// sramWriteDriver.CalculateArea(NULL, NULL, NONE);

		} else {
			voltageSenseAmp.CalculateUnitArea();
			voltageSenseAmp.CalculateArea(NULL);
		}
		
		areaUnit += (hUnit * wUnit) * numEntry ;    
		areaUnit += wlDecoder.area + senseAmp.area + voltageSenseAmp.area;



		// 1.4 update

		// SRAM write driver
		// Precharger 
		/* added part (start) */ 
		// areaUnit += wlDecoder.area + senseAmp.area + voltageSenseAmp.area + precharger.area + sramWriteDriver.area;
		/* added part (end) */ 


		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;
			default:    // NONE
				break;
		}
		
	}
}


void Sigmoid::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {     // assign multiple sigmoid unit to operate in parallel
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		area = 0;
		height = 0;
		width = 0;
		area = areaUnit * numFunction;
		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area/width;
		} else {
			height = _newHeight;
            		width = area/height;
		}
		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;  
			default:    // NONE
				break;
		}
	}
}

void Sigmoid::CalculateLatency(double numRead) {
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;

		// 1.4 update: relocate the SRAM parameters to the initialization 

		if (SRAM) {

			// 1.4 update: update the row decoder arguments
			wlDecoder.CalculateLatency(1e20, 0, capSRAMCell, resRow, numYbit, 1, 1);
			senseAmp.CalculateLatency(1);

			// 1.4 update

			/* added part (start) */ 
			
			// 1.4 update : new bitline model
			double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double BLCap_perCell = capCol / numRow;
			double BLRes_perCell = resCol / numRow;
			double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;
			colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));  

			// readLatency = wlDecoder.readLatency + senseAmp.readLatency + colDelay + precharger.readLatency;

			/* added part (end)*/

			readLatency = wlDecoder.readLatency + senseAmp.readLatency;
		} else {	// RRAM
			// Assuming no delay on RRAM wires

			// 1.4 update: update the row decoder arguments
			wlDecoder.CalculateLatency(1e20, 0, capCellAccess, resRow, numYbit, 1, 1);
			voltageSenseAmp.CalculateLatency(0, 1);
			readLatency = wlDecoder.readLatency + voltageSenseAmp.readLatency;
		}
		if (param->synchronous) {
			readLatency = ceil(readLatency*clkFreq);
		}
		readLatency *= numRead;
	}
}

void Sigmoid::CalculatePower(double numRead) {
	if (!initialized) {
		cout << "[Sigmoid] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;

		if (SRAM) {
			wlDecoder.CalculatePower(1,1);
			senseAmp.CalculatePower(1);
			
			readDynamicEnergy += wlDecoder.readDynamicEnergy + senseAmp.readDynamicEnergy;
			
			// 1.4 update: precharger, WL energy, BL energy?
			// precharger.CalculatePower(1);
			// readDynamicEnergy += precharger.readDynamicEnergy;

			// Array leakage (assume 2 INV)

			// 1.4 update: for compatiblity with FinFET and beyond
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS* ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize,
					cell.widthSRAMCellPMOS* ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			leakage *= numCell;
			leakage += wlDecoder.leakage;
			leakage += senseAmp.leakage;

			// 1.4 update: precharger, WL energy, BL energy?
			// sramWriteDriver.CalculatePower(1);
			// leakage += precharger.leakage;
			// leakage += sramWriteDriver.leakage;
			
						
		} else {	// RRAM
			wlDecoder.CalculatePower(1,1);
			voltageSenseAmp.CalculatePower(1);
			readDynamicEnergy += voltageSenseAmp.readDynamicEnergy + wlDecoder.readDynamicEnergy;

			leakage += voltageSenseAmp.leakage;
			leakage += wlDecoder.leakage;
		}
		readDynamicEnergy *= numRead*numFunction;
	}
}

void Sigmoid::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void Sigmoid::SaveOutput(const char* str) {
	FunctionUnit::SaveOutput(str);
}

