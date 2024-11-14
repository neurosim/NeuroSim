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
#include "typedef.h"
#include "formula.h"
#include "XYBus.h"
#include "Param.h"

using namespace std;

extern Param *param;

XYBus::XYBus(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void XYBus::Initialize(int _numRow, int _numCol, double _delaytolerance, double _busWidth, double _clkFreq){
	if (initialized)
		cout << "[HTree] Warning: Already initialized!" << endl;
	
	numRow = _numRow;
	numCol = _numCol;     // num of Row and Col in tile level

	delaytolerance = _delaytolerance;
	busWidth = _busWidth;

	clkFreq = _clkFreq;

	unitLengthWireResistance = param->unitLengthWireResistance;
	unitLengthWireCap = 0.2e-15/1e-6;;   // 0.2 fF/mm
	
	// define min INV resistance and capacitance to calculate repeater size
	widthMinInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthMinInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	CalculateGateArea(INV, 1, widthMinInvN, widthMinInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hMinInv, &wMinInv);
	CalculateGateCapacitance(INV, 1, widthMinInvN, widthMinInvP, hMinInv, tech, &capMinInvInput, &capMinInvOutput);
	
	
	double resOnRep = (CalculateOnResistance(widthMinInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthMinInvP, PMOS, inputParameter.temperature, tech))/2;
	// optimal repeater design to achieve highest speed
	repeaterSize = floor((double)sqrt( (double) resOnRep*unitLengthWireCap/capMinInvInput/unitLengthWireResistance));
	minDist = sqrt(2*resOnRep*(capMinInvOutput+capMinInvInput)/(unitLengthWireResistance*unitLengthWireCap));
	CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hRep, &wRep);
	CalculateGateCapacitance(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, hRep, tech, &capRepInput, &capRepOutput);
	resOnRep = (CalculateOnResistance(MIN_NMOS_SIZE * tech.featureSize * repeaterSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, PMOS, inputParameter.temperature, tech))/2;
	double minUnitLengthDelay = 0.7*(resOnRep*(capRepInput+capRepOutput+unitLengthWireCap*minDist)+0.54*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capRepInput)/minDist;
	double maxUnitLengthEnergy = (capRepInput+capRepOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist;
	
	if (delaytolerance) {   // tradeoff: increase delay to decrease energy
		double delay = 0;
		double energy = 100;
		while(delay<minUnitLengthDelay*(1+delaytolerance)) {
			repeaterSize /=2;
			minDist *= 0.9;
			CalculateGateArea(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hRep, &wRep);
			CalculateGateCapacitance(INV, 1, MIN_NMOS_SIZE * tech.featureSize * repeaterSize, tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, hRep, tech, &capRepInput, &capRepOutput);
			resOnRep = (CalculateOnResistance(MIN_NMOS_SIZE * tech.featureSize * repeaterSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize * repeaterSize, PMOS, inputParameter.temperature, tech))/2;
			delay = 0.7*(resOnRep*(capRepInput+capRepOutput+unitLengthWireCap*minDist)+0.54*unitLengthWireResistance*minDist*unitLengthWireCap*minDist+unitLengthWireResistance*minDist*capRepInput)/minDist;
			energy = (capRepInput+capRepOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist;
		}
	}
	
	widthInvN = MAX(1,repeaterSize) * MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = MAX(1,repeaterSize) * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	
	initialized = true;
}

void XYBus::CalculateArea(double unitHeight, double unitWidth, double foldedratio) {
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
		
		area = 0;
		double wireLengthV = unitHeight*numRow;   // Y Bus
		double wireLengthH = unitWidth*numCol;    // X Bus
		double wireWidV = 0;
		double wireWidH = 0;
		double numRepeater = 0;

        /*** Y Bus ***/
        wireWidthV, unitLengthWireResistanceV = GetUnitLengthRes(wireLengthV);
        numRepeater = ceil(wireLengthV/minDist);
        if (numRepeater > 0) {
            wireWidV += busWidth*wInv/foldedratio;   // which ever stage, the sum of wireWidth should always equal to busWidth (main bus width)
        } else {
            wireWidV += busWidth*wireWidthV/foldedratio;
        }
        area += wireWidV*wireLengthV;
        
        /*** X Bus ***/
        wireWidthH, unitLengthWireResistanceH = GetUnitLengthRes(wireLengthH);
        numRepeater = ceil(wireLengthH/minDist);
        if (numRepeater > 0) {
            wireWidH += busWidth*hInv/foldedratio;   // which ever stage, the sum of wireWidth should always equal to busWidth (main bus width)
        } else {
            wireWidH += busWidth*wireWidthH/foldedratio;
        }
        area += wireWidH*wireLengthH;

		// Capacitance
		// INV
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
		
	}
}

void XYBus::CalculateLatency(int x_end, int y_init, double unitHeight, double unitWidth, double numReadInput, double numReadOutput){
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		double readLatencyInput = 0;
        double readLatencyOutput = 0;
		
		double wireLengthV = unitHeight*(numRow-y_init-1);   // Y Bus
		double wireLengthH = unitWidth*x_end;    // X Bus
		double numRepeater = 0;
		double resOnRep = (CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech))/2;

		// for input: X Bus
        unitLatencyRep = 0.7*(resOnRep*(capInvInput+capInvOutput+unitLengthWireCap*minDist)+0.54*unitLengthWireResistanceH*minDist*unitLengthWireCap*minDist+unitLengthWireResistanceH*minDist*capInvInput)/minDist;
        unitLatencyWire = 0.7*unitLengthWireResistanceH*minDist*unitLengthWireCap*minDist/minDist;
        numRepeater = ceil(wireLengthH/minDist);

        if (numRepeater > 0) {
            readLatencyInput += wireLengthH*unitLatencyRep;
        } else {
            readLatencyInput += wireLengthH*unitLatencyWire;
        }
        if (param->synchronous) {
			readLatencyInput = ceil(readLatencyInput*clkFreq);
		}
		// for output: Y Bus
        unitLatencyRep = 0.7*(resOnRep*(capInvInput+capInvOutput+unitLengthWireCap*minDist)+0.54*unitLengthWireResistanceV*minDist*unitLengthWireCap*minDist+unitLengthWireResistanceV*minDist*capInvInput)/minDist;
        unitLatencyWire = 0.7*unitLengthWireResistanceV*minDist*unitLengthWireCap*minDist/minDist;
        numRepeater = ceil(wireLengthV/minDist);
        if (numRepeater > 0) {
            readLatencyOutput += wireLengthV*unitLatencyRep;
        } else {
            readLatencyOutput += wireLengthV*unitLatencyWire;
        }
		if (param->synchronous) {
			readLatencyOutput = ceil(readLatencyOutput*clkFreq);
		}
		
		readLatency = readLatencyInput * numReadInput + readLatencyOutput * numReadOutput; 	
	}
}

void XYBus::CalculatePower(int x_end, int y_init, double unitHeight, double unitWidth, double numBitInput, double numBitOutput) {
	if (!initialized) {
		cout << "[HTree] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		double readDynamicEnergyInput = 0;
        double readDynamicEnergyOutput = 0;

		// 230920 update
		double wireLengthV = unitHeight*(numRow-y_init-1);   // Y Bus
		double wireLengthH = unitWidth*x_end;    // X Bus
		totalWireLength = wireLengthH + wireLengthV;

		unitLengthLeakage = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd / minDist;
		leakage = unitLengthLeakage * totalWireLength * busWidth;
		unitLengthEnergyRep = (capInvInput+capInvOutput+unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist*0.5;
		unitLengthEnergyWire = (unitLengthWireCap*minDist)*tech.vdd*tech.vdd/minDist*0.5;

		
        // for input: X Bus
        numRepeater = ceil(wireLengthH/minDist);
        if (numRepeater > 0) {
            readDynamicEnergyInput += wireLengthH*unitLengthEnergyRep;
        } else {
            readDynamicEnergyInput += wireLengthH*unitLengthEnergyWire;
        }
        // for output: Y Bus
        numRepeater = ceil(wireLengthV/minDist);
        if (numRepeater > 0) {
            readDynamicEnergyOutput += wireLengthV*unitLengthEnergyRep;
        } else {
            readDynamicEnergyOutput += wireLengthV*unitLengthEnergyWire;
        }

		readDynamicEnergy = readDynamicEnergyInput * numBitInput + readDynamicEnergyOutput * numBitOutput;
	}
}

void XYBus::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

double XYBus::GetUnitLengthRes(double wireLength) {
	double wireWidth, AR, Rho, unitLengthWireResistance, wireResistance;

	if (wireLength/tech.featureSize >= 100000) {
		wireWidth = 4*param->wireWidth;
	} else if ((10000 <= wireLength/tech.featureSize) && (wireLength/tech.featureSize<= 100000)) {
		wireWidth = 2*param->wireWidth;
	} else {
		wireWidth = 1*param->wireWidth;
	}
	
	// 1.4 update 
	double barrierthickness=0;

	if (wireWidth >= 175) {
		AR = 1.6; 
		Rho = 2.01e-8;
		barrierthickness = 10.0e-9 ;
	} else if ((110 <= wireWidth ) && (wireWidth < 175)) {
		AR = 1.6; 
		Rho = 2.20e-8;
		barrierthickness = 10.0e-9 ;
	} else if ((105 <= wireWidth) && (wireWidth< 110)) {
		AR = 1.7; 
		Rho = 2.21e-8;
		barrierthickness = 7.0e-9 ;
	} else if ((80 <= wireWidth) && (wireWidth<105)) {
		AR = 1.7; 
		Rho = 2.37e-8;
		barrierthickness = 5.0e-9 ;
	} else if ((56 <= wireWidth) &&   (wireWidth<80)) {
		AR = 1.8; 
		Rho = 2.63e-8;
		barrierthickness = 4.0e-9 ; 
	} else if ((40 <= wireWidth) &&  (wireWidth<56)) {
		AR = 1.9; 
		Rho = 2.97e-8;
		barrierthickness = 3.0e-9 ;
	} else if ((32 <= wireWidth) &&  (wireWidth< 40)) {
		AR = 2.0; 
		Rho = 3.25e-8;
		barrierthickness = 2.5e-9 ;
	} else if ((22 <= wireWidth) && (wireWidth< 32)){
		AR = 2.00; Rho = 3.95e-8;
		barrierthickness = 2.5e-9 ;
	} else if ((20 <= wireWidth) && (wireWidth< 22)){
		AR = 2.00; Rho = 4.17e-8; 
		barrierthickness = 2.5e-9 ;
	} else if ((15 <= wireWidth) && (wireWidth< 20)){
		AR = 2.00; Rho = 4.98e-8; 
		barrierthickness = 2.0e-9 ; 
	} else if ((12 <= wireWidth) && (wireWidth< 15)){
		AR = 2.00; Rho = 5.8e-8; 
		 barrierthickness = 1.5e-9 ;
	} else if ((10 <= wireWidth) && (wireWidth< 12)){
		AR = 3.00; Rho = 6.65e-8; 
		barrierthickness = 0.5e-9 ;
	} else if ((8 <= wireWidth) && (wireWidth< 10)){
		AR = 3.00; Rho = 7.87e-8; 
		barrierthickness = 0.5e-9 ;
	} else {
		exit(-1); puts("Wire width out of range"); 
	}

	Rho = Rho * 1 / (1- ( (2*AR*wireWidth + wireWidth)*barrierthickness / (AR*pow(wireWidth,2) ) ));
	

	Rho *= (1+0.00451*(param->temp-300));
	if (wireWidth == -1) {
		unitLengthWireResistance = 1.0;	// Use a small number to prevent numerical error for NeuroSim
	} else {
		unitLengthWireResistance =  Rho / ( wireWidth*1e-9 * wireWidth*1e-9 * AR );
	}
	
	return wireWidth, unitLengthWireResistance;
}


