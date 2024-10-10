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
#include "TSVPath.h"

using namespace std;

extern Param *param;

TSVPath::TSVPath(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void TSVPath::Initialize(){
	if (initialized)
		cout << "[TSVPath] Warning: Already initialized!" << endl;
	
	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	EnlargeSize(&widthInvN, &widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
	
	initialized = true;
}

void TSVPath::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[TSVPath] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv;
		// INV
		CalculateGateArea(INV, 1, MIN_NMOS_SIZE*tech.featureSize, tech.pnSizeRatio*MIN_NMOS_SIZE*tech.featureSize, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
		// assme TSV path includes 2*INV and TSV, this function returns TSV path area for one single subarray
		area = (2*hInv*wInv + pow(param->tsvPitch,2)) * (param->numRowSubArray+param->numColSubArray);
		
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
		
		// INV
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
	}
}

void TSVPath::CalculateLatency(double numTSV, double numRead){
	if (!initialized) {
		cout << "[TSVPath] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		rampInput = 1e20;
		double tr;		/* time constant */
		double gm;		/* transconductance */
		double beta;	/* for horowitz calculation */
		double resPullUp, resPullDown;
		double readLatencyIntermediate = 0;
		
		// 1st
		resPullUp = CalculateOnResistance(widthInvP, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capInvOutput + param->tsvCap*numTSV);       
		gm = CalculateTransconductance(widthInvP, PMOS, tech);  
		beta = 1 / (resPullUp * gm);
		readLatency += horowitz(tr, beta, rampInput, NULL);
		
		// 2nd
		resPullDown = param->tsvRes*numTSV;
		tr = resPullDown * capInvInput; 
		readLatency += horowitz(tr, beta, rampInput, NULL);		
		
		if (param->synchronous) {
			readLatency  = ceil(readLatency*clkFreq);	//#cycles
		}
		readLatency *= numRead;
	}
}

void TSVPath::CalculatePower(double numTSV, double numRead) {
	if (!initialized) {
		cout << "[TSVPath] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		/* Leakage power */
		leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * (param->numRowSubArray+param->numColSubArray);

		/* Dynamic energy */
		readDynamicEnergy += (capInvInput) * tech.vdd * tech.vdd;    
        readDynamicEnergy += (capInvOutput + param->tsvCap) * tech.vdd * tech.vdd;  
		readDynamicEnergy += (capInvInput) * tech.vdd * tech.vdd;
		
		readDynamicEnergy *= numRead;
		
		if(param->validated){
				readDynamicEnergy *= param->delta; 	// switching activity of adder, delta = 0.15 by default
		}
	}
}

void TSVPath::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}
