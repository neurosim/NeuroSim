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
#include "MultilevelSAEncoder.h"

using namespace std;

extern Param *param;

MultilevelSAEncoder::MultilevelSAEncoder(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	initialized = false;
}

void MultilevelSAEncoder::Initialize(int _numLevel, int _numEncoder){
	if (initialized)
		cout << "[MultilevelSAEncoder] Warning: Already initialized!" << endl;
	
	numEncoder = _numEncoder;      // number of encoder needed
	numLevel= _numLevel;           // number of levels from MultilevelSA
	numInput = ceil(numLevel/2);       // number of NAND gate in encoder
	numGate = ceil(log2(numLevel));      // number of NAND gate in encoder 
	
	// 1.4 update - updated

	widthInvN = MIN_NMOS_SIZE * tech.featureSize;
	widthInvP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	widthNandN = 4 * MIN_NMOS_SIZE * tech.featureSize;
	widthNandP = 2 * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

	// not enlarge it to reduce the dynamic energy consumption

	// EnlargeSize(&widthInvN, &widthInvP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
	// EnlargeSize(&widthNandN, &widthNandP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);

	initialized = true;
}

void MultilevelSAEncoder::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[MultilevelSAEncoder] Error: Require initialization first!" << endl;
	} else {
        	double wEncoder, hEncoder, wNand, hNand, wNandLg, hNandLg, wInv, hInv;
		area = 0;
		height = 0;

		// 1.4 update - updated
		// NAND2
		if ((tech.featureSize <= 2e-9) && param->speciallayout) {CalculateGateArea(NAND, 2, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand, &wNand);}
		else {CalculateGateArea(NAND, 2, widthNandN, widthNandP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNand, &wNand);}
		// INV
		CalculateGateArea(INV, 1, widthInvN, widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
		// Large NAND in Encoder
		CalculateGateArea(NAND, numInput, widthNandN, widthNandP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hNandLg, &wNandLg);
		
		// 230920 update
		wEncoder = wInv + wNand + wNandLg; // latch part with two inverters are included in the senseamp estimation
		hEncoder = ((wInv*hInv + wNand*hNand)*(numLevel-1) + wNandLg*numGate*hNandLg)/wEncoder  ;

		if (_newWidth && _option==NONE) {
			int numEncoderPerRow = (int)ceil(_newWidth/wEncoder);
			if (numEncoderPerRow > numEncoder) {
				numEncoderPerRow = numEncoder;
			}
			int numRowEncoder = (int)ceil((double)numEncoder / numEncoderPerRow);
			width = MAX(_newWidth, wEncoder);
			height = hEncoder * numRowEncoder;
		} else if (_newHeight && _option==NONE) {
			int numEncoderPerColumn = (int) ceil(_newHeight/hEncoder);
			if (numEncoderPerColumn > numEncoder) {
				numEncoderPerColumn = numEncoder;
			}
			int numColEncoder = (int)ceil((double)numEncoder / numEncoderPerColumn);
			height = MAX(_newHeight, hEncoder);
			width = wEncoder*numColEncoder;
		}
		area = height * width;
		
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
		
		// Capacitance
		// 1.4 update - updated
		if ((tech.featureSize == 2e-9) && param->speciallayout) { 
			CalculateGateCapacitance_GAA(INV, 1, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, hInv, tech, &capInvInput, &capInvOutput, 1.0/2.0,  4.5/15.0,  4.5/15.0); 
		}
	    else if ((tech.featureSize == 1e-9) && param->speciallayout) {
			CalculateGateCapacitance_GAA(INV, 1, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize,hInv, tech, &capInvInput, &capInvOutput, 10.5/16.0,  4.5/10.0,  4.5/10.0); 
		}	
		else {
		CalculateGateCapacitance(INV, 1, widthInvN, widthInvP, hInv, tech, &capInvInput, &capInvOutput);
		}

		// 1.4 update - updated 
		if ((tech.featureSize == 2e-9) && param->speciallayout) { 
			CalculateGateCapacitance_GAA(NAND, 2, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, hNand, tech, &capNandInput, &capNandOutput, 1.0, 22.0/15.0, 8.0/15.0); }
	    else if ((tech.featureSize == 1e-9) && param->speciallayout) {
			CalculateGateCapacitance_GAA(NAND, 2, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, hNand, tech, &capNandInput, &capNandOutput, 1.0, 23.0/15.0, 7.0/15.0); }
		else {
			CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, hNand, tech, &capNandInput, &capNandOutput);
		}
		// Large NAND in Encoder
		CalculateGateCapacitance(NAND, numInput, widthNandN, widthNandP, hNandLg, tech, &capNandLgInput, &capNandLgOutput);
	}
}

void MultilevelSAEncoder::CalculateLatency(double _rampInput, double numRead){
	if (!initialized) {
		cout << "[MultilevelSAEncoder] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		rampInput = _rampInput;
		double tr;		/* time constant */
		double gm;		/* transconductance */
		double beta;	/* for horowitz calculation */
		double resPullUp, resPullDown;
		double readLatencyIntermediate = 0;
		double ramp[10];
		
		ramp[0] = rampInput;

			// 1st INV to NAND2
			resPullDown = CalculateOnResistance(widthInvN, NMOS, inputParameter.temperature, tech) ;
			tr = resPullDown * (capInvOutput + 2* capNandInput);
			gm = CalculateTransconductance(widthInvN, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency += horowitz(tr, beta, ramp[0], &ramp[1]);
			
			// 2nd NAND2 to large NAND
			resPullDown = CalculateOnResistance(widthNandN, NMOS, inputParameter.temperature, tech)* 2;
			if ((tech.featureSize <= 2e-9) && param->speciallayout)
			{resPullDown= resPullDown *3.0/2.0;}
			tr = resPullDown * (capNandOutput + capNandLgInput * numInput);
			gm = CalculateTransconductance(widthNandN, NMOS, tech);
			beta = 1 / (resPullDown * gm);
			readLatency += horowitz(tr, beta, ramp[1], &ramp[2]);
			
			readLatency *= numRead;
			rampOutput = ramp[2];
	}
}

void MultilevelSAEncoder::CalculatePower(double numRead) {
	if (!initialized) {
		cout << "[MultilevelSAEncoder] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		leakage = 0;


		// 1.4 update - updated
		// 230920 update
		leakage =  CalculateGateLeakage(NAND, 2, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * (numLevel-1) * numEncoder
				  + CalculateGateLeakage(NAND, numInput, widthNandN, widthNandP, inputParameter.temperature, tech) * tech.vdd * numGate * numEncoder;
		
		if ((tech.featureSize == 2e-9) && param->speciallayout) { 
			leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * 5 * (numLevel-1) * numEncoder * 1.0/2.0;

		}
	    else if ((tech.featureSize == 1e-9) && param->speciallayout) {
			leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * 5 * (numLevel-1) * numEncoder * 10.5/16.0;
		}
		
		else {
			leakage += CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * tech.vdd * 5 * (numLevel-1) * numEncoder;
		}	

		// 1.4 update - updated
		// can neglect the dynamic energy from the encoding logic, as the toggling activity is negigible compared to the senseamplifier part
		readDynamicEnergy = 0; 
		
		if(param->validated){
			readDynamicEnergy *= param->epsilon; 	// switching activity of control circuits, epsilon = 0.05 by default
		}
		
		if (!readLatency) {
			//cout << "[MultilevelSenseAmp] Error: Need to calculate read latency first" << endl;
		} else {
			readPower = readDynamicEnergy/readLatency;
		}
	}
}

void MultilevelSAEncoder::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}


