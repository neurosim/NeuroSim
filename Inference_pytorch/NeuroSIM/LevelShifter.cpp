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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// This LevelShifter is used for eNVM mode////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include <cmath>
#include <iostream>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "LevelShifter.h"

using namespace std;

extern Param *param;

LevelShifter::LevelShifter(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), FunctionUnit() {
	// TODO Auto-generated constructor stub
	initialized = false;
}

LevelShifter::~LevelShifter() {
	// TODO Auto-generated destructor stub
}

void LevelShifter::Initialize(int _numOutput, double _activityRowRead, double _clkFreq){
	if (initialized)
		cout << "[LevelShifter] Warning: Already initialized!" << endl;
	
	numOutput = _numOutput;
	activityRowRead = _activityRowRead;
	clkFreq = _clkFreq;
    
	widthN = MIN_NMOS_SIZE * tech.featureSize;
	widthP = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;
	initialized = true;
}

void LevelShifter::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[LevelShifter] Error: Require initialization first!" << endl;
	} else {
		area = 0;
		height = 0;
		width = 0;
		
		double hlow, hlatch, hhigh, wlow, wlatch, whigh;

		// 1.4 update: needs check - compatibility with FinFET?
		if (param->validated){			
			CalculateGateArea(INV, 1, widthN*15*2, widthP*20*2, tech.featureSize*MAX_TRANSISTOR_HEIGHT*2.5, tech, &hlow, &wlow); 	//width*2, height*2.5
			CalculateGateArea(INV, 1, widthN*32*2, widthP*10*2, tech.featureSize*MAX_TRANSISTOR_HEIGHT*2.5, tech, &hlatch, &wlatch);
			CalculateGateArea(INV, 1, widthN*64*2, widthP*82*2, tech.featureSize*MAX_TRANSISTOR_HEIGHT*2.5, tech, &hhigh, &whigh);
			double hLS = max(max(hlow, hlatch), hhigh);
			double wLS = (wlow + (2*wlatch + whigh)*2.5);		//latch and high_voltage_pull_up are IO transistors, l=270nm
			area = hLS * wLS * numOutput*param->alpha;	//consider wire areas, alpha = 1.44 by default
		}else{
			CalculateGateArea(INV, 1, widthN*15, widthP*20, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hlow, &wlow);
			CalculateGateArea(INV, 1, widthN*32, widthP*10, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hlatch, &wlatch);
			CalculateGateArea(INV, 1, widthN*64, widthP*82, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hhigh, &whigh);
			double hLS = max(max(hlow, hlatch), hhigh); 
			double wLS = wlow + 2*wlatch + whigh;
			area = hLS * wLS * numOutput;
		}
		
		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area / width;
		} else if (_newHeight && _option==NONE) {
			height = _newHeight;
			width = area / height;
		} else {
			cout << "[LevelShifter] Error: No width or height assigned for the LevelShifter circuit" << endl;
			exit(-1);
		}
		
	    // Modify layout
	    newHeight = _newHeight;
	    newWidth = _newWidth;
	    switch (_option) {
		    case MAGIC:
			    MagicLayout();       // if MAGIC, call Magiclayout() in FunctionUnit.cpp
			    break;
		    case OVERRIDE:
			    OverrideLayout();    // if OVERRIDE, call Overridelayout() in FunctionUnit.cpp
			    break;
		    default:    // NONE
			    break;
		}

		// Capacitance

			capMidGateN = CalculateGateCap(widthN*32, tech);;
			CalculateGateCapacitance(INV, 1, widthN*15, widthP*20, hlow, tech, NULL, &capLowDrain);
			CalculateGateCapacitance(INV, 1, widthN*64, widthP*82, hhigh, tech, NULL, &capHighDrain);

	}
}


void LevelShifter::CalculateLatency(double _rampInput, double _capLoad, double _resLoad, double numRead, double numWrite) {	// For simplicity, assume shift register is ideal
	if (!initialized) {
		cout << "[LevelShifter] Error: Require initialization first!" << endl;
	} else {
		rampInput = _rampInput;
		capLoad = _capLoad;
		resLoad = _resLoad;
		double tr, gm, beta, resPullUp;  
		double ramp[10];
		readLatency = 0;
		writeLatency = 0;
		
		// 1st low voltage triggered pull up
		resPullUp = CalculateOnResistance(widthP*20, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capLowDrain + capMidGateN * 2);
		gm = CalculateTransconductance(widthP*20, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatency += horowitz(tr, beta, 1e20, &ramp[0]);
		writeLatency += horowitz(tr, beta, 1e20, &ramp[0]);
		
		// 2ed high voltage pull up
		resPullUp = CalculateOnResistance(widthP*82, PMOS, inputParameter.temperature, tech);
		tr = resPullUp * (capLoad + capHighDrain);
		gm = CalculateTransconductance(widthP*82, PMOS, tech);
		beta = 1 / (resPullUp * gm);
		readLatency += horowitz(tr, beta, 1e20, &ramp[1]);
		writeLatency += horowitz(tr, beta, 1e20, &ramp[1]);
		rampOutput = ramp[1];
		
		readLatency *= numRead;
		writeLatency *= numWrite;
	}
}

void LevelShifter::CalculatePower(double numRead, double numWrite, double activeRowRead) {      
	if (!initialized) {
		cout << "[LevelShifter] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		
		// Read dynamic energy
		// during read, high voltage is not triggered
		readDynamicEnergy += (capLowDrain + capMidGateN * 2) * tech.vdd * tech.vdd * numOutput * activityRowRead;  
		readDynamicEnergy *= numRead;
		
		// Write dynamic energy (2-step write and average case half SET and half RESET)
		// 1T1R
		writeDynamicEnergy += (capLowDrain + capMidGateN * 2) * tech.vdd * tech.vdd;    
		writeDynamicEnergy += (capLoad + capHighDrain) * 2 * cell.writeVoltage * cell.writeVoltage;   	
		writeDynamicEnergy *= numWrite;
	}
}


void LevelShifter::PrintProperty(const char* str) {
	//cout << "LevelShifter Properties:" << endl;
	FunctionUnit::PrintProperty(str);
}

void LevelShifter::UnInitialize(){
	initialized = false;
}

