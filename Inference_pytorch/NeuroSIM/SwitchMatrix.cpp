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
#include "constant.h"
#include "formula.h"
#include "SwitchMatrix.h"

// 1.4 update : include Param.h
#include "Param.h" 

// 1.4 update : include Param.h
extern Param *param;

using namespace std;

SwitchMatrix::SwitchMatrix(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), dff(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void SwitchMatrix::Initialize(int _mode, int _numOutput, double _resTg, bool _neuro, bool _parallelWrite, double _activityRowRead, double _activityColWrite, int _numWriteCellPerOperationMemory, int _numWriteCellPerOperationNeuro, double _numWritePulse, double _clkFreq){
	if (initialized)
		cout << "[SwitchMatrix] Warning: Already initialized!" << endl;
	
	mode = _mode;
	numOutput = _numOutput;
	neuro = _neuro;
	parallelWrite = _parallelWrite;
	activityRowRead = _activityRowRead;
	activityColWrite = _activityColWrite;
	numWriteCellPerOperationMemory = _numWriteCellPerOperationMemory;
	numWriteCellPerOperationNeuro = _numWriteCellPerOperationNeuro;
	numWritePulse = _numWritePulse;
	clkFreq = _clkFreq;
    
	// DFF
	dff.Initialize(numOutput, clkFreq);       // used for scan-in ...
	
	// TG  resTg = cell.resMemCellOn / numLoad * IR_DROP_TOLERANCE;
	resTg = _resTg * param->switchmatrixsizeratio;      // given actual TG resistance
	
	// Why use pre-defined resTg? Becasue we want to define TG resistance according to loading and performance ...
	
	// 1.4 update: for < 14 nm compatibility, changed to on-resistance formula
	widthTgN = CalculateOnResistance_normal(((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech) * tech.featureSize * LINEAR_REGION_RATIO/ (resTg*2);
	// R~(1/W), calculate actual TG width based on feature-sized TG resistance and given actual TG resistance 
	
	// 1.4 update: for < 14 nm compatibility, changed to on-resistance formula 
	widthTgP = CalculateOnResistance_normal(((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, PMOS, inputParameter.temperature, tech) * tech.featureSize * LINEAR_REGION_RATIO/ (resTg*2);
	// assuming resTgN = resTgP, so resTgN = resTgP = 2*resTg (connected in parallel)
	
	// 1.4 update: no Enlarge Size
	// EnlargeSize(&widthTgN, &widthTgP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech);
	

	// 1.4 update: recalculate width for FinFET case

	if (tech.featureSize <= 14*1e-9){
		widthTgN = 2* ceil(widthTgN/tech.featureSize) * tech.featureSize;
		widthTgP = 2* ceil(widthTgP/tech.featureSize) * tech.featureSize;
	} else {
		if (widthTgN < tech.featureSize)
			widthTgN = tech.featureSize;
		if (widthTgP < tech.featureSize)
			widthTgP = tech.featureSize;
	}
	resTg = 1 / (1/CalculateOnResistance_normal(widthTgN, NMOS, inputParameter.temperature, tech)
			+ 1/CalculateOnResistance_normal(widthTgP, PMOS, inputParameter.temperature, tech));
	

	initialized = true;
}

void SwitchMatrix::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[SwitchMatrix] Error: Require initialization first!" << endl;
	} else {
		double hTg, wTg;
		area = 0;
		height = 0;
		width = 0;
		if (mode == ROW_MODE) {	// Connect to rows
			double minCellHeight = MAX_TRANSISTOR_HEIGHT * tech.featureSize;

			// 1.4 update : new cell dimension 

			if (tech.featureSize == 14 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_14nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 10 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_10nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 7 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_7nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 5 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_5nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 3 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_3nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 2 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_2nm /MAX_TRANSISTOR_HEIGHT);
			else if (tech.featureSize == 1 * 1e-9)
			minCellHeight *= ( (double)MAX_TRANSISTOR_HEIGHT_1nm /MAX_TRANSISTOR_HEIGHT);
			else
			minCellHeight *= 1;

			if (_newHeight && _option==NONE) {
				if (_newHeight < minCellHeight) {
					cout << "[SwitchMatrix] Error: pass gate height is even larger than the array height" << endl;
				}
				int numTgPairPerCol = (int)(_newHeight / minCellHeight);// Get max # Tg pair per column (this is not the final # Tg pair per column because the last column may have less # Tg)
				numColTgPair = (int)ceil((double)numOutput / numTgPairPerCol);	// Get min # columns based on this max # Tg pair per column
				numTgPairPerCol = (int)ceil((double)numOutput / numColTgPair);	// Get # Tg pair per column based on this min # columns
				TgHeight = _newHeight / numTgPairPerCol;
				CalculateGateArea(INV, 1, widthTgN, widthTgP, TgHeight, tech, &hTg, &wTg);
				
				// DFF
				dff.CalculateArea(_newHeight, NULL, NONE);
				
				height = _newHeight;
				width = (wTg * 2) * numColTgPair + dff.width;

			} else {
				CalculateGateArea(INV, 1, widthTgN, widthTgP, minCellHeight, tech, &hTg, &wTg); // Pass gate with folding
				height = hTg * numOutput;
				dff.CalculateArea(height, NULL, NONE);	// Need to give the height information, otherwise by default the area calculation of DFF is in column mode
				width = (wTg * 2) + dff.width;
			}
		} else {	// Connect to columns
			if (_newWidth && _option==NONE) {
				numRowTgPair = 1;
				double minCellWidth = 2 * (POLY_WIDTH + MIN_GAP_BET_GATE_POLY) * tech.featureSize; // min standard cell width for 1 Tg

				// 1.4 update : new cell dimension 

				if (tech.featureSize == 14 * 1e-9)
				minCellWidth  *= ( (double)CPP_14nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 10 * 1e-9)
				minCellWidth  *= ( (double)CPP_10nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 7 * 1e-9)
				minCellWidth  *= ( (double)CPP_7nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 5 * 1e-9)
				minCellWidth  *= ( (double)CPP_5nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 3 * 1e-9)
				minCellWidth  *= ( (double)CPP_3nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 2 * 1e-9)
				minCellWidth  *= ( (double)CPP_2nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else if (tech.featureSize == 1 * 1e-9)
				minCellWidth  *= ( (double)CPP_1nm/(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
				else
				minCellWidth  *= 1;
				
				if (minCellWidth > _newWidth) {
					cout << "[SwitchMatrix] Error: pass gate width is even larger than the array width" << endl;
				}

				int numTgPairPerRow = (int)(_newWidth / (minCellWidth*2));    // Get max # Tg pair per row (this is not the final # Tg pair per row because the last row may have less # Tg)
				numRowTgPair = (int)ceil((double)numOutput / numTgPairPerRow); // Get min # rows based on this max # Tg pair per row
				numTgPairPerRow = (int)ceil((double)numOutput / numRowTgPair);     // Get # Tg pair per row based on this min # rows
				TgWidth = _newWidth / numTgPairPerRow / 2;	// division of 2 because there are 2 Tg in one pair
				int numFold = (int)(TgWidth / (0.5*minCellWidth)) - 1;  // get the max number of folding

				// widthTgN, widthTgP and numFold can determine the height and width of each pass gate
				CalculatePassGateArea(widthTgN, widthTgP, tech, numFold, &hTg, &wTg);
				
				// DFF
				dff.CalculateArea(NULL, _newWidth, NONE);

				width = _newWidth;
				height = hTg * numRowTgPair + dff.height;
			} else {
				// Default (pass gate with folding=1)
				CalculatePassGateArea(widthTgN, widthTgP, tech, 1, &hTg, &wTg);
				width = wTg * 2 * numOutput;
				dff.CalculateArea(NULL, width, NONE);
				height = hTg + dff.height;
			}
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
		// TG
		capTgGateN = CalculateGateCap(widthTgN, tech);
		capTgGateP = CalculateGateCap(widthTgP, tech);
		CalculateGateCapacitance(INV, 1, widthTgN, widthTgP, hTg, tech, NULL, &capTgDrain);
		
	}
}

void SwitchMatrix::CalculateLatency(double _rampInput, double _capLoad, double _resLoad, double numRead, double numWrite) {	// For simplicity, assume shift register is ideal
	if (!initialized) {
		cout << "[SwitchMatrix] Error: Require initialization first!" << endl;
	} else {
		rampInput = _rampInput;
		capLoad = _capLoad;
		resLoad = _resLoad;
		double capOutput;
		double tr;  /* time constant */
		readLatency = 0;

		// DFF
		dff.CalculateLatency(1e20, numRead);

		// TG
		// 1.4 update: TG outputcap modified capTgDrain * 3; -> capTgDrain * 2;
		// needs check, due to the GS/GD parasitic cap

		capOutput = capTgDrain * 2 + capTgGateN*0.5 + capTgGateP*0.5 ;

		// 1.4 update: resTg, buffer latency model for switch matrix (propagation delay for distributed RC)
		// eq resistance (rn//rp) stays the same regardless of the biasing in transmission gate. So we take ron,n // ron,p which is the resistance at the onset
		
		double sectionnum = param->numColSubArray/(param->buffernumber+1);

		if (param->buffernumber ==0){
			tr =  resTg * (capOutput) * 0.69 
			+ param->unitcap * sectionnum * (0.69*resTg + 0.38*param->unitres*sectionnum);						
		} else {
			tr =  resTg * (capOutput) * 0.69 
			+ param->unitcap *  sectionnum * (0.69*resTg + 0.38*param->unitres*sectionnum)
			+ (param->unitcap * sectionnum * 0.69 + 0.69 * resTg) * param-> drivecapin;		
		}

		readLatency += tr;
		// readLatency += horowitz(tr, 0, rampInput, &rampOutput);	// we do not use horitz model here 
		readLatency *= numRead;
		// readLatency += dff.readLatency;

		writeLatency = horowitz(tr, 0, rampInput, &rampOutput);
		writeLatency *= numWrite;
		writeLatency += dff.readLatency;	// Use DFF read latency here because no write in the DFF module
	}
}

void SwitchMatrix::CalculatePower(double numRead, double numWrite, double activityRowRead, double activityColWrite) {
	if (!initialized) {
		cout << "[SwitchMatrix] Error: Require initialization first!" << endl;
	} else {
		
		leakage = 0;
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		
		// DFF
		dff.CalculatePower(numRead, numOutput, false);	// Use numOutput since every DFF will pass signal (either 0 or 1)

		// Leakage power
		leakage += dff.leakage;	// Only DFF has leakage

		// Read dynamic energy
		if (!neuro) {    // Memory mode
			readDynamicEnergy += (capTgDrain * 3) * cell.readVoltage * cell.readVoltage;
			readDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd;
		} else {	// Neuro mode
			if (mode == ROW_MODE) {
				readDynamicEnergy += (capTgDrain * 3) * cell.readVoltage * cell.readVoltage * numOutput * activityRowRead;
				readDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput * activityRowRead;
			} // No read energy in COL_MODE
		}
		readDynamicEnergy *= numRead;
		readDynamicEnergy += dff.readDynamicEnergy;
		
		// Write dynamic energy (2-step write and average case half SET and half RESET)
		if (cell.accessType == CMOS_access) {	// 1T1R

			if (mode == ROW_MODE) {	// Connects to rows
				if (!neuro) {    // Memory mode
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * 2;	// Selected row in SET, *2 means switching from one selected row to another
				} else {    // Neuro mode
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * 2;	// Selected row in LTP, *2 means switching from one selected row to another
				}
			} else {	// Connects to columns
				if (!neuro) {    // Memory mode
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * (numOutput - MIN(numWriteCellPerOperationMemory, numOutput)/2);   // Unselected columns in SET
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationMemory, numOutput)/2;   // Selected columns in RESET
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
				} else {    // Neuro mode
					// LTP
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * (numOutput - MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite)/2);   // Unselected columns 
					// LTD
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns
					
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
				}
			}

		} else {	// Cross-point
			
			if (mode == ROW_MODE) { // Connects to rows
				if (!neuro) {    // Memory mode
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage;   // Selected row in SET
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage/2 * cell.writeVoltage/2 * (numOutput-1);  // Unselected rows in SET and RESET
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
				} else {    // Neuro mode
					if (parallelWrite) {	// Z: one-time charge to Vw, assuming the Z pulse will still go to Vw at the R<0 phase
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numOutput;
						writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
					} else {
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage;   // Selected row in LTP
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage/2 * cell.writeVoltage/2 * (numOutput-1);   // Unselected rows in LTP and LTD
						writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
					}
				}
			} else {    // Connects to columns
				if (!neuro) {    // Memory mode
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationMemory, numOutput) / 2;   // Selected columns in RESET
					writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage/2 * cell.writeVoltage/2 * numOutput;   // Total unselected columns in SET and RESET within the 2-step write
					writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
				} else {    // Neuro mode
					if (parallelWrite) {
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * numOutput * activityColWrite;
						writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
					} else {
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns in LTP
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage * cell.writeVoltage * numWritePulse * MIN(numWriteCellPerOperationNeuro, numOutput*activityColWrite) / 2;   // Selected columns in LTD
						writeDynamicEnergy += (capTgDrain * 3) * cell.writeVoltage/2 * cell.writeVoltage/2 * numOutput;   // Total unselected columns in LTP and LTD within the 2-step write
						writeDynamicEnergy += (capTgGateN + capTgGateP) * tech.vdd * tech.vdd * numOutput;
					}
				}
			}

		}
		writeDynamicEnergy *= numWrite;
		writeDynamicEnergy += dff.readDynamicEnergy;	// Use DFF read energy here because no write in the DFF module
	}
}

void SwitchMatrix::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

void SwitchMatrix::SaveOutput(const char* str) {
	FunctionUnit::SaveOutput(str);
}

