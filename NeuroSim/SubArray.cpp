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
#include "SubArray.h"

using namespace std;

SubArray::SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell),
						wlDecoder(_inputParameter, _tech, _cell),
						wlDecoderOutput(_inputParameter, _tech, _cell),
						mux(_inputParameter, _tech, _cell),
						muxDecoder(_inputParameter, _tech, _cell),
						slSwitchMatrix(_inputParameter, _tech, _cell),
						blSwitchMatrix(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						readCircuit(_inputParameter, _tech, _cell),
						shiftAdd(_inputParameter, _tech, _cell) {
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
}

void SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	if (initialized)
		cout << "[Subarray] Warning: Already initialized!" << endl;  //avioding initialize twice
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		// XXX: To be released
	} else if (cell.memCellType == Type::RRAM) {  //if array is RRAM
		double cellHeight = cell.heightInFeatureSize;  //set RRAM cell height
		double cellWidth = cell.widthInFeatureSize;  //set RRAM cell width
		if (cell.accessType == CMOS_access) {  // 1T1R
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth, MIN_CELL_WIDTH*2) * tech.featureSize;	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * tech.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight, MIN_CELL_HEIGHT) * tech.featureSize;
			} else {
				lengthCol = (double)numRow * cellHeight * tech.featureSize;
			}
		} else {	// Cross-point, if enter anything else except 'CMOS_access'
			if (relaxArrayCellWidth) {
				lengthRow = (double)numCol * MAX(cellWidth*cell.featureSize, MIN_CELL_WIDTH*2*tech.featureSize);	// Width*2 because generally switch matrix has 2 pass gates per column, even the SL/BL driver has 2 pass gates per column in traditional 1T1R memory
			} else {
				lengthRow = (double)numCol * cellWidth * cell.featureSize;
			}
			if (relaxArrayCellHeight) {
				lengthCol = (double)numRow * MAX(cellHeight*cell.featureSize, MIN_CELL_HEIGHT*tech.featureSize);
			} else {  
				lengthCol = (double)numRow * cellHeight * cell.featureSize;
			}
		}
	}      //finish setting array size
	
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	

	//start to initializing the subarray modules
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		// XXX: To be released
    } else if (cell.memCellType == Type::RRAM) {
		if (cell.accessType == CMOS_access) {	// 1T1R
			if (!cell.resCellAccess)    // If not defined
				cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
			cell.widthAccessCMOS = CalculateOnResistance(tech.featureSize, NMOS, inputParameter.temperature, tech) / cell.resCellAccess;   //get access CMOS width
			if (cell.widthAccessCMOS > cell.widthInFeatureSize) {	// Place transistor vertically
				printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
				exit(-1);
			}

			cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;       //calculate single memory cell resistance_ON
			cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF
			cell.resMemCellAvg = cell.resCellAccess + cell.resistanceAvg;      //calculate single memory cell resistance_AVG

			capRow2 += CalculateGateCap(cell.widthAccessCMOS * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
			capCol += CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit

			if (digitalModeNeuro) {  //if digital mode pseudo-1T1R
				// XXX: To be released
			} else {  //analog mode pesudo-1T1R
				wlDecoderOutput.Initialize(numRow, false, true);     //WL decoder follower    
				wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false);
				
				double resTg = cell.resMemCellOn * IR_DROP_TOLERANCE;
				slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, activityRowRead, activityColWrite, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);     //SL use switch matrix
				
				resTg = cell.resMemCellOn / numCol * IR_DROP_TOLERANCE;
				blSwitchMatrix.Initialize(ROW_MODE, numRow, resTg, activityRowRead, activityColWrite, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);    //BL use switch matrix
				
				int numInput = (int)ceil((double)numCol/numColMuxed);     //input number of mux (num of column/ num of column that share one SA)
				resTg = cell.resMemCellOn / numRow * IR_DROP_TOLERANCE;
				mux.Initialize(numInput, numColMuxed, resTg, false);

				if (numColMuxed > 1) {    //if more than one column share one SA
					muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true);
				}
				readCircuit.Initialize(readCircuitMode, (int)ceil((double)numCol/numColMuxed), maxNumIntBit, spikingMode, clkFreq);
				shiftAdd.Initialize((int)ceil((double)numCol/numColMuxed), readCircuit.maxNumIntBit, clkFreq, spikingMode, numReadPulse);
			}
		} else {	// Cross-point
			
			// The nonlinearity is from the selector, assuming RRAM itself is linear
			if (cell.nonlinearIV) {   //introduce nonlinearity to the RRAM resistance
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOffAtHalfVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage/2);
				cell.resMemCellOnAtVw = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellOffAtVw = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, cell.readVoltage, cell.writeVoltage);
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = (cell.resMemCellOnAtHalfVw + cell.resMemCellOffAtHalfVw) / 2;
				cell.resMemCellAvgAtVw = (cell.resMemCellOnAtVw + cell.resMemCellOffAtVw) / 2;
			} else {  //simply assume RRAM resistance is linear
				cell.resMemCellOn = cell.resistanceOn;
				cell.resMemCellOff = cell.resistanceOff;
				cell.resMemCellOnAtHalfVw = cell.resistanceOn;
				cell.resMemCellOffAtHalfVw = cell.resistanceOff;
				cell.resMemCellOnAtVw = cell.resistanceOn;
				cell.resMemCellOffAtVw = cell.resistanceOff;
				cell.resMemCellAvg = cell.resistanceAvg;
				cell.resMemCellAvgAtHalfVw = cell.resistanceAvg;
				cell.resMemCellAvgAtVw = cell.resistanceAvg;
			}

			if (digitalModeNeuro) {
				// XXX: To be released
			} else {  //analog mode cross-point
				double resTg = cell.resMemCellOnAtVw / numCol * IR_DROP_TOLERANCE;
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg, activityRowRead, activityColWrite, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);
				
				resTg = cell.resMemCellOnAtVw / numRow * IR_DROP_TOLERANCE;
				blSwitchMatrix.Initialize(COL_MODE, numCol, resTg, activityRowRead, activityColWrite, numWriteCellPerOperationNeuro, numWritePulse, clkFreq);

				int numInput = (int)ceil((double)numCol/numColMuxed);
				resTg = cell.resMemCellOnAtVw / numRow * IR_DROP_TOLERANCE;
				mux.Initialize(numInput, numColMuxed, resTg, false);

				if (numColMuxed > 1) {
					muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true);
				}
				
				readCircuit.Initialize(readCircuitMode, (int)ceil((double)numCol/numColMuxed), maxNumIntBit, spikingMode, clkFreq);
				shiftAdd.Initialize((int)ceil((double)numCol/numColMuxed), readCircuit.maxNumIntBit, clkFreq, spikingMode, numReadPulse);
			}
		}
	}

	initialized = true;  //finish initialization
}



void SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		if (cell.memCellType == Type::SRAM) {       
			// XXX: To be released
	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {	// 1T1R
				
				// Array only
				heightArray = lengthCol;
				widthArray = lengthRow;
				areaArray = heightArray * widthArray;
				
				if (digitalModeNeuro) {
					// XXX: To be released
				} else {  //analog mode 1T1R RRAM
					wlDecoder.CalculateArea(heightArray, NULL, NONE);
					wlDecoderOutput.CalculateArea(heightArray, NULL, NONE);
					slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
					blSwitchMatrix.CalculateArea(heightArray, NULL, NONE);

					// Get Mux height, compare it with Mux decoder height, and select whichever is larger for Mux
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);

					readCircuit.CalculateUnitArea();
					readCircuit.CalculateArea(mux.width);
					if (shiftAddEnable) {
						shiftAdd.CalculateArea(NULL, mux.width, NONE);
					}

					height = slSwitchMatrix.height + heightArray + mux.height + readCircuit.height + shiftAdd.height;
					width = MAX(wlDecoder.width+wlDecoderOutput.width, muxDecoder.width) + widthArray + blSwitchMatrix.width;
					area = height * width;
					usedArea = areaArray + wlDecoder.area + wlDecoderOutput.area + slSwitchMatrix.area + blSwitchMatrix.area + mux.area + readCircuit.area + muxDecoder.area + shiftAdd.area;
					emptyArea = area - usedArea;
				}

			} else {        // Cross-point
				
				// Array only
				heightArray = lengthCol;
				widthArray = lengthRow;
				areaArray = heightArray * widthArray;
				
				if (digitalModeNeuro) {
					// XXX: To be released
				} else {  //analog cross-point
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
					blSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
					// Get Mux height, compare it with Mux decoder height, and select whichever is larger for Mux
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
					
					readCircuit.CalculateUnitArea();
					readCircuit.CalculateArea(mux.width);
					if (shiftAddEnable) {
						shiftAdd.CalculateArea(NULL, mux.width, NONE);
					}
					height = blSwitchMatrix.height + heightArray + mux.height + readCircuit.height + shiftAdd.height;
					width = MAX(wlSwitchMatrix.width, muxDecoder.width) + widthArray;
					area = height * width;
					usedArea = areaArray + wlSwitchMatrix.area + blSwitchMatrix.area + mux.area + readCircuit.area + muxDecoder.area + shiftAdd.area;
					emptyArea = area - usedArea;
				}
			}
		}
	}
}

void SubArray::CalculateLatency(double _rampInput) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		if (cell.memCellType == Type::SRAM) {
			// XXX: To be released
	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {   // 1T1R
				
				if (digitalModeNeuro) {
					// XXX: To be released
				} else {
					int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
					wlDecoder.CalculateLatency(1e20, wlDecoderOutput.capNorInput, NULL, 1, numRow*activityRowWrite);
					wlDecoderOutput.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, 1, numRow*activityRowWrite);
					slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 1, 1);
					blSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numReadPulse, maxNumWritePulse*2*numWriteOperationPerRow*numRow*activityRowWrite);	// *2 means 2-step write

					// Calculate column latency
					double colRamp = 0;
					double tau = resCol * capCol / 2 * (cell.resMemCellOff + resCol / 3) / (cell.resMemCellOff + resCol);
					colDelay = horowitz(tau, 0, blSwitchMatrix.rampOutput, &colRamp);	// Just to generate colRamp

					mux.CalculateLatency(colRamp, 0, 1);
					int numInput = (int)ceil((double)numCol/numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numInput, mux.capTgGateP*numInput, 1, 1);
					//if (multifunctional)
					//	deMux.CalculateLatency(1e20, 1);
					
					// Read
					if (readCircuit.mode == CMOS) {
						double Cin = capCol + mux.capTgDrain * (2 + numColMuxed - 1) + readCircuit.capTgDrain + readCircuit.capPmosGate;
						double Imax = numRow * cell.readVoltage / cell.resMemCellOn;
						cell.readPulseWidth = Cin * readCircuit.voltageIntThreshold / Imax * readCircuit.maxNumIntPerCycle;
					} else {	// mode==OSCILLATION
						double Cin = capCol + mux.capTgDrain * (2 + numColMuxed - 1) + readCircuit.capInvInput;
						double Rmin = cell.resMemCellOn / numRow;
						double Rp = 1 / (1/Rmin + 1/readCircuit.R_OSC_OFF);
						double t_rise = -Rp * Cin * log((readCircuit.Vth - readCircuit.Vrow*Rp/Rmin) / (readCircuit.Vhold - readCircuit.Vrow*Rp/Rmin));
						cell.readPulseWidth = t_rise * readCircuit.maxNumIntPerCycle;
					}
					readCircuit.CalculateLatency(numColMuxed*numReadPulse);
					if (shiftAddEnable) {
						shiftAdd.CalculateLatency(numReadPulse);
					}
					
					readLatency += wlDecoderOutput.readLatency;
					readLatency += blSwitchMatrix.readLatency;
					readLatency += readCircuit.readLatency;
					readLatency += shiftAdd.readLatency;
					
					// Write
					writeLatency += wlDecoder.writeLatency;
					writeLatency += wlDecoderOutput.writeLatency;
					writeLatency += blSwitchMatrix.writeLatency;
				}

			} else {	// Cross-point
				if (digitalModeNeuro) {
					// XXX: To be released
				} else {
					int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numReadPulse, maxNumWritePulse*2*numWriteOperationPerRow*numRow*activityRowWrite);	// *2 means 2-step write
					blSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 1, 1);

					// Calculate column latency
					double colRamp = 0;
					double tau = resCol * capCol / 2 * (cell.resMemCellOff + resCol / 3) / (cell.resMemCellOff + resCol);
					colDelay = horowitz(tau, 0, wlSwitchMatrix.rampOutput, &colRamp);	// Just to generate colRamp

					mux.CalculateLatency(colRamp, 0, 1);
					int numInput = (int)ceil((double)numCol/numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*numInput, mux.capTgGateP*numInput, 1, 1);

					// Read
					if (readCircuit.mode == CMOS) {
						double Cin = capCol + mux.capTgDrain * (2 + numColMuxed - 1) + readCircuit.capTgDrain + readCircuit.capPmosGate;
						double Imax = numRow * cell.readVoltage / cell.resMemCellOn;
						cell.readPulseWidth = Cin * readCircuit.voltageIntThreshold / Imax * readCircuit.maxNumIntPerCycle;
					} else {    // mode==OSCILLATION
						double Cin = capCol + mux.capTgDrain * (2 + numColMuxed - 1) + readCircuit.capInvInput;
						double Rmin = cell.resMemCellOn / numRow;
						double Rp = 1 / (1/Rmin + 1/readCircuit.R_OSC_OFF);
						double t_rise = -Rp * Cin * log((readCircuit.Vth - readCircuit.Vrow*Rp/Rmin) / (readCircuit.Vhold - readCircuit.Vrow*Rp/Rmin));
						cell.readPulseWidth = t_rise * readCircuit.maxNumIntPerCycle;
					}
					readCircuit.CalculateLatency(numColMuxed*numReadPulse);
					if (shiftAddEnable) {
						shiftAdd.CalculateLatency(numReadPulse);
					}
					
					readLatency += wlSwitchMatrix.readLatency;
					readLatency += readCircuit.readLatency;
					readLatency += shiftAdd.readLatency;
					
					// Write
					writeLatency += wlSwitchMatrix.writeLatency;
				}
			}
		}
	}
}

void SubArray::CalculatePower() {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		if (cell.memCellType == Type::SRAM) {
			// XXX: To be released
	    } else if (cell.memCellType == Type::RRAM) {
			if (cell.accessType == CMOS_access) {   // 1T1R
							
				if (digitalModeNeuro) {
					// XXX: To be released
				} else {
					double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
					if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
						numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
					else
						numWriteOperationPerRow = 1;
					wlDecoder.CalculatePower(numReadPulse, numRow*activityRowWrite);
					wlDecoderOutput.CalculatePower(numReadPulse, numRow*activityRowWrite);
					blSwitchMatrix.CalculatePower(numReadPulse, numRow*activityRowWrite);
					slSwitchMatrix.CalculatePower(1, numWriteOperationPerRow*numRow*activityRowWrite);
					mux.CalculatePower(numColMuxed*numReadPulse);
					muxDecoder.CalculatePower(numColMuxed*numReadPulse, 1);
					readCircuit.CalculatePower(numColMuxed*numReadPulse);
					if (shiftAddEnable) {
						shiftAdd.CalculatePower(numReadPulse);
					}
					
					// Array
					readDynamicEnergyArray += capRow1 * readCircuit.Vrow * readCircuit.Vrow * numRow * activityRowRead;   // Selected BLs
					readDynamicEnergyArray += capCol * readCircuit.Vcol * readCircuit.Vcol * numCol;	// Read all columns in total
					readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow; // All WLs open
					readDynamicEnergyArray += cell.readVoltage * cell.readVoltage / cell.resMemCellAvg * cell.readPulseWidth * numRow * activityRowRead * numCol; // Unselected SLs are floating (assume the read integration threshold is small enough)
					readDynamicEnergyArray *= numReadPulse;
					// Use average case in write energy calculation: half LTP and half LTD with average resistance
					// LTP
					writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol * activityColWrite) / 2 * numWritePulse;	// Selected SLs
					writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * (numCol - MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite)/2);	// Unselected SLs
					writeDynamicEnergyArray += capRow1 * cell.writeVoltage * cell.writeVoltage;	// Selected BL
					writeDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd;	// Selected WL
					writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvg * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth;	// LTP
					// LTD
					writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol * activityColWrite) / 2 * numWritePulse;    // Selected SLs
					writeDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
					writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvg * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth;	// LTD
					writeDynamicEnergyArray *= numWriteOperationPerRow * numRow * activityRowWrite;

					// Read
					readDynamicEnergy += wlDecoder.readDynamicEnergy;
					readDynamicEnergy += wlDecoderOutput.readDynamicEnergy;
					readDynamicEnergy += blSwitchMatrix.readDynamicEnergy;
					readDynamicEnergy += readDynamicEnergyArray;
					readDynamicEnergy += mux.readDynamicEnergy;
					readDynamicEnergy += muxDecoder.readDynamicEnergy;
					readDynamicEnergy += readCircuit.readDynamicEnergy;
					readDynamicEnergy += shiftAdd.readDynamicEnergy;
					
					// Write
					writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
					writeDynamicEnergy += wlDecoderOutput.writeDynamicEnergy;
					writeDynamicEnergy += blSwitchMatrix.writeDynamicEnergy;
					writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
					writeDynamicEnergy += writeDynamicEnergyArray;
				}
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += wlDecoderOutput.leakage;
				leakage += blSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += readCircuit.leakage;
				leakage += shiftAdd.leakage;

			} else {	// Cross-point

				if (digitalModeNeuro) {
					// XXX: To be released
				} else {

					double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
					if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
						numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
					else
						numWriteOperationPerRow = 1;
					wlSwitchMatrix.CalculatePower(numReadPulse, numRow*activityRowWrite);
					blSwitchMatrix.CalculatePower(1, numWriteOperationPerRow*numRow*activityRowWrite);
					mux.CalculatePower(numColMuxed*numReadPulse);
					muxDecoder.CalculatePower(numColMuxed*numReadPulse, 1);
					readCircuit.CalculatePower(numColMuxed*numReadPulse);
					if (shiftAddEnable) {
						shiftAdd.CalculatePower(numReadPulse);
					}
					
					// Array
					readDynamicEnergyArray += capRow1 * readCircuit.Vrow * readCircuit.Vrow * numRow * activityRowRead;   // Selected BLs
					readDynamicEnergyArray += capCol * readCircuit.Vcol * readCircuit.Vcol * numCol;    // Read all columns in total
					readDynamicEnergyArray += cell.readVoltage * cell.readVoltage / cell.resMemCellAvg * cell.readPulseWidth * numRow * activityRowRead * numCol; // Unselected SLs are floating (assume the read integration threshold is small enough)
					readDynamicEnergyArray *= numReadPulse;
					// Use average case in write energy calculation: half LTP and half LTD with average resistance
					double totalWriteTime = cell.writePulseWidth * maxNumWritePulse;
					// LTP
					writeDynamicEnergyArray += capRow1 * cell.writeVoltage * cell.writeVoltage;	// Selected WL
					writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse;	// Selected BLs
					writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvgAtVw * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth;	// LTP
					writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numCol - MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite)/2) * totalWriteTime;    // Half-selected cells on the row
					writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numRow-1) * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * totalWriteTime;   // Half-selected cells on the selected columns
					// LTD
					writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse;  // Selected BLs
					writeDynamicEnergyArray += cell.writeVoltage * cell.writeVoltage / cell.resMemCellAvgAtVw * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * numWritePulse * cell.writePulseWidth; // LTD
					writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numCol - MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite)/2) * totalWriteTime;    // Half-selected cells on the row
					writeDynamicEnergyArray += cell.writeVoltage/2 * cell.writeVoltage/2 / cell.resMemCellAvgAtHalfVw * (numRow-1) * MIN(numWriteCellPerOperationNeuro, numCol*activityColWrite) / 2 * totalWriteTime;   // Half-selected cells on the selected columns
					// Both SET and RESET
					writeDynamicEnergyArray += capCol * cell.writeVoltage/2 * cell.writeVoltage/2 * numCol; // Unselected BLs (every BL has one time to charge to V/2 within the 2-step write)
					writeDynamicEnergyArray += capRow1 * cell.writeVoltage/2 * cell.writeVoltage/2 * (numRow-1);  // Unselected WLs
					
					writeDynamicEnergyArray *= numWriteOperationPerRow * numRow * activityRowWrite;
					
					// Read
					readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
					readDynamicEnergy += readDynamicEnergyArray;
					readDynamicEnergy += mux.readDynamicEnergy;
					readDynamicEnergy += muxDecoder.readDynamicEnergy;
					readDynamicEnergy += readCircuit.readDynamicEnergy;
					readDynamicEnergy += shiftAdd.readDynamicEnergy;

					// Write
					writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
					writeDynamicEnergy += blSwitchMatrix.writeDynamicEnergy;
					writeDynamicEnergy += writeDynamicEnergyArray;
				}
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += wlSwitchMatrix.leakage;
				leakage += blSwitchMatrix.leakage;
				leakage += mux.leakage;
				leakage += muxDecoder.leakage;
				leakage += readCircuit.leakage;
				leakage += shiftAdd.leakage;

			}
		}

		if (!readLatency) {
			cout << "[SubArray] Error: Need to calculate read latency first" << endl;
		} else {
			readPower = readDynamicEnergy/readLatency + leakage;
		}
		if (!writeLatency) {
			cout << "[SubArray] Error: Need to calculate write latency first" << endl;
		} else {
			writePower = writeDynamicEnergy/writeLatency + leakage;
		}

	}
}

void SubArray::PrintProperty() {
	cout << endl << endl;
	cout << "Array:" << endl;
	cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
	if (cell.memCellType == Type::SRAM) {
		// XXX: To be released
	} else if (cell.memCellType == Type::RRAM) {
		if (cell.accessType == CMOS_access) {   // 1T1R
			if (digitalModeNeuro) {
				// XXX: To be released
			} else {
				wlDecoderOutput.PrintProperty("wlDecoderOutput");
				wlDecoder.PrintProperty("wlDecoder");
				slSwitchMatrix.PrintProperty("slSwitchMatrix");
				blSwitchMatrix.PrintProperty("blSwitchMatrix");
				mux.PrintProperty("mux");
				muxDecoder.PrintProperty("muxDecoder");
				readCircuit.PrintProperty("readCircuit");
				if (shiftAddEnable) {
					shiftAdd.PrintProperty("shiftAdd");
				}
			}
		} else {	// Crosspoint
			if (digitalModeNeuro) {
				// XXX: To be released
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
				blSwitchMatrix.PrintProperty("blSwitchMatrix");
				mux.PrintProperty("mux");
				muxDecoder.PrintProperty("muxDecoder");
				readCircuit.PrintProperty("readCircuit");
				if (shiftAddEnable) {
					shiftAdd.PrintProperty("shiftAdd");
				}
			}
		}
	}
	FunctionUnit::PrintProperty("SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

