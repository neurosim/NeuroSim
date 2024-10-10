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
#include "NeuroSim.h"
#include "NeuroSim/constant.h"
#include "NeuroSim/formula.h"
#include "Cell.h"
#include "Param.h"

using namespace std;

extern Param *param;

void NeuroSimSubArrayInitialize(SubArray *& subArray, Array *array, InputParameter& inputParameter, Technology& tech, MemCell& cell) {
	
	/* Create SubArray object and link the required global objects (not initialization) */
	subArray = new SubArray(inputParameter, tech, cell);

	inputParameter.deviceRoadmap = HP;	// HP: high performance, LSTP: low power
	inputParameter.temperature = 301;	// Temperature (K)
	inputParameter.processNode = param->processNode;	// Technology node
	
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap);
	
	subArray->activityRowWrite = (double)1/2;	// Dynamic parameter (to be determined)
	subArray->activityColWrite = (double)1/2;	// Dynamic parameter (to be determined)
	subArray->activityRowRead = (double)1/2;	// Dynamic parameter (to be determined)
	subArray->spikingMode = NONSPIKING; // NONSPIKING: input data using pulses in binary representation
										// SPIKING: input data using # of pulses
	if (subArray->spikingMode == SPIKING) {
		subArray->numReadPulse = pow(2, param->numBitInput);
	} else {
		subArray->numReadPulse = param->numBitInput;
	}
	subArray->numWritePulse = 8;	// Dynamic parameter (to be determined)
	subArray->digitalModeNeuro = 0;	// Use analog eNVM
	subArray->clkFreq = param->clkFreq;		// Clock frequency
	subArray->numCellPerSynapse = array->numCellPerSynapse;	// # of cells per synapse
	subArray->numColMuxed = param->numColMuxed;    // How many columns share 1 read circuit (for analog RRAM) or 1 S/A (for digital RRAM)
	subArray->numWriteColMuxed = param->numWriteColMuxed;	// Time multiplexing during write operation
	if (subArray->spikingMode == NONSPIKING && subArray->numReadPulse > 1) {
		subArray->shiftAddEnable = true;	// Need to shift & add the partial weighted sum
	} else {
		subArray->shiftAddEnable = false;
	}
	subArray->relaxArrayCellHeight = param->relaxArrayCellHeight;
	subArray->relaxArrayCellWidth = param->relaxArrayCellWidth;

	cell.heightInFeatureSize = (array->cell[0][0])->heightInFeatureSize;	// Cell height in feature size
	cell.widthInFeatureSize = (array->cell[0][0])->widthInFeatureSize;		// Cell width in feature size

	if (SRAM *temp = dynamic_cast<SRAM*>(array->cell[0][0])) {	// SRAM
		// XXX: To be released
	} else {	// eNVM
		cell.memCellType = Type::RRAM;
		subArray->readCircuitMode  = CMOS;	// CMOS implementation for integrate-and-fire neuron
		subArray->maxNumIntBit = param->numBitPartialSum;	// Max # bits for the integrate-and-fire neuron
		
		if (subArray->digitalModeNeuro) {
			// XXX: To be released
		} else {
			int maxNumLevelLTP = static_cast<AnalogNVM*>(array->cell[0][0])->maxNumLevelLTP;
			int maxNumLevelLTD = static_cast<AnalogNVM*>(array->cell[0][0])->maxNumLevelLTD;
			subArray->maxNumWritePulse = (maxNumLevelLTP > maxNumLevelLTD)? maxNumLevelLTP : maxNumLevelLTD;
		}
		
		cell.accessType = (static_cast<eNVM*>(array->cell[0][0])->cmosAccess)? CMOS_access : none_access;	// CMOS_access: 1T1R (pseudo-crossbar), none_access: crossbar

		cell.resistanceOn = 1/static_cast<eNVM*>(array->cell[0][0])->maxConductance;	// Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
		cell.resistanceOff = 1/static_cast<eNVM*>(array->cell[0][0])->minConductance;	// Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
		cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;	// Average resistance (used for energy estimation)
		cell.resCellAccess = static_cast<eNVM*>(array->cell[0][0])->resistanceAccess;   // Access transistor resistance
		cell.readVoltage = static_cast<eNVM*>(array->cell[0][0])->readVoltage;	// On-chip read voltage for memory cell
		double writeVoltageLTP = static_cast<eNVM*>(array->cell[0][0])->writeVoltageLTP;
		double writeVoltageLTD = static_cast<eNVM*>(array->cell[0][0])->writeVoltageLTD;
		cell.writeVoltage = sqrt(writeVoltageLTP * writeVoltageLTP + writeVoltageLTD * writeVoltageLTD);	// Use an average value of write voltage for NeuroSim
		cell.readPulseWidth = static_cast<eNVM*>(array->cell[0][0])->readPulseWidth;
		double writePulseWidthLTP = static_cast<eNVM*>(array->cell[0][0])->writePulseWidthLTP;
		double writePulseWidthLTD = static_cast<eNVM*>(array->cell[0][0])->writePulseWidthLTD;
		cell.writePulseWidth = (writePulseWidthLTP + writePulseWidthLTD) / 2;
		cell.nonlinearIV = static_cast<eNVM*>(array->cell[0][0])->nonlinearIV; // This option is to consider I-V nonlinearity in cross-point array or not
		cell.nonlinearity = (cell.nonlinearIV)? 10 : 2;	// This is the nonlinearity for the current ratio at Vw and Vw/2
		if (cell.nonlinearIV) {
			double Vr_exp = 1;  // XXX: Modify this to Vr in the reported measurement data (can be different than cell.readVoltage)
			// Calculation of resistance at on-chip Vr
			cell.resistanceOn = NonlinearResistance(cell.resistanceOn, cell.nonlinearity, cell.writeVoltage, Vr_exp, cell.readVoltage);
			cell.resistanceOff = NonlinearResistance(cell.resistanceOff, cell.nonlinearity, cell.writeVoltage, Vr_exp, cell.readVoltage);
			cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;    // Average resistance (for energy estimation)
		}
		cell.accessVoltage = 1.1;	// Gate voltage for the transistor in 1T1R
	}

	int numRow = array->arrayRowSize;	// Transfer the parameter of # of rows from the MLP simulator to NeuroSim
	int numCol = array->arrayColSize * subArray->numCellPerSynapse;	// Transfer the parameter of # of columns from the MLP simulator to NeuroSim (times the # of cells per synapse for SRAM)
	if (param->numColMuxed > numCol) {	// Set the upperbound of param->numColMuxed
		param->numColMuxed = numCol;
	}

	cell.featureSize = array->wireWidth * 1e-9;
	if(cell.featureSize <= 0) {
		puts("NeuroSim does not take ideal array. Has Assigned the width to 200nm.");
		cell.featureSize = 200e-9;
	}

	subArray->numWriteCellPerOperationNeuro = (int)ceil((double)numCol / subArray->numWriteColMuxed);
	
	/* NeuroSim SubArray Initialization */
	double unitLengthWireResistance = array->unitLengthWireResistance;
	subArray->Initialize(numRow, numCol, unitLengthWireResistance);
	/* Recalculate wire resistance after possible layout adjustment by NeuroSim */
	array->wireResistanceRow = subArray->lengthRow / numCol * unitLengthWireResistance;
	array->wireResistanceCol = subArray->lengthCol / numRow * unitLengthWireResistance;
	/* Transfer the wire capacitances from NeuroSim to MLP simulator */
	array->wireCapRow = subArray->capRow1;
	array->wireCapCol = subArray->capCol;
	array->wireGateCapRow = subArray->capRow2;
	array->wireCapBLCol = subArray->lengthCol * 0.2e-15/1e-6;	// For BL cap of digital eNVM in 1T1R
	
	/* Transfer the resistance of access transistor in 1T1R from NeuroSim to MLP simulator */
	//for (int col=0; col<numCol; col++) {
	//	for (int row=0; row<numRow; row++) {
	//		if (eNVM *temp = dynamic_cast<eNVM*>(array->cell[col][row])) {
	//			static_cast<eNVM*>(array->cell[col][row])->resistanceAccess = cell.resCellAccess;
	//		}
	//	}
	//}
}

void NeuroSimSubArrayArea(SubArray *subArray) {
	subArray->CalculateArea();
}

double NeuroSimSubArrayReadLatency(SubArray *subArray) {	// For 1 weighted sum task on selected columns
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	if (subArray->cell.memCellType == Type::SRAM) {   // SRAM
		// XXX: To be released
	} else {	// eNVM
		if (subArray->digitalModeNeuro) {	// Digital eNVM
			// XXX: To be released
		} else {	// Analog eNVM
			if (subArray->cell.accessType == CMOS_access) {   // 1T1R
				subArray->wlDecoder.CalculateLatency(1e20, subArray->wlDecoderOutput.capNorInput, NULL, 1, 1);	// Don't care write
				subArray->wlDecoderOutput.CalculateLatency(subArray->wlDecoder.rampOutput, subArray->capRow2, subArray->resRow, 1, 1);	// Don't care write
				subArray->blSwitchMatrix.CalculateLatency(1e20, subArray->capRow1, subArray->resRow, subArray->numReadPulse, 1);    // Don't care write
				if (subArray->readCircuit.mode == CMOS) {
					double Cin = subArray->capCol + subArray->mux.capTgDrain * (2 + subArray->numColMuxed - 1) + subArray->readCircuit.capTgDrain + subArray->readCircuit.capPmosGate;
					double Imax = subArray->numRow * subArray->cell.readVoltage / subArray->cell.resMemCellOn;
					subArray->cell.readPulseWidth = Cin * subArray->readCircuit.voltageIntThreshold / Imax * subArray->readCircuit.maxNumIntPerCycle;
				} else {    // mode==OSCILLATION
					double Cin = subArray->capCol + subArray->mux.capTgDrain * (2 + subArray->numColMuxed - 1) + subArray->readCircuit.capInvInput;
					double Rmin = subArray->cell.resMemCellOn / subArray->numRow;
					double Rp = 1 / (1/Rmin + 1/subArray->readCircuit.R_OSC_OFF);
					double t_rise = -Rp * Cin * log((subArray->readCircuit.Vth - subArray->readCircuit.Vrow * Rp / Rmin) / (subArray->readCircuit.Vhold - subArray->readCircuit.Vrow * Rp / Rmin));
					subArray->cell.readPulseWidth = t_rise * subArray->readCircuit.maxNumIntPerCycle;
				}
				subArray->readCircuit.CalculateLatency(subArray->numReadPulse);
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculateLatency(subArray->numReadPulse);
				}
				return 	subArray->wlDecoderOutput.readLatency +
						subArray->blSwitchMatrix.readLatency +
						subArray->readCircuit.readLatency +
						subArray->shiftAdd.readLatency;

			} else {		// Cross-point
				subArray->wlSwitchMatrix.CalculateLatency(1e20, subArray->capRow1, subArray->resRow, subArray->numReadPulse, 1);	// Don't care write
				if (subArray->readCircuit.mode == CMOS) {
					double Cin = subArray->capCol + subArray->mux.capTgDrain * (2 + subArray->numColMuxed - 1) + subArray->readCircuit.capTgDrain + subArray->readCircuit.capPmosGate;
					double Imax = subArray->numRow * subArray->cell.readVoltage / subArray->cell.resMemCellOn;
					subArray->cell.readPulseWidth = Cin * subArray->readCircuit.voltageIntThreshold / Imax * subArray->readCircuit.maxNumIntPerCycle;
				} else {    // mode==OSCILLATION
					double Cin = subArray->capCol + subArray->mux.capTgDrain * (2 + subArray->numColMuxed - 1) + subArray->readCircuit.capInvInput;
					double Rmin = subArray->cell.resMemCellOn / subArray->numRow;
					double Rp = 1 / (1/Rmin + 1/subArray->readCircuit.R_OSC_OFF);
					double t_rise = -Rp * Cin * log((subArray->readCircuit.Vth - subArray->readCircuit.Vrow * Rp / Rmin) / (subArray->readCircuit.Vhold - subArray->readCircuit.Vrow * Rp / Rmin));
					subArray->cell.readPulseWidth = t_rise * subArray->readCircuit.maxNumIntPerCycle;
				}
				subArray->readCircuit.CalculateLatency(subArray->numReadPulse);
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculateLatency(subArray->numReadPulse);
				}
				return 	subArray->wlSwitchMatrix.readLatency +
						subArray->readCircuit.readLatency +
						subArray->shiftAdd.readLatency;
			}
		}
	}
}

double NeuroSimSubArrayWriteLatency(SubArray *subArray) {	// For 1 weight update task of whole array
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	subArray->activityRowWrite = 1;
	subArray->activityColWrite = 1;
	int numWriteOperationPerRow = (int)ceil((double)subArray->numCol * subArray->activityColWrite / subArray->numWriteCellPerOperationNeuro);
	if (subArray->cell.memCellType == Type::SRAM) {	// SRAM
		// XXX: To be released
	} else {	// eNVM
		if (subArray->digitalModeNeuro) {   // Digital eNVM
			// XXX: To be released
		} else {	// Analog eNVM
			if (subArray->cell.accessType == CMOS_access) {	// 1T1R
				subArray->wlDecoder.CalculateLatency(1e20, subArray->wlDecoderOutput.capNorInput, NULL, 1, subArray->numRow * subArray->activityRowWrite);	// Don't care read
				subArray->wlDecoderOutput.CalculateLatency(subArray->wlDecoder.rampOutput, subArray->capRow2, subArray->resRow, 1, subArray->numRow * subArray->activityRowWrite);	// Don't care read
				subArray->blSwitchMatrix.CalculateLatency(1e20, subArray->capRow1, subArray->resRow, 1, subArray->maxNumWritePulse * 2 * numWriteOperationPerRow * subArray->numRow * subArray->activityRowWrite);	// *2 means 2-step write

				return 	subArray->wlDecoder.writeLatency +
						subArray->wlDecoderOutput.writeLatency +
						subArray->blSwitchMatrix.writeLatency;

			} else {	// Cross-point
				subArray->wlSwitchMatrix.CalculateLatency(1e20, subArray->capRow1, subArray->resRow, 1, subArray->maxNumWritePulse * 2 * numWriteOperationPerRow * subArray->numRow * subArray->activityRowWrite);	// *2 means 2-step write

				return subArray->wlSwitchMatrix.writeLatency;
			}
		}
	}
}

double NeuroSimSubArrayReadEnergy(SubArray *subArray) {	// For 1 weighted sum task on selected columns
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	if (subArray->cell.memCellType == Type::SRAM) {   // SRAM
		// XXX: To be released
	} else {    // eNVM
		if (subArray->digitalModeNeuro) {   // Digital eNVM
			// XXX: To be released
		} else {	// Analog eNVM
			if (subArray->cell.accessType == CMOS_access) {   // 1T1R
				subArray->wlDecoder.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->wlDecoderOutput.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->blSwitchMatrix.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->mux.CalculatePower(subArray->numReadPulse);
				subArray->muxDecoder.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->readCircuit.CalculatePower(subArray->numReadPulse);
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculatePower(subArray->numReadPulse);
				}
				return	subArray->wlDecoder.readDynamicEnergy +
						subArray->wlDecoderOutput.readDynamicEnergy +
						subArray->blSwitchMatrix.readDynamicEnergy +
						subArray->mux.readDynamicEnergy +
						subArray->muxDecoder.readDynamicEnergy +
						subArray->readCircuit.readDynamicEnergy +
						subArray->shiftAdd.readDynamicEnergy;
			} else {	// Cross-point
				subArray->wlSwitchMatrix.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->mux.CalculatePower(subArray->numReadPulse);
				subArray->muxDecoder.CalculatePower(subArray->numReadPulse, 1);	// Don't care write
				subArray->readCircuit.CalculatePower(subArray->numReadPulse);
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculatePower(subArray->numReadPulse);
				}
				return 	subArray->wlSwitchMatrix.readDynamicEnergy +
						subArray->mux.readDynamicEnergy +
						subArray->muxDecoder.readDynamicEnergy +
						subArray->readCircuit.readDynamicEnergy +
						subArray->shiftAdd.readDynamicEnergy;
			}
		}
	}
}

double NeuroSimSubArrayWriteEnergy(SubArray *subArray) {	// For 1 weight update task of one row
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	subArray->activityRowWrite = 1;
	subArray->activityColWrite = 1;
	double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
	if (subArray->numCol * subArray->activityColWrite > subArray->numWriteCellPerOperationNeuro) {
		numWriteOperationPerRow = subArray->numCol * subArray->activityColWrite / subArray->numWriteCellPerOperationNeuro;
	} else {
		numWriteOperationPerRow = 1;
	}
	if (subArray->cell.memCellType == Type::SRAM) {   // SRAM
		// XXX: To be released
	} else {    // eNVM
		if (subArray->digitalModeNeuro) {   // Digital eNVM
			// XXX: To be released
		} else {    // Analog eNVM
			if (subArray->cell.accessType == CMOS_access) {   // 1T1R
				subArray->blSwitchMatrix.numWritePulse = 1;	// Does not matter
				subArray->slSwitchMatrix.numWritePulse = subArray->numWritePulse;
				subArray->wlDecoder.CalculatePower(1, 1);	// Don't care read
				subArray->wlDecoderOutput.CalculatePower(1, 1);	// Don't care read
				subArray->blSwitchMatrix.CalculatePower(1, 1);	// Don't care read
				subArray->slSwitchMatrix.CalculatePower(1, numWriteOperationPerRow);	// Don't care read
				return 	subArray->wlDecoder.writeDynamicEnergy +
						subArray->wlDecoderOutput.writeDynamicEnergy +
						subArray->blSwitchMatrix.writeDynamicEnergy +
						subArray->slSwitchMatrix.writeDynamicEnergy;
			} else {    // Cross-point
				subArray->wlSwitchMatrix.numWritePulse = subArray->numWritePulse;
				subArray->blSwitchMatrix.numWritePulse = subArray->numWritePulse;
				subArray->wlSwitchMatrix.CalculatePower(1, 1);	// Don't care read
				subArray->blSwitchMatrix.CalculatePower(1, numWriteOperationPerRow);	// Don't care read
				return  subArray->wlSwitchMatrix.writeDynamicEnergy +
						subArray->blSwitchMatrix.writeDynamicEnergy;
			}
		}
	}
}

double NeuroSimSubArrayLeakagePower(SubArray *subArray) {
	if (subArray->cell.memCellType == Type::SRAM) {	// SRAM
		// XXX: To be released
	} else {	// eNVM
		if (subArray->digitalModeNeuro) {   // Digital eNVM
			// XXX: To be released
		} else {    // Analog eNVM
			if (subArray->cell.accessType == CMOS_access) {	// 1T1R
				subArray->wlDecoder.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->wlDecoderOutput.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->blSwitchMatrix.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->slSwitchMatrix.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->mux.CalculatePower(1);	// Don't care numRead
				subArray->muxDecoder.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->readCircuit.CalculatePower(1);	// Don't care numRead
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculatePower(1);	// Don't care numRead
				}
				subArray->leakage += subArray->wlDecoder.leakage;
				subArray->leakage += subArray->wlDecoderOutput.leakage;
				subArray->leakage += subArray->blSwitchMatrix.leakage;
				subArray->leakage += subArray->slSwitchMatrix.leakage;
				subArray->leakage += subArray->mux.leakage;
				subArray->leakage += subArray->muxDecoder.leakage;
				subArray->leakage += subArray->readCircuit.leakage;
				subArray->leakage += subArray->shiftAdd.leakage;
			} else {	// Cross-point
				subArray->wlSwitchMatrix.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->blSwitchMatrix.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->mux.CalculatePower(1);	// Don't care numRead
				subArray->muxDecoder.CalculatePower(1, 1);	// Don't care numRead and numWrite
				subArray->readCircuit.CalculatePower(1);	// Don't care numRead
				if (subArray->shiftAddEnable) {
					subArray->shiftAdd.CalculatePower(1);	// Don't care numRead
				}
				subArray->leakage += subArray->wlSwitchMatrix.leakage;
				subArray->leakage += subArray->blSwitchMatrix.leakage;
				subArray->leakage += subArray->mux.leakage;
				subArray->leakage += subArray->muxDecoder.leakage;
				subArray->leakage += subArray->readCircuit.leakage;
				subArray->leakage += subArray->shiftAdd.leakage;
			}
		}
	}
}

void NeuroSimNeuronInitialize(SubArray *& subArray, InputParameter& inputParameter, Technology& tech, MemCell& cell, Adder& adder, Mux& mux, RowDecoder& muxDecoder, DFF& dff) {
	int numAdderBit;
	if (subArray->shiftAddEnable) {	// Here we only support adder in non-spiking fashion
		numAdderBit = subArray->shiftAdd.numAdderBit + 1 + subArray->shiftAdd.numReadPulse - 1;
	} else {
		if (cell.memCellType == Type::SRAM) {	// SRAM
			// XXX: To be released
		} else {	// eNVM
			numAdderBit = param->numBitPartialSum;
		}
	}
	numAdderBit = numAdderBit + 2;	// Need 1 more bit for *2 in 2w'-1 of MLP algorithm, and 1 more bit for 2's complement implementation
	
	int numAdder = (int)ceil((double)subArray->numCol/subArray->numCellPerSynapse/subArray->numColMuxed);

	// Only need the MSB of adder output to determine it is positive or negative in 2's complement
	dff.Initialize(subArray->numCol/subArray->numCellPerSynapse, subArray->clkFreq);
	if (subArray->numColMuxed > 1) {
		mux.Initialize(numAdder*param->numBitInput, subArray->numColMuxed, NULL, true);    // Digital Mux
		muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(subArray->numColMuxed)), true);
		adder.Initialize(numAdderBit, numAdder);
	} else {	// No need for Mux and Mux decoder
		adder.Initialize(numAdderBit, numAdder);
	}
}

void NeuroSimNeuronArea(SubArray *subArray, Adder& adder, Mux& mux, RowDecoder& muxDecoder, DFF& dff, double *height, double *width) {
	adder.CalculateArea(NULL, subArray->widthArray, NONE);
	if (subArray->numColMuxed > 1) {
		mux.CalculateArea(NULL, subArray->widthArray, NONE);	// Digital Mux
		muxDecoder.CalculateArea(mux.height * 4, NULL, NONE);	// Set muxDecoder height to be 4 times (for example) mux height to avoid large muxDecoder width
	}
	dff.CalculateArea(NULL, subArray->widthArray, NONE);
	
	*height = MAX(adder.height + mux.height + dff.height, muxDecoder.height);
	*width = subArray->widthArray + muxDecoder.width;
}

double NeuroSimNeuronReadLatency(SubArray *subArray, Adder& adder, Mux& mux, RowDecoder& muxDecoder, DFF& dff) {	// For 1 weighted sum task on selected columns
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	if (subArray->numColMuxed > 1) {
		adder.CalculateLatency(1e20, mux.capTgDrain, 1);
		mux.CalculateLatency(adder.rampOutput, dff.capTgDrain, 1);
		muxDecoder.CalculateLatency(1e20, mux.capTgGateN * adder.numAdder, mux.capTgGateP * adder.numAdder, 1, 1);	// Don't care write
	} else {	// No need for Mux and Mux decoder
		adder.CalculateLatency(1e20, dff.capTgDrain, 1);
	}
	dff.CalculateLatency(1e20, 1);
	return adder.readLatency + mux.readLatency + dff.readLatency;
}

double NeuroSimNeuronReadEnergy(SubArray *subArray, Adder& adder, Mux& mux, RowDecoder& muxDecoder, DFF& dff) {	// For 1 weighted sum task on selected columns
	if (!param->NeuroSimDynamicPerformance) { return 0; }	// Skip this function if param->NeuroSimDynamicPerformance is false
	adder.CalculatePower(1, adder.numAdder);
	if (subArray->numColMuxed > 1) {
		mux.CalculatePower(1);
		muxDecoder.CalculatePower(1, 1);	// Don't care write
	}
	dff.CalculatePower(1, adder.numAdder);
	return adder.readDynamicEnergy + mux.readDynamicEnergy + muxDecoder.readDynamicEnergy + dff.readDynamicEnergy;
}

double NeuroSimNeuronLeakagePower(SubArray *subArray, Adder& adder, Mux& mux, RowDecoder& muxDecoder, DFF& dff) { // Same as NeuroSimNeuronReadEnergy
	adder.CalculatePower(1, adder.numAdder);
	if (subArray->numColMuxed > 1) {
		mux.CalculatePower(1);
		muxDecoder.CalculatePower(1, 1);	// Don't care write
	}
	dff.CalculatePower(1, adder.numAdder);
	return adder.leakage + mux.leakage + muxDecoder.leakage + dff.leakage;
}

