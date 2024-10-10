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

#include <ctime>
#include "formula.h"
#include "Cell.h"

/* General eNVM */
void AnalogNVM::WriteEnergyCalculation(double conductance, double conductanceNew, double wireCapCol) {
	if (nonlinearIV) {  // Currently only for cross-point array
		/* I-V nonlinearity */
		conductanceAtVwLTP = NonlinearConductance(conductance, NL, writeVoltageLTP, readVoltage, writeVoltageLTP);
		conductanceAtHalfVwLTP = NonlinearConductance(conductance, NL, writeVoltageLTP, readVoltage, writeVoltageLTP/2);
		conductanceAtVwLTD = NonlinearConductance(conductance, NL, writeVoltageLTD, readVoltage, writeVoltageLTD);
		conductanceAtHalfVwLTD = NonlinearConductance(conductance, NL, writeVoltageLTD, readVoltage, writeVoltageLTD/2);
		double conductanceNewAtVwLTP = NonlinearConductance(conductanceNew, NL, writeVoltageLTP, readVoltage, writeVoltageLTP);
		double conductanceNewAtHalfVwLTP = NonlinearConductance(conductanceNew, NL, writeVoltageLTP, readVoltage, writeVoltageLTP/2);
		double conductanceNewAtVwLTD = NonlinearConductance(conductanceNew, NL, writeVoltageLTD, readVoltage, writeVoltageLTD);
		double conductanceNewAtHalfVwLTD = NonlinearConductance(conductanceNew, NL, writeVoltageLTD, readVoltage, writeVoltageLTD/2);
		if (numPulse > 0) { // If the cell needs LTP pulses
			writeEnergy = writeVoltageLTP * writeVoltageLTP * (conductanceAtVwLTP+conductanceNewAtVwLTP)/2 * writePulseWidthLTP * numPulse;
			writeEnergy += writeVoltageLTP * writeVoltageLTP * wireCapCol * numPulse;
			writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * conductanceNewAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD;    // Half-selected during LTD phase (use the new conductance value if LTP phase is before LTD phase)
			writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * wireCapCol;
		} else if (numPulse < 0) {  // If the cell needs LTD pulses
			writeEnergy = writeVoltageLTP/2 * writeVoltageLTP/2 * conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP;    // Half-selected during LTP phase (use the old conductance value if LTP phase is before LTD phase)
			writeEnergy += writeVoltageLTP/2 * writeVoltageLTP/2 * wireCapCol;
			writeEnergy += writeVoltageLTD * writeVoltageLTD * wireCapCol * (-numPulse);
			writeEnergy += writeVoltageLTD * writeVoltageLTD * (conductanceAtVwLTD+conductanceNewAtVwLTD)/2 * writePulseWidthLTD * (-numPulse);
		} else {    // Half-selected during both LTP and LTD phases
			writeEnergy = writeVoltageLTP/2 * writeVoltageLTP/2 * conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP;
			writeEnergy += writeVoltageLTP/2 * writeVoltageLTP/2 * wireCapCol;
			writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * conductanceAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD;
			writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * wireCapCol;
		}
		/* Update the nonlinear conductances with new values */
		conductanceAtVwLTP = conductanceNewAtVwLTP;
		conductanceAtHalfVwLTP = conductanceNewAtHalfVwLTP;
		conductanceAtVwLTD = conductanceNewAtVwLTD;
		conductanceAtHalfVwLTD = conductanceNewAtHalfVwLTD;
	} else {    // If not cross-point array or not considering I-V nonlinearity
		if (FeFET) {    // FeFET structure
			// XXX: To be released
			
		} else {
			if (numPulse > 0) { // If the cell needs LTP pulses
				writeEnergy = writeVoltageLTP * writeVoltageLTP * (conductance+conductanceNew)/2 * writePulseWidthLTP * numPulse;
				writeEnergy += writeVoltageLTP * writeVoltageLTP * wireCapCol * numPulse;
				if (!cmosAccess) {	// Crossbar
					writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * conductanceNew * writePulseWidthLTD * maxNumLevelLTD;    // Half-selected during LTD phase (use the new conductance value if LTP phase is before LTD phase)
					writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * wireCapCol;
				}
			} else if (numPulse < 0) {  // If the cell needs LTD pulses
				if (!cmosAccess) {	// Crossbar
					writeEnergy = writeVoltageLTP/2 * writeVoltageLTP/2 * conductance * writePulseWidthLTP * maxNumLevelLTP;    // Half-selected during LTP phase (use the old conductance value if LTP phase is before LTD phase)
					writeEnergy += writeVoltageLTP/2 * writeVoltageLTP/2 * wireCapCol;
				} else {	// 1T1R
					writeEnergy = writeVoltageLTP * writeVoltageLTP * wireCapCol;
				}
				writeEnergy += writeVoltageLTD * writeVoltageLTD * wireCapCol * (-numPulse);
				writeEnergy += writeVoltageLTD * writeVoltageLTD * (conductance+conductanceNew)/2 * writePulseWidthLTD * (-numPulse);
			} else {    // Half-selected during both LTP and LTD phases
				if (!cmosAccess) {	// Crossbar
					writeEnergy = writeVoltageLTP/2 * writeVoltageLTP/2 * conductance * writePulseWidthLTP * maxNumLevelLTP;
					writeEnergy += writeVoltageLTP/2 * writeVoltageLTP/2 * wireCapCol;
					writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * conductance * writePulseWidthLTD * maxNumLevelLTD;
					writeEnergy += writeVoltageLTD/2 * writeVoltageLTD/2 * wireCapCol;
				} else {	// 1T1R
					writeEnergy = writeVoltageLTP * writeVoltageLTP * wireCapCol;
				}
			}
		}
	}
}

/* Ideal device (no weight update nonlinearity) */
IdealDevice::IdealDevice(int x, int y) {
	this->x = x; this->y = y;	// Cell location: x (column) and y (row) start from index 0
	maxConductance = 5e-6;		// Maximum cell conductance (S)
	minConductance = 100e-9;	// Minimum cell conductance (S)
	conductance = minConductance;	// Current conductance (S) (dynamic variable)
	readVoltage = 0.5;	// On-chip read voltage (Vr) (V)
	readPulseWidth = 5e-9;	// Read pulse width (s) (will be determined by ADC)
	writeVoltageLTP = 2;	// Write voltage (V) for LTP or weight increase
	writeVoltageLTD = 2;	// Write voltage (V) for LTD or weight decrease
	writePulseWidthLTP = 10e-9;	// Write pulse width (s) for LTP or weight increase
	writePulseWidthLTD = 10e-9;	// Write pulse width (s) for LTD or weight decrease
	writeEnergy = 0;	// Dynamic variable for calculation of write energy (J)
	maxNumLevelLTP = 63;	// Maximum number of conductance states during LTP or weight increase
	maxNumLevelLTD = 63;	// Maximum number of conductance states during LTD or weight decrease
	numPulse = 0;	// Number of write pulses used in the most recent write operation (dynamic variable)
	cmosAccess = true;	// True: Pseudo-crossbar (1T1R), false: cross-point
	FeFET = false;      // True: FeFET structure (Pseudo-crossbar only, should be cmosAccess=1)
	resistanceAccess = 15e3;	// The resistance of transistor (Ohm) in Pseudo-crossbar array when turned ON
	nonlinearIV = false;	// Consider I-V nonlinearity or not (Currently for cross-point array only)
	NL = 10;	// Nonlinearity in write scheme (the current ratio between Vw and Vw/2), assuming for the LTP side
	if (nonlinearIV) {	// Currently for cross-point array only
		double Vr_exp = readVoltage;  // XXX: Modify this value to Vr in the reported measurement data (can be different than readVoltage)
		// Calculation of conductance at on-chip Vr
		maxConductance = NonlinearConductance(maxConductance, NL, writeVoltageLTP, Vr_exp, readVoltage);
		minConductance = NonlinearConductance(minConductance, NL, writeVoltageLTP, Vr_exp, readVoltage);
	}
	readNoise = false;	// Consider read noise or not
	sigmaReadNoise = 0.25;	// Sigma of read noise in gaussian distribution
	gaussian_dist = new std::normal_distribution<double>(0, sigmaReadNoise);	// Set up mean and stddev for read noise

	heightInFeatureSize = cmosAccess? 4 : 2;	// Cell height = 4F (Pseudo-crossbar) or 2F (cross-point)
	widthInFeatureSize = cmosAccess? (FeFET? 6 : 4) : 2;    // Cell width = 6F (FeFET) or 4F (Pseudo-crossbar) or 2F (cross-point)
}

double IdealDevice::Read(double voltage) {
	extern std::mt19937 gen;
	// TODO: nonlinear read
	if (readNoise) {
		return voltage * conductance * (1 + (*gaussian_dist)(gen));
	} else {
		return voltage * conductance;
	}
}

void IdealDevice::Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol) {
	if (deltaWeightNormalized >= 0) {
		deltaWeightNormalized = truncate(deltaWeightNormalized, maxNumLevelLTP);
		numPulse = deltaWeightNormalized * maxNumLevelLTP;
	} else {
		deltaWeightNormalized = truncate(deltaWeightNormalized, maxNumLevelLTD);
		numPulse = deltaWeightNormalized * maxNumLevelLTD;	// will be a negative number
	}
	double conductanceNew = conductance + deltaWeightNormalized * (maxConductance - minConductance);
	if (conductanceNew > maxConductance) {
		conductanceNew = maxConductance;
	} else if (conductanceNew < minConductance) {
		conductanceNew = minConductance;
	}
	
	/* Energy calculation */
	if (writeEnergyReport) {
		WriteEnergyCalculation(conductance, conductanceNew, wireCapCol);
	}

	conductance = conductanceNew;
}

/* Real Device */
RealDevice::RealDevice(int x, int y) {
	this->x = x; this->y = y;	// Cell location: x (column) and y (row) start from index 0
	maxConductance = 3.8462e-8;		// Maximum cell conductance (S)
	minConductance = 3.0769e-9;	// Minimum cell conductance (S)
	conductance = minConductance;	// Current conductance (S) (dynamic variable)
	readVoltage = 0.5;	// On-chip read voltage (Vr) (V)
	readPulseWidth = 5e-9;	// Read pulse width (s) (will be determined by ADC)
	writeVoltageLTP = 3.2;	// Write voltage (V) for LTP or weight increase
	writeVoltageLTD = 2.8;	// Write voltage (V) for LTD or weight decrease
	writePulseWidthLTP = 300e-6;	// Write pulse width (s) for LTP or weight increase
	writePulseWidthLTD = 300e-6;	// Write pulse width (s) for LTD or weight decrease
	writeEnergy = 0;	// Dynamic variable for calculation of write energy (J)
	maxNumLevelLTP = 97;	// Maximum number of conductance states during LTP or weight increase
	maxNumLevelLTD = 100;	// Maximum number of conductance states during LTD or weight decrease
	numPulse = 0;	// Number of write pulses used in the most recent write operation (dynamic variable)
	cmosAccess = true;	// True: Pseudo-crossbar (1T1R), false: cross-point
	FeFET = false;      // True: FeFET structure (Pseudo-crossbar only, should be cmosAccess=1)
	resistanceAccess = 15e3;	// The resistance of transistor (Ohm) in Pseudo-crossbar array when turned ON
	nonlinearIV = false;	// Consider I-V nonlinearity or not (Currently for cross-point array only)
	NL = 10;    // I-V nonlinearity in write scheme (the current ratio between Vw and Vw/2), assuming for the LTP side
	if (nonlinearIV) {  // Currently for cross-point array only
		double Vr_exp = readVoltage;  // XXX: Modify this value to Vr in the reported measurement data (can be different than readVoltage)
		// Calculation of conductance at on-chip Vr
		maxConductance = NonlinearConductance(maxConductance, NL, writeVoltageLTP, Vr_exp, readVoltage);
		minConductance = NonlinearConductance(minConductance, NL, writeVoltageLTP, Vr_exp, readVoltage);
	}
	nonlinearWrite = true;	// Consider weight update nonlinearity or not
	readNoise = false;		// Consider read noise or not
	sigmaReadNoise = 0;		// Sigma of read noise in gaussian distribution
	gaussian_dist = new std::normal_distribution<double>(0, sigmaReadNoise);	// Set up mean and stddev for read noise

	std::mt19937 localGen;	// It's OK not to use the external gen, since here the device-to-device vairation is a one-time deal
	localGen.seed(std::time(0));
	
	/* Device-to-device weight update variation */
	NL_LTP = 2.4; // LTP nonlinearity
	NL_LTD = -4.88; // LTD nonlinearity
	sigmaDtoD = 0;	// Sigma of device-to-device weight update vairation in gaussian distribution
	gaussian_dist2 = new std::normal_distribution<double>(0, sigmaDtoD);	// Set up mean and stddev for device-to-device weight update vairation
	paramALTP = getParamA(NL_LTP + (*gaussian_dist2)(localGen)) * maxNumLevelLTP;	// Parameter A for LTP nonlinearity
	paramALTD = getParamA(NL_LTD + (*gaussian_dist2)(localGen)) * maxNumLevelLTD;	// Parameter A for LTD nonlinearity

	/* Cycle-to-cycle weight update variation */
	sigmaCtoC = 0.035 * (maxConductance - minConductance);	// Sigma of cycle-to-cycle weight update vairation: defined as the percentage of conductance range
	gaussian_dist3 = new std::normal_distribution<double>(0, sigmaCtoC);    // Set up mean and stddev for cycle-to-cycle weight update vairation

	heightInFeatureSize = cmosAccess? 4 : 2;	// Cell height = 4F (Pseudo-crossbar) or 2F (cross-point)
	widthInFeatureSize = cmosAccess? (FeFET? 6 : 4) : 2;    // Cell width = 6F (FeFET) or 4F (Pseudo-crossbar) or 2F (cross-point)
}

double RealDevice::Read(double voltage) {	// Return read current (A)
	extern std::mt19937 gen;
	if (nonlinearIV) {
		// TODO: nonlinear read
		if (readNoise) {
			return voltage * conductance * (1 + (*gaussian_dist)(gen));
		} else {
			return voltage * conductance;
		}
	} else {
		if (readNoise) {
			return voltage * conductance * (1 + (*gaussian_dist)(gen));
		} else {
			return voltage * conductance;
		}
	}
}

void RealDevice::Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol) {
	double conductanceNew = conductance;	// =conductance if no update
	if (deltaWeightNormalized > 0) {	// LTP
		deltaWeightNormalized = truncate(deltaWeightNormalized, maxNumLevelLTP);
		numPulse = deltaWeightNormalized * maxNumLevelLTP;
		if (nonlinearWrite) {
			paramBLTP = (maxConductance - minConductance) / (1 - exp(-maxNumLevelLTP/paramALTP));
			xPulse = InvNonlinearWeight(conductance, maxNumLevelLTP, paramALTP, paramBLTP, minConductance);
			conductanceNew = NonlinearWeight(xPulse+numPulse, maxNumLevelLTP, paramALTP, paramBLTP, minConductance);
		} else {
			xPulse = (conductance - minConductance) / (maxConductance - minConductance) * maxNumLevelLTP;
			conductanceNew = (xPulse+numPulse) / maxNumLevelLTP * (maxConductance - minConductance) + minConductance;
		}
	} else {	// LTD
		deltaWeightNormalized = truncate(deltaWeightNormalized, maxNumLevelLTD);
		numPulse = deltaWeightNormalized * maxNumLevelLTD;	// will be a negative number
		if (nonlinearWrite) {
			paramBLTD = (maxConductance - minConductance) / (1 - exp(-maxNumLevelLTD/paramALTD));
			xPulse = InvNonlinearWeight(conductance, maxNumLevelLTD, paramALTD, paramBLTD, minConductance);
			conductanceNew = NonlinearWeight(xPulse+numPulse, maxNumLevelLTD, paramALTD, paramBLTD, minConductance);
		} else {
			xPulse = (conductance - minConductance) / (maxConductance - minConductance) * maxNumLevelLTD;
			conductanceNew = (xPulse+numPulse) / maxNumLevelLTD * (maxConductance - minConductance) + minConductance;
		}
	}
	
	/* Cycle-to-cycle variation */
	extern std::mt19937 gen;
	if (sigmaCtoC && numPulse != 0) {
		conductanceNew += (*gaussian_dist3)(gen) * sqrt(abs(numPulse));	// Absolute variation
	}
	
	if (conductanceNew > maxConductance) {
		conductanceNew = maxConductance;
	} else if (conductanceNew < minConductance) {
		conductanceNew = minConductance;
	}
	
	/* Energy calculation */
	if (writeEnergyReport) {
		WriteEnergyCalculation(conductance, conductanceNew, wireCapCol);
	}
	conductance = conductanceNew;
}

/* Measured device */
MeasuredDevice::MeasuredDevice(int x, int y) {
	// XXX: To be released
}

double MeasuredDevice::Read(double voltage) {
	// XXX: To be released
}

void MeasuredDevice::Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol) {
	// XXX: To be released
}

/* SRAM */
SRAM::SRAM(int x, int y) {
	// XXX: To be released
}

/* Digital eNVM */
DigitalNVM::DigitalNVM(int x, int y) {
	// XXX: To be released
}

double DigitalNVM::Read(double voltage) {
	// XXX: To be released
}

void DigitalNVM::Write(int bitNew, double wireCapCol) {
	// XXX: To be released
}

