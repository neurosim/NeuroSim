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

#ifndef CELL_H_
#define CELL_H_

#include <random>
#include <vector>

class Cell {
public:
	int x, y;	// Cell location: x (column) and y (row) start from index 0
	double heightInFeatureSize, widthInFeatureSize;	// Cell height/width in terms of feature size (F)
	double area;	// Cell area (m^2)
	virtual ~Cell() {}	// Add a virtual function to enable dynamic_cast
};

class eNVM: public Cell {
public:
	double readVoltage;	// On-chip read voltage (Vr) (V)
	double readPulseWidth;	// Read pulse width (s) (will be determined by ADC)
	double readEnergy;	// Dynamic variable for calculation of read energy (J)
	double writeVoltageLTP;	// Write voltage (V) for LTP or weight increase
	double writeVoltageLTD;	// Write voltage (V) for LTD or weight decrease
	double writePulseWidthLTP;	// Write pulse width (s) of LTP or weight increase
	double writePulseWidthLTD;	// Write pulse width (s) of LTD or weight decrease
	double writeEnergy;	// Dynamic variable for calculation of write energy (J)
	double conductance;	// Current conductance (S) (Dynamic variable) at on-chip Vr (different than the Vr in the reported measurement data)
	double maxConductance;	// Maximum cell conductance (S)
	double minConductance;	// Minimum cell conductance (S)
	bool cmosAccess;	// True: Pseudo-crossbar (1T1R), false: cross-point
	bool FeFET;         // True: FeFET structure (Pseudo-crossbar only, should be cmosAccess=1)
	double resistanceAccess;	// The resistance of transistor (Ohm) in Pseudo-crossbar array when turned ON
	bool nonlinearIV;	// Consider I-V nonlinearity or not (Currently this option is for cross-point array. It is hard to have this option in pseudo-crossbar since it has an access transistor and the transistor's resistance can be comparable to RRAM's resistance after considering the nonlinearity. In this case, we have to iteratively find both the resistance and Vw across RRAM.)
	bool readNoise;	// Consider read noise or not
	double sigmaReadNoise;	// Sigma of read noise in gaussian distribution
	double NL;	// Nonlinearity in write scheme (the current ratio between Vw and Vw/2), assuming for the LTP side
	std::normal_distribution<double> *gaussian_dist;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist2;	// Normal distribution object
	std::normal_distribution<double> *gaussian_dist3;	// Normal distribution object
	/* Need the 4 variables below if nonlinearIV=true */
	double conductanceAtVwLTP;		// Conductance at the LTP write voltage
	double conductanceAtVwLTD;		// Conductance at the LTD write voltage
	double conductanceAtHalfVwLTP;	// Conductance at 1/2 LTP write voltage
	double conductanceAtHalfVwLTD;	// Conductance at 1/2 LTD write voltage
};

class SRAM: public Cell {
public:
	SRAM(int x, int y);
	// XXX: To be released
};

class AnalogNVM: public eNVM {
public:
	double numBitLTP;	// Number of bits in the conductance states during LTP or weight increase
	double numBitLTD;	// Number of bits in the conductance states during LTD or weight decrease
	int maxNumLevelLTP;	// Maximum number of conductance states during LTP or weight increase
	int maxNumLevelLTD;	// Maximum number of conductance states during LTD or weight decrease
	int numPulse;   // Number of write pulses used in the most recent write operation (Positive number: LTP, Negative number: LTD) (dynamic variable)

	virtual double Read(double voltage) = 0;
	virtual void Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol) = 0;
	double GetMaxReadCurrent() {return readVoltage * maxConductance;}
	double GetMinReadCurrent() {return readVoltage * minConductance;}
	void WriteEnergyCalculation(double conductance, double conductanceNew, double wireCapCol);
};

class DigitalNVM: public eNVM {
public:
	DigitalNVM(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(int bitNew, double wireCapCol);
	// XXX: To be released
};

class IdealDevice: public AnalogNVM {
public:
	IdealDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol);
};

class RealDevice: public AnalogNVM {
public:
	bool nonlinearWrite;	// Consider weight update nonlinearity or not
	double xPulse;		// Conductance state in terms of the pulse number (doesn't need to be integer)
	double NL_LTP;      // LTP nonlinearity
	double NL_LTD;      // LTD nonlinearity
	double paramALTP;	// Parameter A for LTP nonlinearity
	double paramBLTP;	// Parameter B for LTP nonlinearity
	double paramALTD;	// Parameter A for LTD nonlinearity
	double paramBLTD;	// Parameter B for LTD nonlinearity
	double sigmaDtoD;	// Sigma of device-to-device variation on weight update nonliearity baseline
	double sigmaCtoC;	// Sigma of cycle-to-cycle variation on weight update

	RealDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol);
};

class MeasuredDevice: public AnalogNVM {
public:
	MeasuredDevice(int x, int y);
	double Read(double voltage);	// Return read current (A)
	void Write(double deltaWeightNormalized, bool writeEnergyReport, double wireCapCol);
	// XXX: To be released
};

#endif
