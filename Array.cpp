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

#include "formula.h"
#include "Array.h"

double Array::ReadCell(int x, int y) {
	if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(**cell)) {	// Analog eNVM
		double readVoltage = static_cast<eNVM*>(cell[x][y])->readVoltage;
		double totalWireResistance;
		if (static_cast<eNVM*>(cell[x][y])->cmosAccess) {
			if (static_cast<eNVM*>(cell[x][y])->FeFET) {    // FeFET
				// XXX: To be released
			} else {	// Normal
				totalWireResistance = (x + 1) * wireResistanceRow + (arrayRowSize - y) * wireResistanceCol + static_cast<eNVM*>(cell[x][y])->resistanceAccess;
			}
		} else {
			totalWireResistance = (x + 1) * wireResistanceRow + (arrayRowSize - y) * wireResistanceCol;
		}
		double cellCurrent;
		if (static_cast<eNVM*>(cell[x][y])->nonlinearIV) {
			/* Bisection method to calculate read current with nonlinearity */
			int maxIter = 30;
			double v1 = 0, v2 = readVoltage, v3;
			double wireCurrent;
			for (int iter=0; iter<maxIter; iter++) {
				//printf("iter: %d, %f\t%f\n", iter, v1, v2);
				v3 = (v1 + v2)/2;
				wireCurrent = (readVoltage - v3)/totalWireResistance;
				cellCurrent = static_cast<AnalogNVM*>(cell[x][y])->Read(v3);
				if (wireCurrent > cellCurrent)
					v1 = v3;
				else
					v2 = v3;
			}
		} else {	// No nonlinearity
			if (static_cast<eNVM*>(cell[x][y])->readNoise) {
				extern std::mt19937 gen;
				cellCurrent = readVoltage / (1/static_cast<eNVM*>(cell[x][y])->conductance * (1 + (*static_cast<eNVM*>(cell[x][y])->gaussian_dist)(gen)) + totalWireResistance);
			} else {
				cellCurrent = readVoltage / (1/static_cast<eNVM*>(cell[x][y])->conductance + totalWireResistance);
			}
		}
		return cellCurrent;

	} else {	// SRAM or digital eNVM
		// XXX: To be released
	}
}

void Array::WriteCell(int x, int y, double deltaWeight, double maxWeight, double minWeight, 
						bool regular /* False: ideal write, True: regular write considering device properties */, bool writeEnergyReport) {
	// TODO: include wire resistance
	double deltaWeightNormalized = deltaWeight / (maxWeight - minWeight);
	if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(**cell)) { // Analog eNVM
		if (regular) {	// Regular write
			static_cast<AnalogNVM*>(cell[x][y])->Write(deltaWeightNormalized, writeEnergyReport, wireCapCol);
		} else {	// Preparation stage (ideal write)
			double conductance = static_cast<eNVM*>(cell[x][y])->conductance;
			double maxConductance = static_cast<eNVM*>(cell[x][y])->maxConductance;
			double minConductance = static_cast<eNVM*>(cell[x][y])->minConductance;
			conductance += deltaWeightNormalized * (maxConductance - minConductance);
			if (conductance > maxConductance) {
				conductance = maxConductance;
			} else if (conductance < minConductance) {
				conductance = minConductance;
			}
			static_cast<eNVM*>(cell[x][y])->conductance = conductance;
		}
	} else {    // SRAM or digital eNVM
		// XXX: To be released
	}
}

double Array::GetMaxCellReadCurrent(int x, int y) {
	return static_cast<AnalogNVM*>(cell[x][y])->GetMaxReadCurrent();
}

double Array::ConductanceToWeight(int x, int y, double maxWeight, double minWeight) {
	if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(**cell)) {	// Analog eNVM
		/* Measure current */
		double I = this->ReadCell(x, y);
		/* Convert current to weight */
		double Imax = static_cast<AnalogNVM*>(cell[x][y])->GetMaxReadCurrent();
		double Imin = static_cast<AnalogNVM*>(cell[x][y])->GetMinReadCurrent();
		if (I<Imin)
			I = Imin;
		else if (I>Imax)
			I = Imax;
		return (I-Imin) / (Imax-Imin) * (maxWeight-minWeight) + minWeight;
	} else {	// SRAM or digital eNVM
		// XXX: To be released
	}
}

