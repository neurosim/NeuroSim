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
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include "Bus.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "ProcessingUnit.h"
#include "Param.h"
#include "RowDecoder.h"


using namespace std;

extern Param *param;

Bus *busInput;
Bus *busOutput;
RowDecoder *predecoder;
DFF *dff;

void ProcessingUnitInitialize(SubArray *& subArray, InputParameter& inputParameter, Technology& tech, MemCell& cell, int _numSubArrayRow, int _numSubArrayCol) {

	/*** circuit level parameters ***/
	switch(param->memcelltype) {
		case 3:     cell.memCellType = Type::FeFET; break;
		case 2:	    cell.memCellType = Type::RRAM; break;
		case 1:	    cell.memCellType = Type::SRAM; break;
		case 4: 	cell.memCellType = Type::STTMRAM; break;
		case 5: 	cell.memCellType = Type::SOTMRAM; break;
		case 6: 	cell.memCellType = Type::eDRAM; break;
		case -1:	break;
		default:	exit(-1);
	}
					
	switch(param->transistortype) {
		case 3:	    inputParameter.transistorType = TFET;          break;
		case 2:	    inputParameter.transistorType = FET_2D;        break;
		case 1:	    inputParameter.transistorType = conventional;  break;
		case -1:	break;
		default:	exit(-1);
	}
	
	switch(param->deviceroadmap) {
		case 2:	    inputParameter.deviceRoadmap = LSTP;  break;
		case 1:	    inputParameter.deviceRoadmap = HP;    break;
		case -1:	break;
		default:	exit(-1);
	}
	
	subArray = new SubArray(inputParameter, tech, cell);
	busInput = new Bus(inputParameter, tech, cell);
	busOutput = new Bus(inputParameter, tech, cell);
	predecoder = new RowDecoder(inputParameter, tech, cell);
	dff = new DFF(inputParameter, tech, cell);
	
	/* Create SubArray object and link the required global objects (not initialization) */
	inputParameter.temperature = param->temp;   // Temperature (K)
	inputParameter.processNode = param->technode;    // Technology node
	tech.Initialize(inputParameter.processNode, inputParameter.deviceRoadmap, inputParameter.transistorType);

	cell.resistanceOn = param->resistanceOn;	                                // Ron resistance at Vr in the reported measurement data (need to recalculate below if considering the nonlinearity)
	cell.resistanceOff = param->resistanceOff;	                                // Roff resistance at Vr in the reported measurement dat (need to recalculate below if considering the nonlinearity)
	cell.resistanceAvg = (cell.resistanceOn + cell.resistanceOff)/2;            // Average resistance (for energy estimation)
	cell.readVoltage = param->readVoltage;	                                    // On-chip read voltage for memory cell
	cell.readPulseWidth = param->readPulseWidth;
	cell.accessVoltage = param->accessVoltage;                                       // Gate voltage for the transistor in 1T1R
	cell.resistanceAccess = param->resistanceAccess;
	cell.featureSize = param->featuresize; 
	cell.writeVoltage = param->writeVoltage;
	cell.widthAccessCMOSWrite = param->widthAccessCMOSWrite;
	cell.widthAccessCMOS = param->widthAccessCMOS;
	cell.capDRAMCell = param->capDRAMCell;
	cell.retentionTime = param->retentionTime;
	cell.setCurrentMTJ = param->setCurrentMTJ;
	cell.setCurrent = param->setCurrent;
	cell.resistanceSOT = param->resistanceSOT;
	cell.minSenseVoltage = param->minSenseVoltage;
	cell.writePulseWidth = param->writePulseWidth;
	// cell.gateCapFeFET = param->gateCapFeFET;	        // Gate capacitance of FeFET (F)
	// cell.polarization = param->polarization;

	if (cell.memCellType == Type::SRAM) {   // SRAM
		cell.heightInFeatureSize = param->heightInFeatureSizeSRAM;                   // Cell height in feature size
		cell.widthInFeatureSize = param->widthInFeatureSizeSRAM;                     // Cell width in feature size
		cell.widthSRAMCellNMOS = param->widthSRAMCellNMOS;
		cell.widthSRAMCellPMOS = param->widthSRAMCellPMOS;
		// cell.widthAccessCMOS = param->widthAccessCMOS;
		// cell.minSenseVoltage = param->minSenseVoltage;
	} else {
		cell.heightInFeatureSize = param->heightInFeatureSize1T1R;         // Cell height in feature size
		cell.widthInFeatureSize = param->widthInFeatureSize1T1R;            // Cell width in feature size
	} 
        
	subArray->numRow = param->numRowSubArray;
	subArray->numCol = param->numRowSubArray;
    subArray->clkFreq = param->clkFreq;                       // Clock frequency
	subArray->validated = param->validated;
	subArray->spikingMode = NONSPIKING;
	
	int numRow = param->numRowSubArray;
	int numCol = param->numColSubArray;
	
    subArray->maxNumWritePulse = MAX(cell.maxNumLevelLTP, cell.maxNumLevelLTD);

	int numSubArrayRow = _numSubArrayRow;
	int numSubArrayCol = _numSubArrayCol;

	/*** initialize modules ***/
	subArray->Initialize(numRow, numCol, param->unitLengthWireResistance);        // initialize subArray	
	subArray->CalculateArea();

	busInput->Initialize(HORIZONTAL, numSubArrayRow, numSubArrayCol, 0, numRow, subArray->height, subArray->width, param->clkFreq);
	busOutput->Initialize(VERTICAL, numSubArrayRow, numSubArrayCol, 0, numCol, subArray->height, subArray->width, param->clkFreq);
	predecoder->Initialize(REGULAR_ROW, (int)ceil(log2(numSubArrayRow*numSubArrayCol)), true, false);

	dff->Initialize(1, param->clkFreq);
}


double ProcessingUnitCalculateArea(SubArray *subArray, int numSubArrayRow, int numSubArrayCol) {
	double height = 0;
	double width = 0;
	double area = 0;
	
	dff->CalculateArea(subArray->height, NULL, NONE);

	subArray->CalculateArea();
	busInput->CalculateArea(1, true); 
	busOutput->CalculateArea(1, true);	
	predecoder->CalculateArea(subArray->height, NULL, NONE);
	area += subArray->usedArea * (numSubArrayRow*numSubArrayCol);

	height = sqrt(area);
	width = area/(height);

	return area;
}


double ProcessingUnitCalculatePerformance(SubArray *subArray, int numSubArrayRow, int numSubArrayCol, MemCell& cell, 
											double *readLatency, double *readDynamicEnergy, double *writeLatency, double *writeDynamicEnergy, double *leakage, 
											double *refreshDynamicEnergy, double *refreshLatency) {
	
	busInput->CalculatePower(busInput->busWidth, 512/(busInput->busWidth));
	busOutput->CalculatePower(busOutput->busWidth, 512/(busOutput->busWidth));
	predecoder->CalculatePower(1,1);

	subArray->CalculateLatency(1e20);
	subArray->CalculatePower();
	*readLatency = subArray->readLatency;		
	*readDynamicEnergy = subArray->readDynamicEnergy + busInput->readDynamicEnergy + busOutput->readDynamicEnergy + predecoder->readDynamicEnergy; // *numSubArrayRow*numSubArrayCol;
	*writeLatency = subArray->writeLatency;		
	*writeDynamicEnergy = subArray->writeDynamicEnergy + busInput->readDynamicEnergy + busOutput->readDynamicEnergy + predecoder->readDynamicEnergy; // *numSubArrayRow*numSubArrayCol;
	*leakage = subArray->leakage*numSubArrayRow*numSubArrayCol + busInput->leakage + busOutput->leakage + predecoder->leakage;
	*refreshDynamicEnergy = subArray->refreshDynamicEnergy + busInput->readDynamicEnergy + busOutput->readDynamicEnergy + predecoder->readDynamicEnergy*param->numRowSubArray;
	*refreshLatency = subArray->refreshLatency;
		
	return 0;
}


