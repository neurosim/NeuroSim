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
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Tile.h"

using namespace std;

extern Param *param;
// Anni update
int numInBufferCoreCM = 0;
int numOutBufferCoreCM = 0;	
int numInBufferCoreNM = 0;
int numOutBufferCoreNM = 0;									 

SubArray *subArrayInPE;
Buffer *inputBufferCM;
Buffer *outputBufferCM;
HTree *hTreeCM;
AdderTree *accumulationCM;
Sigmoid *sigmoidCM;
BitShifter *reLuCM;
Buffer *inputBufferNM;
Buffer *outputBufferNM;
HTree *hTreeNM;
AdderTree *accumulationNM;
Sigmoid *sigmoidNM;
BitShifter *reLuNM;

void TileInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, double _numPENM, double _peSizeNM, double _numPECM, double _peSizeCM){
	
	subArrayInPE = new SubArray(inputParameter, tech, cell);
	inputBufferNM = new Buffer(inputParameter, tech, cell);
	outputBufferNM = new Buffer(inputParameter, tech, cell);
	hTreeNM = new HTree(inputParameter, tech, cell);
	accumulationNM = new AdderTree(inputParameter, tech, cell);
	inputBufferCM = new Buffer(inputParameter, tech, cell);
	outputBufferCM = new Buffer(inputParameter, tech, cell);
	hTreeCM = new HTree(inputParameter, tech, cell);
	accumulationCM = new AdderTree(inputParameter, tech, cell);
	// Anni update
	reLuNM = new BitShifter(inputParameter, tech, cell);
	reLuCM = new BitShifter(inputParameter, tech, cell);
	sigmoidNM = new Sigmoid(inputParameter, tech, cell);
	sigmoidCM = new Sigmoid(inputParameter, tech, cell);

	
	/*** Parameters ***/
	double numPENM, peSizeNM, numPECM, peSizeCM, numSubArrayNM, numSubArrayCM;
	int numRowPerSynapse, numColPerSynapse;
	
	numPECM = _numPECM;
	peSizeCM = _peSizeCM;
	numPENM = _numPENM;
	peSizeNM = _peSizeNM;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	/*** Initialize ProcessingUnit ***/
	numSubArrayNM = ceil((double)peSizeNM/(double)param->numRowSubArray)*ceil((double)peSizeNM/(double)param->numColSubArray);
	numSubArrayCM = ceil((double)peSizeCM/(double)param->numRowSubArray)*ceil((double)peSizeCM/(double)param->numColSubArray);

	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, ceil(sqrt(numSubArrayNM)), ceil(sqrt(numSubArrayNM)), ceil(sqrt(numSubArrayCM)), ceil(sqrt(numSubArrayCM)));

	// Anni update: numBitPEOutput
	int numBitSubarrayOutput, numBitPEOutputCM, numBitPEOutputNM;
	if (param->parallelRead) {		
		numBitSubarrayOutput = log2((double)param->levelOutput)+ceil(log2(ceil(param->numRowSubArray/param->numRowParallel)))+param->numBitInput+(param->numColPerSynapse-1)*param->cellBit+1;
	} else{
		numBitSubarrayOutput = ceil(log2((double)param->numRowSubArray))+param->cellBit+param->numBitInput+(param->numColPerSynapse-1)*param->cellBit+1;
	}

	if (param->novelMapping) {
		// Anni update: numBitPEOutputNM, numOutBufferCoreNM, numInBufferCoreNM
		numBitPEOutputNM = numBitSubarrayOutput + ceil(sqrt(numSubArrayNM));	
        // 230920 update
        accumulationNM->Initialize(numPENM, numBitPEOutputNM, ceil((double)peSizeNM/(double)param->numColMuxed), param->clkFreq);
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuNM->Initialize(ceil((double)peSizeNM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				// Anni updare: 1.4 update
				sigmoidNM->Initialize(param->Activationtype, param->numBitInput, numBitPEOutputNM+ceil(log2((double)numPENM)), ceil((double)numPENM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
			}
			numOutBufferCoreNM = ceil((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
			
			if ((param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
				outputBufferNM->Initialize(param->numBitInput*numPENM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPENM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			} else {
				outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
			}									
		} else {
			numOutBufferCoreNM = ceil(((numBitPEOutputNM+ceil(log2((double)numPENM)))*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));

            // 230920 update
            if (param->sync_data_transfer) {
				numOutBufferCoreNM = ceil(((numBitPEOutputNM+ceil(log2((double)numPENM)))*numPENM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
                outputBufferNM->Initialize((numBitPEOutputNM+ceil(log2((double)numPENM)))*(double)peSizeNM/param->numColMuxed, (numBitPEOutputNM+ceil(log2((double)numPENM)))*(double)peSizeNM/param->numColMuxed, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
            }

            else {
                if (((numBitPEOutputNM+ceil(log2((double)numPENM)))*numPENM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
                    outputBufferNM->Initialize((numBitPEOutputNM+ceil(log2((double)numPENM)))*numPENM*param->numColSubArray/param->numColMuxed, (numBitPEOutputNM+ceil(log2((double)numPENM)))*numPENM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
                } else {
                    outputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
                }
            }
		}

		numInBufferCoreNM = ceil((numPENM*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));

        // 230920 update
        if (param->sync_data_transfer) {
			 numInBufferCoreNM = ceil((numPENM*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
            inputBufferNM->Initialize(numPENM*(double)peSizeNM*param->numBitInput, numPENM*(double)peSizeNM*param->numBitInput, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
        }
        else {
            if ((numPENM*param->numBitInput*param->numRowSubArray) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
                inputBufferNM->Initialize(numPENM*param->numBitInput*param->numRowSubArray, numPENM*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
            } else {
                inputBufferNM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
            }
        }

		// 230920 update
		 hTreeNM->Initialize(ceil(sqrt((double)numPENM)), ceil(sqrt((double)numPENM)), param->localBusDelayTolerance, numPENM*(double)peSizeNM, param->clkFreq);
	} 

	// Anni update: numBitPEOutputCM, numOutBufferCoreCM, numInBufferCoreCM
	numBitPEOutputCM = numBitSubarrayOutput + ceil(sqrt(numSubArrayCM));	
    // 230920 update
    accumulationCM->Initialize(numPECM, numBitPEOutputCM, ceil((double)numPECM*(double)peSizeCM/(double)param->numColMuxed), param->clkFreq);
	if (!param->chipActivation) {
		if (param->reLu) {
			reLuCM->Initialize(ceil((double)peSizeCM*(double)param->numColSubArray/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
		} else {
			// 1.4 update
			sigmoidCM->Initialize(param->Activationtype, param->numBitInput, numBitPEOutputCM+ceil(log2((double)numPECM)), ceil((double)numPECM*(double)param->numColSubArray/(double)param->numColMuxed), param->clkFreq);
		}
		numOutBufferCoreCM = ceil((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
		
		if ((param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
			outputBufferCM->Initialize(param->numBitInput*numPECM*param->numColSubArray/param->numColMuxed, param->numBitInput*numPECM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		} else {
			outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
		}									
	} else {
		numOutBufferCoreCM = ceil(((numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*param->numColSubArray/param->numColMuxed)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
		
        // 230920 update
        if (param->sync_data_transfer) {
			numOutBufferCoreCM = ceil((numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*(double)peSizeCM/param->numColMuxed/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
            outputBufferCM->Initialize((numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*(double)peSizeCM/param->numColMuxed, (numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*(double)peSizeCM/param->numColMuxed, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
        }

        else {
            if (((numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*param->numColSubArray/param->numColMuxed) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
                outputBufferCM->Initialize((numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM*param->numColSubArray/param->numColMuxed, (numBitPEOutputCM+ceil(log2((double)numPECM)))*numPECM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
            } else {
                outputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
            }
        }		
		
	}

	numInBufferCoreCM = ceil((numPECM*param->numBitInput*param->numRowSubArray)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
	
    // 230920 update
    if (param->sync_data_transfer) {
		numInBufferCoreCM = ceil((numPECM*param->numBitInput*(double)peSizeCM)/(param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol));
        inputBufferCM->Initialize(numPECM*param->numBitInput*(double)peSizeCM, numPECM*param->numBitInput*(double)peSizeCM, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
    }

    else {
        if ((numPECM*param->numBitInput*param->numRowSubArray) < (param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol)) {
            inputBufferCM->Initialize(numPECM*param->numBitInput*param->numRowSubArray, numPECM*param->numRowSubArray, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
        } else {
            inputBufferCM->Initialize((param->tileBufferCoreSizeRow*param->tileBufferCoreSizeCol), param->tileBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->peBufferType);
        }
    }    

	 hTreeCM->Initialize(numPECM, numPECM, param->localBusDelayTolerance, numPECM*(double)peSizeCM, param->clkFreq);

}

vector<double> TileCalculateArea(double numPE, double peSize, bool NMTile, double *height, double *width) {
	double area = 0;
	double PEheight, PEwidth, PEbufferArea;
	*height = 0;
	*width = 0;
	vector<double> areaResults;
	vector<double> peAreaResults;
	double areareLu = 0;
	double areasigmoid = 0;
	
	if (NMTile) {
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), true, &PEheight, &PEwidth, &PEbufferArea);
		double PEarea = peAreaResults[0];
		double PEareaADC = peAreaResults[1];
		double PEareaAccum = peAreaResults[2];
		double PEareaOther = peAreaResults[3];
		double PEareaArray = peAreaResults[4];
		accumulationNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += reLuNM->area;
				areareLu += reLuNM->area;
			} else {
				sigmoidNM->CalculateUnitArea(NONE);
				sigmoidNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += sigmoidNM->area;
				areasigmoid += sigmoidNM->area;
			}
		}
		inputBufferNM->CalculateArea(ceil(sqrt((double)numPE))*PEheight, NULL, NONE);
		outputBufferNM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		// Anni update
		inputBufferNM->area *= numInBufferCoreNM;
		outputBufferNM->area *= numOutBufferCoreNM;												  
		hTreeNM->CalculateArea(PEheight, PEwidth, 16);

		area += PEarea*numPE + accumulationNM->area + inputBufferNM->area + outputBufferNM->area + hTreeNM->area;
		
		*height = sqrt(area);
		*width = area/(*height);
		
		areaResults.push_back(area);
		areaResults.push_back(hTreeNM->area);
		areaResults.push_back(PEareaADC*numPE);
		areaResults.push_back(PEareaAccum*numPE + accumulationNM->area);
		areaResults.push_back(PEareaOther*numPE + inputBufferNM->area + outputBufferNM->area + areareLu + areasigmoid);
		areaResults.push_back(PEareaArray*numPE);
	} else {
		int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
		peAreaResults = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), false, &PEheight, &PEwidth, &PEbufferArea);
		double PEarea = peAreaResults[0];
		double PEareaADC = peAreaResults[1];
		double PEareaAccum = peAreaResults[2];
		double PEareaOther = peAreaResults[3];
		double PEareaArray = peAreaResults[4];
		accumulationCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		if (!param->chipActivation) {
			if (param->reLu) {
				reLuCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += reLuCM->area;
				areareLu += reLuCM->area;
			} else {
				sigmoidCM->CalculateUnitArea(NONE);
				sigmoidCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
				area += sigmoidCM->area;
				areasigmoid += sigmoidCM->area;
			}
		}
		inputBufferCM->CalculateArea(ceil(sqrt((double)numPE))*PEheight, NULL, NONE);
		outputBufferCM->CalculateArea(NULL, ceil(sqrt((double)numPE))*PEwidth, NONE);
		// Anni update
		inputBufferCM->area *= numInBufferCoreCM;
		outputBufferCM->area *= numOutBufferCoreCM;												  
		hTreeCM->CalculateArea(PEheight, PEwidth, 16);
	
		area += PEarea*numPE + accumulationCM->area + inputBufferCM->area + outputBufferCM->area + hTreeCM->area;
		
		*height = sqrt(area);
		*width = area/(*height);
		
		areaResults.push_back(area);
		areaResults.push_back(hTreeCM->area);
		areaResults.push_back(PEareaADC*numPE);
		areaResults.push_back(PEareaAccum*numPE + accumulationCM->area);
		areaResults.push_back(PEareaOther*numPE + inputBufferCM->area + outputBufferCM->area + areareLu + areasigmoid);
		areaResults.push_back(PEareaArray*numPE);
	}
	
	return areaResults;
}

// Anni update: add leakageSRAMInUse
void TileCalculatePerformance(const vector<vector<double> > &newMemory, const vector<vector<double> > &oldMemory, const vector<vector<double> > &inputVector, int novelMap, double numPE, 
							double peSize, int speedUpRow, int speedUpCol, int weightMatrixRow, int weightMatrixCol, int numInVector, MemCell& cell, double *readLatency, double *readDynamicEnergy, double *leakage, double *leakageSRAMInUse,
							double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy,
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther, bool CalculateclkFreq, double*clkPeriod) {

	/*** sweep PE ***/
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	// Anni update: add PEleakageSRAMInUse
	double PEreadLatency, PEreadDynamicEnergy, PEleakage, PEleakageSRAMInUse, PEbufferLatency, PEbufferDynamicEnergy, PEicLatency, PEicDynamicEnergy;
	double peLatencyADC, peLatencyAccum, peLatencyOther, peEnergyADC, peEnergyAccum, peEnergyOther;
	int numSubArrayRow = ceil((double)peSize/(double)param->numRowSubArray);
	int numSubArrayCol = ceil((double)peSize/(double)param->numColSubArray);
	
	*readLatency = 0;
	*readDynamicEnergy = 0;
	*leakage = 0;
	// Anni update
	*leakageSRAMInUse = 0;
	*bufferLatency = 0;
	*bufferDynamicEnergy = 0;
	*icLatency = 0;
	*icDynamicEnergy = 0;
	*coreEnergyADC = 0;
	*coreEnergyAccum = 0;
	*coreEnergyOther = 0;
	*coreLatencyADC = 0;
	*coreLatencyAccum = 0;
	*coreLatencyOther = 0;

	// Anni update: update Clock frequency
	if(!CalculateclkFreq) {	
		inputBufferCM->clkFreq = param->clkFreq; 
		outputBufferCM->clkFreq = param->clkFreq; 
		hTreeCM->clkFreq = param->clkFreq; 
		accumulationCM->clkFreq = param->clkFreq; 
		reLuCM->clkFreq = param->clkFreq; 
		sigmoidCM->clkFreq = param->clkFreq; 
		
		inputBufferNM->clkFreq = param->clkFreq; 
		outputBufferNM->clkFreq = param->clkFreq; 
		hTreeNM->clkFreq = param->clkFreq; 
		accumulationNM->clkFreq = param->clkFreq; 	
		reLuNM->clkFreq = param->clkFreq; 
		sigmoidNM->clkFreq = param->clkFreq; 	
	}

	if (!novelMap) {   // conventional Mapping
		if (speedUpRow*speedUpCol > 1) {
			if ((speedUpRow >= numPE) && (speedUpCol >= numPE)) {

				
				// duplication in PE or subArray --> tell each PE to take the whole assigned weight  --> "fully" duplication
				// assign weight and input to specific tile
				vector<vector<double> > pEMemory;
				pEMemory = CopyPEArray(newMemory, 0, 0, weightMatrixRow, weightMatrixCol);
				vector<vector<double> > pEInput;
				pEInput = CopyPEInput(inputVector, 0, numInVector, weightMatrixRow);
				// Anni update

				ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, ceil((double)speedUpRow/(double)numPE), ceil((double)speedUpCol/(double)numPE), 
											numSubArrayRow, numSubArrayCol, weightMatrixRow, weightMatrixCol, numInVector, cell, false,
											&PEreadLatency, &PEreadDynamicEnergy, &PEleakage, &PEleakageSRAMInUse,
											&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
											&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, CalculateclkFreq, clkPeriod);

				*readLatency = PEreadLatency/(numPE*numPE);  // further speed up in PE level
				*readDynamicEnergy = PEreadDynamicEnergy;   // since subArray.cpp takes all input vectors, no need to *numPE here

				*bufferLatency = PEbufferLatency/(numPE*numPE);		//#cycles of PE-level buffers (DFF)
				*bufferDynamicEnergy = PEbufferDynamicEnergy;
				*icLatency = PEicLatency/(numPE*numPE);				//s
				*icDynamicEnergy = PEicDynamicEnergy;
				
				*coreLatencyADC = peLatencyADC/(numPE*numPE);		//#sensing cycles
				*coreLatencyAccum = peLatencyAccum/(numPE*numPE);	//#cycles
				*coreLatencyOther = peLatencyOther/(numPE*numPE);
				
				*coreEnergyADC = peEnergyADC;
				*coreEnergyAccum = peEnergyAccum;
				*coreEnergyOther = peEnergyOther;
				// no accumulation access
			} else {
				// # duplication is smaller then # PE, means only a group of PE take the assigned weight  --> not "fully" duplication
				// also need to redefine a few data-grab start-point
				for (int i=0; i<ceil((double)weightMatrixRow/(double)peSize); i++) {
					for (int j=0; j<ceil((double)weightMatrixCol/(double)peSize); j++) {
						if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
							int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
							int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);
					
							// assign weight and input to specific tile
							vector<vector<double> > pEMemory;
							pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
							vector<vector<double> > pEInput;
							pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);
							
							// Anni update

							ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, 
												numSubArrayRow, numSubArrayCol, numRowMatrix, numColMatrix, numInVector, cell, false,
												&PEreadLatency, &PEreadDynamicEnergy, &PEleakage, &PEleakageSRAMInUse,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy,
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, CalculateclkFreq, clkPeriod);

							*readLatency = MAX(PEreadLatency, (*readLatency));
							*readDynamicEnergy += PEreadDynamicEnergy;
							*bufferLatency = MAX(PEbufferLatency, (*bufferLatency));
							*bufferDynamicEnergy += PEbufferDynamicEnergy;
							*icLatency = MAX(PEicLatency,(*icLatency));
							*icDynamicEnergy += PEicDynamicEnergy;
							
							*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
							*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
							*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
							
							*coreEnergyADC += peEnergyADC;
							*coreEnergyAccum += peEnergyAccum;
							*coreEnergyOther += peEnergyOther;
						}
					}
				}
				*readLatency /= (speedUpRow*speedUpCol);   // further speedup in PE level
				*coreLatencyADC /= (speedUpRow*speedUpCol);
				*coreLatencyAccum /= (speedUpRow*speedUpCol);
				*coreLatencyOther /= (speedUpRow*speedUpCol);
				*bufferLatency /= (speedUpRow*speedUpCol);
				*icLatency /= (speedUpRow*speedUpCol);
				
				// whether go through accumulation?
				if (ceil((double)weightMatrixRow/(double)peSize) > 1) {
					accumulationCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil((double)param->numColMuxed/(double)param->numColPerSynapse), ceil((double)weightMatrixRow/(double)peSize), 0);
					accumulationCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil((double)param->numColMuxed/(double)param->numColPerSynapse), ceil((double)weightMatrixRow/(double)peSize));

                    // 230920 update
                    if(!param->sync_data_transfer){
                    *readLatency += accumulationCM->readLatency; 
					*coreLatencyAccum += accumulationCM->readLatency; 
                    }

					*readDynamicEnergy += accumulationCM->readDynamicEnergy;					
					*coreEnergyAccum += accumulationCM->readDynamicEnergy;
				}
			}
			
		} else {
			// no duplication --> tell PE to further partition the weight and grab data (redefine a few data-grab start-point)
			for (int i=0; i<numPE; i++) {
				for (int j=0; j<numPE; j++) {
					// each cycle assign to different PE
					if ( (i*peSize < weightMatrixRow) && (j*peSize < weightMatrixCol) ) {
						// assign weight and input to specific tile
						int numRowMatrix = min(peSize, (double) weightMatrixRow-i*peSize);
						int numColMatrix = min(peSize, (double) weightMatrixCol-j*peSize);
						
						vector<vector<double> > pEMemory;
						pEMemory = CopyPEArray(newMemory, i*peSize, j*peSize, numRowMatrix, numColMatrix);
						vector<vector<double> > pEInput;
						pEInput = CopyPEInput(inputVector, i*peSize, numInVector, numRowMatrix);
						// Anni update
						
						// Anni update: PEleakageSRAMInUse
						ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, numRowMatrix,
												numColMatrix, numInVector, cell, false, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage, &PEleakageSRAMInUse,
												&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy, 
												&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, CalculateclkFreq, clkPeriod);

					}
					*readLatency = max(PEreadLatency, (*readLatency));
					*readDynamicEnergy += PEreadDynamicEnergy;
					
					*bufferLatency = max(PEbufferLatency, (*bufferLatency));
					*bufferDynamicEnergy += PEbufferDynamicEnergy;
					*icLatency = max(PEicLatency,(*icLatency));
					*icDynamicEnergy += PEicDynamicEnergy;
					
					*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
					*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
					*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
					
					*coreEnergyADC += peEnergyADC;
					*coreEnergyAccum += peEnergyAccum;
					*coreEnergyOther += peEnergyOther;
				}
			}
			accumulationCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil((double)param->numColMuxed/(double)param->numColPerSynapse), numPE, 0);
			accumulationCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil((double)param->numColMuxed/(double)param->numColPerSynapse), numPE);
           
		   // 230920 update
		    if(!param->sync_data_transfer){
            *readLatency += accumulationCM->readLatency;
            *coreLatencyAccum += accumulationCM->readLatency;
			}			
			
			*readDynamicEnergy += accumulationCM->readDynamicEnergy;			
			*coreEnergyAccum += accumulationCM->readDynamicEnergy;
		}
		if(!CalculateclkFreq){
			double numBitToLoadOut, numBitToLoadIn;											  
			if (!param->chipActivation) {
				if (param->reLu) {
					reLuCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuCM->numUnit);
					reLuCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuCM->numUnit);
					*readLatency += reLuCM->readLatency;
					*readDynamicEnergy += reLuCM->readDynamicEnergy;
					*coreLatencyOther += reLuCM->readLatency;
					*coreEnergyOther += reLuCM->readDynamicEnergy;
					numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLuCM->numBit)*numInVector/param->numBitInput, 0);
					// 230920 updated
					outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
					outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
				} else {
					sigmoidCM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidCM->numEntry);
					sigmoidCM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidCM->numEntry);
					*readLatency += sigmoidCM->readLatency;
					*readDynamicEnergy += sigmoidCM->readDynamicEnergy;
					*coreLatencyOther += sigmoidCM->readLatency;
					*coreEnergyOther += sigmoidCM->readDynamicEnergy;
					numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoidCM->numYbit)*numInVector/param->numBitInput, 0);
					// 230920 updated
					outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
					outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
				}
			} else {
				// Anni update: accumulationCM->numStage
				
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(accumulationCM->numStage+accumulationCM->numAdderBit)*numInVector/param->numBitInput, 0);
				// 230920 updated
				outputBufferCM->CalculateLatency(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
				outputBufferCM->CalculatePower(outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0, outputBufferCM->interface_width, numBitToLoadIn/outputBufferCM->interface_width/2.0);
			}
			
			//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
			numBitToLoadOut = MAX(weightMatrixRow*numInVector, 0);
			// 230920 updated
			inputBufferCM->CalculateLatency(inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width/2.0, inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width/2.0);
			inputBufferCM->CalculatePower(inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width/2.0, inputBufferCM->interface_width, numBitToLoadOut/inputBufferCM->interface_width/2.0);
			// since multi-core buffer has improve the parallelism
			// Anni update: numInBufferCoreCM, numOutBufferCoreCM
			// cout<<"numBitToLoadOut: "<<numBitToLoadOut<<"	numBitToLoadIn: "<<numBitToLoadIn<<endl;
			// cout<<"inputBufferCM->readLatency before: "<<inputBufferCM->readLatency<<"	outputBufferCM->readLatency: "<<outputBufferCM->readLatency<<endl;
			inputBufferCM->readLatency /= MIN(numInBufferCoreCM, ceil(hTreeCM->busWidth/inputBufferCM->interface_width));
			inputBufferCM->writeLatency /= MIN(numInBufferCoreCM, ceil(hTreeCM->busWidth/inputBufferCM->interface_width));
			outputBufferCM->readLatency /= MIN(numOutBufferCoreCM, ceil(hTreeCM->busWidth/outputBufferCM->interface_width));
			outputBufferCM->writeLatency /= MIN(numOutBufferCoreCM, ceil(hTreeCM->busWidth/outputBufferCM->interface_width));	
			// cout<<"numInBufferCoreCM: "<<numInBufferCoreCM<<"	ceil(hTreeCM->busWidth/inputBufferCM->interface_width): "<<ceil(hTreeCM->busWidth/inputBufferCM->interface_width)<<endl;
			// cout<<"numOutBufferCoreCM: "<<numOutBufferCoreCM<<"	ceil(hTreeCM->busWidth/outputBufferCM->interface_width): "<<ceil(hTreeCM->busWidth/outputBufferCM->interface_width)<<endl;	
			// cout<<"inputBufferCM->interface_width: "<<inputBufferCM->interface_width<<"	outputBufferCM->interface_width: "<<outputBufferCM->interface_width<<endl;
			// cout<<"inputBufferCM->readLatency after: "<<inputBufferCM->readLatency<<"	outputBufferCM->readLatency: "<<outputBufferCM->readLatency<<endl;																					   
			
			// 230920 update
            if(!param->sync_data_transfer){
			*readLatency += (inputBufferCM->readLatency + inputBufferCM->writeLatency);
			*readLatency += (outputBufferCM->readLatency + outputBufferCM->writeLatency);
			}

			*readDynamicEnergy += inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy;
			*readDynamicEnergy += outputBufferCM->readDynamicEnergy + outputBufferCM->writeDynamicEnergy;
			// used to define travel distance
			double PEheight, PEwidth, PEbufferArea;
			int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
			vector<double> PEarea;
			PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), false, &PEheight, &PEwidth, &PEbufferArea);
			hTreeCM->CalculateLatency(NULL, NULL, NULL, NULL, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTreeCM->busWidth);
			hTreeCM->CalculatePower(NULL, NULL, NULL, NULL, PEheight, PEwidth, hTreeCM->busWidth, (numBitToLoadOut * param->inputtoggle +numBitToLoadIn * param->outputtoggle )/hTreeCM->busWidth);	  
			
            // 230920 update
            if(!param->sync_data_transfer){   			
				*readLatency += hTreeCM->readLatency;
				*bufferLatency += (inputBufferCM->readLatency + outputBufferCM->readLatency + inputBufferCM->writeLatency + outputBufferCM->writeLatency);
				*icLatency += hTreeCM->readLatency;
				*coreLatencyOther += (inputBufferCM->readLatency + inputBufferCM->writeLatency + outputBufferCM->readLatency + outputBufferCM->writeLatency + hTreeCM->readLatency);
			}

			*readDynamicEnergy += hTreeCM->readDynamicEnergy;
			*bufferDynamicEnergy += inputBufferCM->readDynamicEnergy + outputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy + outputBufferCM->writeDynamicEnergy;
			*icDynamicEnergy += hTreeCM->readDynamicEnergy;
			*coreEnergyOther += inputBufferCM->readDynamicEnergy + inputBufferCM->writeDynamicEnergy + outputBufferCM->readDynamicEnergy + outputBufferCM->writeDynamicEnergy + hTreeCM->readDynamicEnergy;
			
			// 1.4 update : leakage energy of IC 
			*leakage = PEleakage*numPE*numPE + accumulationCM->leakage + inputBufferCM->leakage + outputBufferCM->leakage + hTreeCM->leakage;
			// Anni update
			*leakageSRAMInUse = PEleakageSRAMInUse*numPE*numPE;
		}
	} else {  // novel Mapping
		for (int i=0; i<numPE; i++) {
			int location = i*MIN(peSize, (int) weightMatrixRow/numPE);
			vector<vector<double> > pEMemory;
			pEMemory = CopyPEArray(newMemory, location, 0, weightMatrixRow/numPE, weightMatrixCol);
			vector<vector<double> > pEInput;
			pEInput = CopyPEInput(inputVector, location, numInVector, weightMatrixRow/numPE);
			// Anni update: PEleakageSRAMInUse
			ProcessingUnitCalculatePerformance(subArrayInPE, pEMemory, pEMemory, pEInput, 1, 1, numSubArrayRow, numSubArrayCol, weightMatrixRow/numPE,
									weightMatrixCol, numInVector, cell, true, &PEreadLatency, &PEreadDynamicEnergy, &PEleakage, &PEleakageSRAMInUse,
									&PEbufferLatency, &PEbufferDynamicEnergy, &PEicLatency, &PEicDynamicEnergy, 
									&peLatencyADC, &peLatencyAccum, &peLatencyOther, &peEnergyADC, &peEnergyAccum, &peEnergyOther, CalculateclkFreq, clkPeriod);

			*readLatency = max(PEreadLatency, (*readLatency));
			*readDynamicEnergy += PEreadDynamicEnergy;
			*bufferLatency = max(PEbufferLatency, (*bufferLatency));
			*bufferDynamicEnergy += PEbufferDynamicEnergy;
			*icLatency = max(PEicLatency,(*icLatency));
			*icDynamicEnergy += PEicDynamicEnergy;
			
			*coreLatencyADC = MAX(peLatencyADC, (*coreLatencyADC));
			*coreLatencyAccum = MAX(peLatencyAccum, (*coreLatencyAccum));
			*coreLatencyOther = MAX(peLatencyOther, (*coreLatencyOther));
			
			*coreEnergyADC += peEnergyADC;
			*coreEnergyAccum += peEnergyAccum;
			*coreEnergyOther += peEnergyOther;
		}
		if(!CalculateclkFreq){
			*readLatency /= (speedUpRow*speedUpCol);
			*coreLatencyADC /= (speedUpRow*speedUpCol);
			*coreLatencyAccum /= (speedUpRow*speedUpCol);
			*coreLatencyOther /= (speedUpRow*speedUpCol);
			*bufferLatency /= (speedUpRow*speedUpCol);
			*icLatency /= (speedUpRow*speedUpCol);
			
			accumulationNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE, 0);
			accumulationNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse), numPE);
			
			// 230920 update
			if(!param->sync_data_transfer){ 
			*readLatency += accumulationNM->readLatency;
			*coreLatencyAccum += accumulationNM->readLatency;
			}

			*readDynamicEnergy += accumulationNM->readDynamicEnergy;
			
			
			*coreEnergyAccum += accumulationNM->readDynamicEnergy;
			
			//considering buffer activation: no matter speedup or not, the total number of data transferred is fixed
			double numBitToLoadOut, numBitToLoadIn;
			numBitToLoadOut= MAX(weightMatrixRow*numInVector/sqrt(numPE), 0);
			// 230920 updated
			inputBufferNM->CalculateLatency(inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width/2.0, inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width/2.0);
			inputBufferNM->CalculatePower(inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width/2.0, inputBufferNM->interface_width, numBitToLoadOut/inputBufferNM->interface_width/2.0);
		
			if (!param->chipActivation) {
				if (param->reLu) {
					reLuNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuNM->numUnit);
					reLuNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/reLuNM->numUnit);
					*readLatency += reLuNM->readLatency;
					*readDynamicEnergy += reLuNM->readDynamicEnergy;
					*coreLatencyOther += reLuNM->readLatency;
					*coreEnergyOther += reLuNM->readDynamicEnergy;
					
					numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+reLuNM->numBit)*numInVector/param->numBitInput/numPE, 0);
					// 230920 updated
					outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
					outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
				} else {
					sigmoidNM->CalculateLatency((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidNM->numEntry);
					sigmoidNM->CalculatePower((int)(numInVector/param->numBitInput)*ceil(param->numColMuxed/param->numColPerSynapse)/sigmoidNM->numEntry);
					*readLatency += sigmoidNM->readLatency;
					*readDynamicEnergy += sigmoidNM->readDynamicEnergy;
					*coreLatencyOther += sigmoidNM->readLatency;
					*coreEnergyOther += sigmoidNM->readDynamicEnergy;
					
					numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(1+sigmoidNM->numYbit)*numInVector/param->numBitInput/numPE, 0);
					// 230920 updated
					outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
					outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
				}
			} else {
				// Anni update: accumulationNM->numStage,  not /numPE (only weightMatrixRow multiplied by numPE in chip.cpp)
				numBitToLoadIn = MAX(ceil(weightMatrixCol/param->numColPerSynapse)*(accumulationNM->numStage+accumulationNM->numAdderBit)*numInVector/param->numBitInput, 0);
				// 230920 updated
				outputBufferNM->CalculateLatency(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
				outputBufferNM->CalculatePower(outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0, outputBufferNM->interface_width, numBitToLoadIn/outputBufferNM->interface_width/2.0);
			}
			// since multi-core buffer has improve the parallelism
			// cout<<"numBitToLoadOut: "<<numBitToLoadOut<<"	numBitToLoadIn: "<<numBitToLoadIn<<endl;
			// cout<<"inputBufferNM->readLatency before: "<<inputBufferNM->readLatency<<"	outputBufferNM->readLatency: "<<outputBufferNM->readLatency<<endl;
			// Anni update: numInBufferCoreNM, numOutBufferCoreNM
			inputBufferNM->readLatency /= MIN(numInBufferCoreNM, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
			inputBufferNM->writeLatency /= MIN(numInBufferCoreNM, ceil(hTreeNM->busWidth/inputBufferNM->interface_width));
			// Anni update: corrct typo outputBufferNM->interface_width
			outputBufferNM->readLatency /= MIN(numOutBufferCoreNM, ceil(hTreeNM->busWidth/outputBufferNM->interface_width));
			outputBufferNM->writeLatency /= MIN(numOutBufferCoreNM, ceil(hTreeNM->busWidth/outputBufferNM->interface_width));
			// cout<<"numInBufferCoreNM: "<<numInBufferCoreNM<<"	ceil(hTreeNM->busWidth/inputBufferNM->interface_width): "<<ceil(hTreeNM->busWidth/inputBufferNM->interface_width)<<endl;
			// cout<<"numOutBufferCoreNM: "<<numOutBufferCoreNM<<"	ceil(hTreeNM->busWidth/outputBufferNM->interface_width): "<<ceil(hTreeNM->busWidth/outputBufferNM->interface_width)<<endl;
			// cout<<"inputBufferNM->interface_width: "<<inputBufferNM->interface_width<<"	outputBufferNM->interface_width: "<<outputBufferNM->interface_width<<endl;
			// cout<<"inputBufferNM->readLatency after: "<<inputBufferNM->readLatency<<"	outputBufferNM->readLatency: "<<outputBufferNM->readLatency<<endl;

	        // 230920 update
            if(!param->sync_data_transfer){  		
				*readLatency += inputBufferNM->readLatency + inputBufferNM->writeLatency;
				*readLatency += (outputBufferNM->readLatency + outputBufferNM->writeLatency);
			}
			
			*readDynamicEnergy += inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy;
			*readDynamicEnergy += outputBufferNM->readDynamicEnergy + outputBufferNM->writeDynamicEnergy;
			
			// used to define travel distance
			double PEheight, PEwidth, PEbufferArea;
			int numSubArray = ceil((double) peSize/(double) param->numRowSubArray)*ceil((double) peSize/(double) param->numColSubArray);
			vector<double> PEarea;
			PEarea = ProcessingUnitCalculateArea(subArrayInPE, ceil((double)sqrt((double)numSubArray)), ceil((double)sqrt((double)numSubArray)), true, &PEheight, &PEwidth, &PEbufferArea);
			hTreeNM->CalculateLatency(0, 0, 1, 1, PEheight, PEwidth, (numBitToLoadOut+numBitToLoadIn)/hTreeNM->busWidth);
			hTreeNM->CalculatePower(0, 0, 1, 1, PEheight, PEwidth, hTreeNM->busWidth, (numBitToLoadOut * param->inputtoggle +numBitToLoadIn * param->outputtoggle )/hTreeNM->busWidth);


            // 230920 update
            if(!param->sync_data_transfer){  
				*readLatency += hTreeNM->readLatency;					
				*bufferLatency += (inputBufferNM->readLatency + outputBufferNM->readLatency + inputBufferNM->writeLatency + outputBufferNM->writeLatency);
				// cout<<"tile: inputBufferNM->readLatency: "<<inputBufferNM->readLatency<<"	outputBufferNM->readLatency: "<<outputBufferNM->readLatency<<"	inputBufferNM->writeLatency: "<<inputBufferNM->writeLatency<<"	outputBufferNM->writeLatency: "<<outputBufferNM->writeLatency<<endl;
				*icLatency += hTreeNM->readLatency;
				*coreLatencyOther += (inputBufferNM->readLatency + inputBufferNM->writeLatency + outputBufferNM->readLatency + outputBufferNM->writeLatency + hTreeNM->readLatency);
			}

			*readDynamicEnergy += hTreeNM->readDynamicEnergy;
			*bufferDynamicEnergy += inputBufferNM->readDynamicEnergy + outputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy + outputBufferNM->writeDynamicEnergy;
			*icDynamicEnergy += hTreeNM->readDynamicEnergy;
			
			
			*coreEnergyOther += inputBufferNM->readDynamicEnergy + inputBufferNM->writeDynamicEnergy + outputBufferNM->readDynamicEnergy + outputBufferNM->writeDynamicEnergy + hTreeNM->readDynamicEnergy;
			
			// 1.4 update: leakage energy of IC
			*leakage = PEleakage*numPE + accumulationNM->leakage + inputBufferNM->leakage + outputBufferNM->leakage +hTreeNM->leakage;
			// Anni update
			*leakageSRAMInUse = PEleakageSRAMInUse*numPE;
		}
	}
}


vector<vector<double> > CopyPEArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numCol; j++) {
			copyRow.push_back(orginal[positionRow+i][positionCol+j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
} 


vector<vector<double> > CopyPEInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
	vector<vector<double> > copy;
	for (int i=0; i<numRow; i++) {
		vector<double> copyRow;
		for (int j=0; j<numInputVector; j++) {
			copyRow.push_back(orginal[positionRow+i][j]);
		}
		copy.push_back(copyRow);
		copyRow.clear();
	}
	return copy;
	copy.clear();
}

