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
#include "MaxPooling.h"
#include "Sigmoid.h"
#include "BitShifter.h"
#include "AdderTree.h"
#include "Buffer.h"
#include "HTree.h"
#include "ProcessingUnit.h"
#include "Tile.h"
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Chip.h"
// Anni update
#include "XYBus.h"			  

using namespace std;

extern Param *param;
double globalBusWidth = 0;
int numBufferCore = 0;				  

/*** Circuit Modules ***/
Buffer *globalBuffer;
HTree *GhTree;
// Anni update
XYBus *GBus;							  
AdderTree *Gaccumulation;
Sigmoid *Gsigmoid;
BitShifter *GreLu;
MaxPooling *maxPool;
 

vector<int> ChipDesignInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, bool pip, const vector<vector<double> > &netStructure,
					double *maxPESizeNM, double *maxTileSizeCM, double *numPENM){

	globalBuffer = new Buffer(inputParameter, tech, cell);
	GhTree = new HTree(inputParameter, tech, cell);
	// Anni update
	GBus = new XYBus(inputParameter, tech, cell);												
	Gaccumulation = new AdderTree(inputParameter, tech, cell);
	Gsigmoid = new Sigmoid(inputParameter, tech, cell);
	GreLu = new BitShifter(inputParameter, tech, cell);
	maxPool = new MaxPooling(inputParameter, tech, cell);

	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	double numLayer, minCube;
	
	// get information of network structure
	numLayer = netStructure.size();
	*maxPESizeNM = 0;
	*maxTileSizeCM = 0;
	*numPENM = 0;

	vector<int> markNM;
	if (param->novelMapping) {
		// define number of PE in COV layers
		int most = 0;
		int numPE = 0;
		for (int i=0; i<numLayer; i++) {
			int temp = netStructure[i][3]*netStructure[i][4];
			int count = 1;
			for (int j=0; j<numLayer; j++) {
				if (temp == netStructure[j][3]*netStructure[j][4] && temp!=1) {
					count ++;
				}
				if (most < count) {
					most = count;
					numPE = temp;
				}
			}
		}
		*numPENM = numPE;

		// mark the layers that use novel mapping
		for (int i=0; i<numLayer; i++) {

			if ((netStructure[i][3]*netStructure[i][4]== (*numPENM))
				// large Cov layers use novel mapping
				&&(netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse >= param->numRowSubArray)) {
				markNM.push_back(1);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				*maxPESizeNM = max(minCube, (*maxPESizeNM));
			} else {
				// small Cov layers and FC layers use conventional mapping
				markNM.push_back(0);
				minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
				*maxTileSizeCM = max(minCube, (*maxTileSizeCM));
			}
		}
		


	} else {
		// all layers use conventional mapping
		for (int i=0; i<numLayer; i++) {
			markNM.push_back(0);
			minCube = pow(2, ceil((double) log2((double) netStructure[i][5]*(double) numColPerSynapse) ) );
			*maxTileSizeCM = max(minCube, (*maxTileSizeCM));
		}
	}
	
	// for pipeline system
	vector<int> pipelineSpeedUp;
	if (param->pipeline) {
		// find max and min IFM size --> define how much the system can be speed-up
		int maxIFMSize = netStructure[0][0];
		int minIFMSize = maxIFMSize;
		for (int i=0; i<numLayer; i++) {
			if (netStructure[i][3] != 1) {
				int thisIFMSize = netStructure[i][0];
				if (thisIFMSize < minIFMSize) {
					minIFMSize = thisIFMSize;
				}
			}
		}
		
		// justify the speed-up degree is necessary
		int maxSpeedUpDegree = int(maxIFMSize/minIFMSize);
		if (maxSpeedUpDegree < param->speedUpDegree) {
			cout << "User assigned speed-up degree is larger than the upper bound (where no idle period during the whole process) " << endl;
			param->speedUpDegree = maxSpeedUpDegree;
			cout << "The speed-up degree is auto-assigned as the upper bound (where no idle period during the whole process) " << endl;
		}
		// define the pipeline speed-up
		int boundIFMSize = ceil((double) maxIFMSize/(param->speedUpDegree));
		for (int i=0; i<numLayer; i++) {
			int speedUp = ceil((double) pow((netStructure[i][0]/boundIFMSize), 2));
			pipelineSpeedUp.push_back(speedUp);
		}
	}
	
	if (pip) {
		return pipelineSpeedUp;
	} else {
		return markNM;
	}
}


vector<vector<double> > ChipFloorPlan(bool findNumTile, bool findUtilization, bool findSpeedUp, const vector<vector<double> > &netStructure, const vector<int > &markNM, 
					double maxPESizeNM, double maxTileSizeCM, double numPENM, const vector<int> &pipelineSpeedUp,
					double *desiredNumTileNM, double *desiredPESizeNM, double *desiredNumTileCM, double *desiredTileSizeCM, double *desiredPESizeCM, int *numTileRow, int *numTileCol) {
	
	
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	double maxUtilizationNM = 0;
	double maxUtilizationCM = 0;
	
	vector<vector<double> > peDup;
	vector<vector<double> > subArrayDup;
	vector<vector<double> > numTileEachLayer;
	vector<vector<double> > utilizationEachLayer;
	vector<vector<double> > speedUpEachLayer;
	
	*desiredNumTileNM = 0;
	*desiredPESizeNM = 0;
	*desiredNumTileCM = 0;
	*desiredTileSizeCM = 0;
	*desiredPESizeCM = 0;
	*numTileRow = 0;
	*numTileCol = 0;

	if (param->novelMapping) {		// Novel Mapping
		if (maxPESizeNM < 2*param->numRowSubArray) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		}else{
		
			/*** Tile Design ***/
			*desiredPESizeNM = MAX(maxPESizeNM, 2*param->numRowSubArray);
			vector<double> initialDesignNM;
			initialDesignNM = TileDesignNM((*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			*desiredNumTileNM = initialDesignNM[0];

			for (double thisPESize = MAX(maxPESizeNM, 2*param->numRowSubArray); thisPESize> 2*param->numRowSubArray; thisPESize/=2) {
				// for layers use novel mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignNM(thisPESize, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationNM) {
					maxUtilizationNM = thisUtilization;
					*desiredPESizeNM = thisPESize;
					*desiredNumTileNM = thisDesign[0];
				}
			}
			*desiredTileSizeCM = MAX(maxTileSizeCM, 4*param->numRowSubArray);
			vector<double> initialDesignCM;
			initialDesignCM = TileDesignCM((*desiredTileSizeCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesignCM[0];
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*desiredPESizeCM = (*desiredTileSizeCM)/2;
			/*** PE Design ***/
			for (double thisPESize = (*desiredTileSizeCM)/2; thisPESize > 2*param->numRowSubArray; thisPESize/=2) {
				// define PE Size for layers use conventional mapping
				double thisUtilization = 0;
				vector<vector<double> > thisDesign;
				thisDesign = PEDesign(true, thisPESize, (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredPESizeCM = thisPESize;
				}
			}
			peDup = PEDesign(false, (*desiredPESizeCM), (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup((*desiredPESizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer(false, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer(true, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer(false, true, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), (*desiredPESizeNM), markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	} else {   // all Conventional Mapping
		if (maxTileSizeCM < 4*param->numRowSubArray) {
			cout << "ERROR: SubArray Size is too large, which break the chip hierarchey, please decrease the SubArray size! " << endl;
		} else {
			/*** Tile Design ***/
			*desiredTileSizeCM = MAX(maxTileSizeCM, 4*param->numRowSubArray);
			vector<double> initialDesign;
			initialDesign = TileDesignCM((*desiredTileSizeCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			*desiredNumTileCM = initialDesign[0];
			for (double thisTileSize = MAX(maxTileSizeCM, 4*param->numRowSubArray); thisTileSize > 4*param->numRowSubArray; thisTileSize/=2) {
				// for layers use conventional mapping
				double thisUtilization = 0;
				vector<double> thisDesign;
				thisDesign = TileDesignCM(thisTileSize, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[2];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredTileSizeCM = thisTileSize;
					*desiredNumTileCM = thisDesign[0];
				}
			}
			*desiredPESizeCM = (*desiredTileSizeCM)/2;
			/*** PE Design ***/
			for (double thisPESize = (*desiredTileSizeCM)/2; thisPESize > 2*param->numRowSubArray; thisPESize/=2) {
				// define PE Size for layers use conventional mapping
				double thisUtilization = 0;
				vector<vector<double> > thisDesign;
				thisDesign = PEDesign(true, thisPESize, (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
				thisUtilization = thisDesign[1][0];
				if (thisUtilization > maxUtilizationCM) {
					maxUtilizationCM = thisUtilization;
					*desiredPESizeCM = thisPESize;
				}
			}
			peDup = PEDesign(false, (*desiredPESizeCM), (*desiredTileSizeCM), (*desiredNumTileCM), markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** SubArray Duplication ***/
			subArrayDup = SubArrayDup((*desiredPESizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse);
			/*** Design SubArray ***/
			numTileEachLayer = OverallEachLayer(false, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			utilizationEachLayer = OverallEachLayer(true, false, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
			speedUpEachLayer = OverallEachLayer(false, true, peDup, subArrayDup, pipelineSpeedUp, (*desiredTileSizeCM), 0, markNM, netStructure, numRowPerSynapse, numColPerSynapse, numPENM);
		}
	}
	
	if (param->pipeline) {
		// update # of tile for pipeline system design
		*desiredNumTileCM = 0;
		*desiredNumTileNM = 0;
		for (int i=0; i<netStructure.size(); i++) {
			if (markNM[i] == 0) {
				*desiredNumTileCM = (*desiredNumTileCM) + numTileEachLayer[0][i]*numTileEachLayer[1][i];
			} else {
				*desiredNumTileNM = (*desiredNumTileNM) + numTileEachLayer[0][i]*numTileEachLayer[1][i];
			}
		}
	}
	
	*numTileRow = ceil((double)sqrt((double)(*desiredNumTileCM)+(double)(*desiredNumTileNM)));
	*numTileCol = ceil((double)((*desiredNumTileCM)+(*desiredNumTileNM))/(double)(*numTileRow));
	
	vector<vector<double> > tileLocaEachLayer;	// start location of each layer
	vector<double> tileLocaEachLayerRow;
	vector<double> tileLocaEachLayerCol;
	double thisTileTotal;
	for (int i=0; i<netStructure.size(); i++) {
		if (i==0) {
			tileLocaEachLayerRow.push_back(0);
			tileLocaEachLayerCol.push_back(0);
		} else {
			thisTileTotal += numTileEachLayer[0][i]*numTileEachLayer[1][i];
			tileLocaEachLayerRow.push_back((int)thisTileTotal/(*numTileRow));
			tileLocaEachLayerCol.push_back((int)thisTileTotal%(*numTileRow));
		}
	}
	tileLocaEachLayer.push_back(tileLocaEachLayerRow);
	tileLocaEachLayer.push_back(tileLocaEachLayerCol);
	
	if (findNumTile) {
		return numTileEachLayer;
	} else if (findUtilization) {
		return utilizationEachLayer;
	} else if (findSpeedUp) {
		return speedUpEachLayer;
	} else {
		return tileLocaEachLayer;
	}
	peDup.clear();
	subArrayDup.clear();
	numTileEachLayer.clear();
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
}


void ChipInitialize(InputParameter& inputParameter, Technology& tech, MemCell& cell, const vector<vector<double> > &netStructure, const vector<int > &markNM, const vector<vector<double> > &numTileEachLayer,
					double numPENM, double desiredNumTileNM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, double desiredPESizeCM, int numTileRow, int numTileCol) { 

	/*** Initialize Tile ***/
	TileInitialize(inputParameter, tech, cell, numPENM, desiredPESizeNM, ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM)), desiredPESizeCM);

	// find max layer and define the global buffer: enough to hold the max layer inputs
	double maxLayerInput = 0;
	// find max # tiles needed to be added at the same time
	double maxTileAdded = 0;
	for (int i=0; i<netStructure.size(); i++) {
		double input = netStructure[i][0]*netStructure[i][1]*netStructure[i][2];  // IFM_Row * IFM_Column * IFM_depth
		if (! param->pipeline) {
			if (input > maxLayerInput) {
				maxLayerInput = input;
			}
			if (markNM[i] == 0) {
				globalBusWidth += (desiredTileSizeCM)+(desiredTileSizeCM)/param->numColMuxed;
			} else {
				globalBusWidth += (desiredPESizeNM)*ceil((double)sqrt(numPENM))+(desiredPESizeNM)*ceil((double)sqrt(numPENM))/param->numColMuxed;
			}
		} else {
			maxLayerInput += netStructure[i][0]*netStructure[i][1]*netStructure[i][2]/2;
			if (markNM[i] == 0) {
				globalBusWidth += ((desiredTileSizeCM)+(desiredTileSizeCM)/param->numColMuxed)*numTileEachLayer[0][i]*numTileEachLayer[1][i];
			} else {
				globalBusWidth += ((desiredPESizeNM)*ceil((double)sqrt(numPENM))+(desiredPESizeNM)*ceil((double)sqrt(numPENM))/param->numColMuxed)*numTileEachLayer[0][i]*numTileEachLayer[1][i];
			}
		}
		
		
		if (numTileEachLayer[0][i] > maxTileAdded) {
			maxTileAdded = numTileEachLayer[0][i];
		}
	}
	// have to limit the global bus width --> cannot grow dramatically with num of tile
	while (globalBusWidth > param->maxGlobalBusWidth) {
		globalBusWidth /= 2;
	}
	
	// define bufferSize for inference operation
	
	// 230920 update
	int bufferSize;
	if (param->sync_data_transfer) {
		bufferSize = param->numBitInput*maxLayerInput*2;
	}
	else {
		bufferSize = param->numBitInput*maxLayerInput;	
	}

	//globalBuffer->Initialize(param->numBitInput*maxLayerInput, globalBusWidth, 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	numBufferCore = ceil((double) bufferSize/((double) param->globalBufferCoreSizeRow*(double) param->globalBufferCoreSizeCol));
	//numBufferCore = ceil(1.5*numBufferCore);
	
	// 230920 update
	if (param->sync_data_transfer) {
		globalBuffer->Initialize((param->globalBufferCoreSizeRow*param->globalBufferCoreSizeCol), globalBusWidth, 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	}
	else {
		globalBuffer->Initialize((param->globalBufferCoreSizeRow*param->globalBufferCoreSizeCol), param->globalBufferCoreSizeCol, 1, param->unitLengthWireResistance, param->clkFreq, param->globalBufferType);
	}

	// maxPool->Initialize(param->numBitInput, 2*2, (desiredTileSizeCM), param->clkFreq);
	// 230920 update
	maxPool->Initialize(param->numBitInput, 2*2,  max((desiredTileSizeCM/(double) param->numColMuxed), desiredPESizeNM/(double) param->numColMuxed), param->clkFreq);
	
	// Anni update
	if (param->globalBusType) {
	
		// 230920 update
		double areaNMTile, areaCMTile;
		double NMheight, NMwidth, CMheight, CMwidth;

		if (param->novelMapping) {
			areaNMTile = TileCalculateArea(numPENM, desiredPESizeNM, true, &NMheight, &NMwidth)[5];
			areaCMTile = TileCalculateArea(pow(ceil((double) desiredTileSizeCM/(double) desiredPESizeCM), 2), desiredPESizeCM, false, &CMheight, &CMwidth)[5];
			
			// assigning # of NM tiles for CM tiles		
			double eq_CMT = areaNMTile; 
			double index = 1; 

			if ( eq_CMT < areaCMTile) {
				while (eq_CMT < areaCMTile )
					{	
						index = index + 1; 
						eq_CMT = index * areaNMTile; 
					}
			}

			int total_eq_NMT = index * desiredNumTileCM +  desiredNumTileNM;
			double row_tile_number = total_eq_NMT * 2.0;
			double column_tile_number = 1.0 ;
			double periphery = 0;
			
			while ( ((double)(row_tile_number-column_tile_number)/(double)row_tile_number) > 0.2  ) {

					periphery = total_eq_NMT * 2;
					for (int m=1; m<=floor((double) total_eq_NMT/2.0); m++) {
						if ((total_eq_NMT % m == 0) && ((m + (double)total_eq_NMT/(double)m) < periphery)) 
							{ periphery = m + (double) total_eq_NMT/(double) m; 
							row_tile_number = (double) total_eq_NMT/(double) m; 
							column_tile_number = m;
							}
					}
					
				if (((double)(row_tile_number-column_tile_number)/(double)row_tile_number) > 0.2) total_eq_NMT = total_eq_NMT + 1;

			}

			param->totaltile_num = total_eq_NMT;
			if (param->sync_data_transfer) GhTree->Initialize(row_tile_number, column_tile_number, param->globalBusDelayTolerance, globalBusWidth, param->clkFreq);
			else GhTree->Initialize((numTileRow), (numTileCol), param->globalBusDelayTolerance, globalBusWidth, param->clkFreq);
		} else {
			GhTree->Initialize((numTileRow), (numTileCol), param->globalBusDelayTolerance, globalBusWidth, param->clkFreq);
		}
	
	} else {
		GBus->Initialize((numTileRow), (numTileCol), param->globalBusDelayTolerance, globalBusWidth, param->clkFreq);
	}
	
	//activation inside Tile or outside?
	if (param->chipActivation) {
		// Anni update: numBitSubarrayOutput
		int numBitSubarrayOutput = 0;
		if (param->parallelRead) {		
			numBitSubarrayOutput = log2((double)param->levelOutput)+ceil(log2(ceil(param->numRowSubArray/param->numRowParallel)))+param->numBitInput+(param->numColPerSynapse-1)*param->cellBit+1;
		} else{
			numBitSubarrayOutput = ceil(log2((double)param->numRowSubArray))+param->cellBit+param->numBitInput+(param->numColPerSynapse-1)*param->cellBit+1;
		}

		int maxThroughputTile, maxAddFromSubArray;
		if (param->novelMapping) {
			// 230920 update
			maxThroughputTile = (int) max((desiredTileSizeCM), ceil((desiredPESizeNM)));
			// maxThroughputTile = (int) max((desiredTileSizeCM), ceil((double)sqrt(numPENM))*(desiredPESizeNM));
			maxAddFromSubArray = (int) max(ceil((double)(desiredPESizeCM)/(double)param->numRowSubArray), ceil((double)(desiredPESizeNM)/(double)param->numRowSubArray));   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) max(ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM)), ceil((double)sqrt(numPENM)));    // from ProcessingUnit to Tile
			if (param->pipeline) {
				maxThroughputTile *= (netStructure.size()+1);
				maxAddFromSubArray *= (netStructure.size()+1);
			}
			// Anni update
			Gaccumulation->Initialize((int) maxTileAdded, numBitSubarrayOutput+ceil(log2((double) maxAddFromSubArray)), ceil((double) maxThroughputTile/(double) param->numColMuxed), param->clkFreq);

			if (param->reLu) {
				GreLu->Initialize(ceil((double) maxThroughputTile/(double) param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				// 1.4 update; Anni update
				Gsigmoid->Initialize(param->Activationtype, param->numBitInput, numBitSubarrayOutput+ceil(log2((double) maxAddFromSubArray))+ceil(log2((double) maxTileAdded)), 
									ceil((double) maxThroughputTile/(double) param->numColMuxed), param->clkFreq);
			}
		} else {
			maxAddFromSubArray = (int) ceil((double)(desiredPESizeCM)/(double)param->numRowSubArray);   // from subArray to ProcessingUnit
			maxAddFromSubArray *= (int) ceil((double)(desiredTileSizeCM)/(double)(desiredPESizeCM));    // from ProcessingUnit to Tile
			if (param->pipeline) {
				maxAddFromSubArray *= (netStructure.size()+1);
			}
			// Anni update
			Gaccumulation->Initialize((int) maxTileAdded, numBitSubarrayOutput+ceil(log2((double)maxAddFromSubArray)), ceil((double)(desiredTileSizeCM)/(double)param->numColMuxed), param->clkFreq);

			if (param->reLu) {
				GreLu->Initialize(ceil((double)(desiredTileSizeCM)/(double)param->numColMuxed), param->numBitInput, param->clkFreq);
			} else {
				// 1.4 update; Anni update
				Gsigmoid->Initialize(param->Activationtype, param->numBitInput, numBitSubarrayOutput+ceil(log2((double) maxAddFromSubArray))+ceil(log2((double) maxTileAdded)), 
									ceil((double) (desiredTileSizeCM)/(double) param->numColMuxed), param->clkFreq);
			}
		}
	} else {   // activation inside tiles
		int maxThroughputTile;
		if (param->novelMapping) {

			// 230920 update
			maxThroughputTile = (int) max((desiredTileSizeCM), ceil((desiredPESizeNM)));
			if (param->pipeline) {
				maxThroughputTile *= (netStructure.size()+1);
			}
			// Anni update
			Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) maxThroughputTile/(double) param->numColMuxed), param->clkFreq);
		} else {
			// Anni update
			Gaccumulation->Initialize((int) maxTileAdded, param->numBitInput, ceil((double) (desiredTileSizeCM)/(double) param->numColMuxed), param->clkFreq);
		}
	}
}



vector<double> ChipCalculateArea(InputParameter& inputParameter, Technology& tech, MemCell& cell, double desiredNumTileNM, double numPENM, double desiredPESizeNM, double desiredNumTileCM, double desiredTileSizeCM, 
						double desiredPESizeCM, int numTileRow, double *height, double *width, double *CMTileheight, double *CMTilewidth, double *NMTileheight, double *NMTilewidth) {
	
	vector<double> areaResults;
	
	double area = 0;
	double areaIC = 0;
	double areaADC = 0;
	double areaAccum = 0;
	double areaOther = 0;
	double areaArray = 0;
	
	double NMheight = 0;
	double NMwidth = 0;
	double CMheight = 0;
	double CMwidth = 0;
	
	*NMTileheight = 0;
	*NMTilewidth = 0;
	*CMTileheight = 0;
	*CMTilewidth = 0;
	*height = 0;
	*width = 0;
	
	vector<double> areaCMTile;
	vector<double> areaNMTile;
	
	if (param->novelMapping) {
		areaNMTile = TileCalculateArea(numPENM, desiredPESizeNM, true, &NMheight, &NMwidth);
		double NMTileArea = areaNMTile[0];
		double NMTileAreaIC = areaNMTile[1];
		double NMTileAreaADC = areaNMTile[2];
		double NMTileAreaAccum = areaNMTile[3];
		double NMTileAreaOther = areaNMTile[4];
		double NMTileAreaArray = areaNMTile[5];

		// 230920 update
		if (param->sync_data_transfer) area += NMTileArea*param->totaltile_num;
		else area += NMTileArea*desiredNumTileNM;

		areaIC += NMTileAreaIC*desiredNumTileNM;
		areaADC += NMTileAreaADC*desiredNumTileNM;
		areaAccum += NMTileAreaAccum*desiredNumTileNM;
		areaOther += NMTileAreaOther*desiredNumTileNM;
		
		// 230920 update
		if (param->sync_data_transfer) areaArray += NMTileAreaArray*param->totaltile_num;
		else areaArray += NMTileAreaArray*desiredNumTileNM;	
		
		
		*NMTileheight = NMheight;
		*NMTilewidth = NMwidth;
	}
	areaCMTile = TileCalculateArea(pow(ceil((double) desiredTileSizeCM/(double) desiredPESizeCM), 2), desiredPESizeCM, false, &CMheight, &CMwidth);
	
	double CMTileArea = areaCMTile[0];
	double CMTileAreaIC = areaCMTile[1];
	double CMTileAreaADC = areaCMTile[2];
	double CMTileAreaAccum = areaCMTile[3];
	double CMTileAreaOther = areaCMTile[4];
	double CMTileAreaArray = areaCMTile[5];

	// 230920 update
	if (param->sync_data_transfer) area += 0;
	else area += CMTileArea*desiredNumTileCM;

	
	areaIC += CMTileAreaIC*desiredNumTileCM;
	areaADC += CMTileAreaADC*desiredNumTileCM;
	areaAccum += CMTileAreaAccum*desiredNumTileCM;
	areaOther += CMTileAreaOther*desiredNumTileCM;
	
	if (param->sync_data_transfer) areaArray += 0;
	else areaArray += CMTileAreaArray*desiredNumTileCM;
	
	*CMTileheight = CMheight;
	*CMTilewidth = CMwidth;
	
	// global buffer is made up by multiple cores
	globalBuffer->CalculateArea(numTileRow*max(NMheight, CMheight), NULL, NONE);
	double globalBufferArea = globalBuffer->area*numBufferCore;
	double globalBufferHeight = numTileRow*max(NMheight, CMheight);
	double globalBufferWidth = globalBufferArea/globalBufferHeight;														
	// Anni update
	if (param->globalBusType) {
		GhTree->CalculateArea(max(NMheight, CMheight), max(NMwidth, CMwidth), param->treeFoldedRatio);
		area += GhTree->area;
		areaIC += GhTree->area;
	} else {													
		GBus->CalculateArea(max(NMheight, CMheight), max(NMwidth, CMwidth), param->treeFoldedRatio);
		area += GBus->area;
		areaIC += GBus->area;
	}
	maxPool->CalculateUnitArea(NONE);
	maxPool->CalculateArea(globalBufferWidth);
	Gaccumulation->CalculateArea(NULL, globalBufferHeight/3, NONE);
	
	double areaGreLu = 0;
	double areaGsigmoid = 0;
	
	if (param->chipActivation) {
		if (param->reLu) {
			GreLu->CalculateArea(NULL, globalBufferWidth/3, NONE);
			area += GreLu->area;
			areaGreLu += GreLu->area;
		} else {
			Gsigmoid->CalculateUnitArea(NONE);
			Gsigmoid->CalculateArea(NULL, globalBufferWidth/3, NONE);
			area += Gsigmoid->area;
			areaGsigmoid += Gsigmoid->area;
		}
	}
	
	// Anni update: GhTree->area added before
	area += globalBufferArea + maxPool->area + Gaccumulation->area;
	
	areaResults.push_back(area);
	areaResults.push_back(areaIC);
	areaResults.push_back(areaADC);
	areaResults.push_back(areaAccum + Gaccumulation->area);
	areaResults.push_back(areaOther + globalBufferArea + maxPool->area + areaGreLu + areaGsigmoid);
	areaResults.push_back(areaArray);
	
	*height = sqrt(area);
	*width = area/(*height);
	
	return areaResults;
}

// Anni update: add double *leakageSRAMInUse
double ChipCalculatePerformance(InputParameter& inputParameter, Technology& tech, MemCell& cell, int layerNumber, const string &newweightfile, const string &oldweightfile, const string &inputfile, bool followedByMaxPool, 
							const vector<vector<double> > &netStructure, const vector<int> &markNM, const vector<vector<double> > &numTileEachLayer, const vector<vector<double> > &utilizationEachLayer, 
							const vector<vector<double> > &speedUpEachLayer, const vector<vector<double> > &tileLocaEachLayer, double numPENM, double desiredPESizeNM, double desiredTileSizeCM, 
							double desiredPESizeCM, double CMTileheight, double CMTilewidth, double NMTileheight, double NMTilewidth,
							double *readLatency, double *readDynamicEnergy, double *leakage, double *leakageSRAMInUse, double *bufferLatency, double *bufferDynamicEnergy, double *icLatency, double *icDynamicEnergy, 
							double *coreLatencyADC, double *coreLatencyAccum, double *coreLatencyOther, double *coreEnergyADC, double *coreEnergyAccum, double *coreEnergyOther, bool CalculateclkFreq, double *clkPeriod) {
	
	
	int numRowPerSynapse, numColPerSynapse;
	numRowPerSynapse = param->numRowPerSynapse;
	numColPerSynapse = param->numColPerSynapse;
	
	// only get performance of single layer
	int l = layerNumber;
	// get weight matrix file Size
	int weightMatrixRow = netStructure[l][2]*netStructure[l][3]*netStructure[l][4]*numRowPerSynapse;
	int weightMatrixCol = netStructure[l][5]*numColPerSynapse;
	
	// load in whole file 
	vector<vector<double> > inputVector;
	inputVector = LoadInInputData(inputfile); 
	vector<vector<double> > newMemory;
	newMemory = LoadInWeightData(newweightfile, numRowPerSynapse, numColPerSynapse, param->maxConductance, param->minConductance);
	
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
	
	double tileLeakage = 0;
	// Anni update
	double tileLeakageSRAMInUse = 0;

	// Anni update: update Clock frequency
	if(!CalculateclkFreq) {	
		globalBuffer->clkFreq = param->clkFreq; 
		GhTree->clkFreq = param->clkFreq; 
		GBus->clkFreq = param->clkFreq; 						  
		Gaccumulation->clkFreq = param->clkFreq; 
		Gsigmoid->clkFreq = param->clkFreq; 
		GreLu->clkFreq = param->clkFreq; 
		maxPool->clkFreq = param->clkFreq; 
	}

	int numInVector = (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
	
	// 230920 update
	int totalNumTile = 0;
	int totalInput=0; 

	for (int i=0; i<netStructure.size(); i++) {
		totalNumTile += numTileEachLayer[0][i] * numTileEachLayer[1][i];
		if (markNM[l]==0){
			totalInput += netStructure[l][2]*netStructure[l][3]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
		}
		else {
			totalInput += netStructure[l][2]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
		}	
	
	}
	// Anni update:
	double totalnumTileRow = ceil((double)sqrt(totalNumTile));
	double totalnumTileCol = ceil((double)(totalNumTile)/(double)(totalnumTileRow));																			 
	
	if (markNM[l] == 0) {   // conventional mapping
		for (int i=0; i<ceil((double) netStructure[l][2]*(double) netStructure[l][3]*(double) netStructure[l][4]*(double) numRowPerSynapse/desiredTileSizeCM); i++) {       // # of tiles in row
			for (int j=0; j<ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredTileSizeCM); j++) {   // # of tiles in Column
				
				
				double tileReadLatency = 0;
				double tileReadDynamicEnergy = 0;
				double tilebufferLatency = 0;
				double tilebufferDynamicEnergy = 0;
				double tileicLatency = 0;
				double tileicDynamicEnergy = 0;
				double tileLatencyADC = 0;
				double tileLatencyAccum = 0;
				double tileLatencyOther = 0;
				double tileEnergyADC = 0;
				double tileEnergyAccum = 0;
				double tileEnergyOther = 0;

				int numRowMatrix = min(desiredTileSizeCM, weightMatrixRow-i*desiredTileSizeCM);
				int numColMatrix = min(desiredTileSizeCM, weightMatrixCol-j*desiredTileSizeCM);
				
				// assign weight and input to specific tile
				vector<vector<double> > tileMemory;
				tileMemory = CopyArray(newMemory, i*desiredTileSizeCM, j*desiredTileSizeCM, numRowMatrix, numColMatrix);
				
				vector<vector<double> > tileInput;
				tileInput = CopyInput(inputVector, i*desiredTileSizeCM, numInVector*param->numBitInput, numRowMatrix);

				// Anni update: add &tileLeakageSRAMInUse
				TileCalculatePerformance(tileMemory, tileMemory, tileInput, markNM[l], ceil((double)desiredTileSizeCM/(double)desiredPESizeCM), desiredPESizeCM, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, numInVector*param->numBitInput, cell, &tileReadLatency, &tileReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse,
									&tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy, 
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther, CalculateclkFreq, clkPeriod);

				*readLatency = MAX(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*bufferLatency = MAX(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = MAX(tileicLatency, (*icLatency));
				*icDynamicEnergy += tileicDynamicEnergy;
				
				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));
				
				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;
			}
		}
		if(!CalculateclkFreq){
			if (param->chipActivation) {
				if (param->reLu) {
					GreLu->CalculateLatency(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
					GreLu->CalculatePower(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
					
					// 230920 update
					if (!param->sync_data_transfer) {
					*readLatency += GreLu->readLatency;
					*coreLatencyOther += GreLu->readLatency;
					}

					*readDynamicEnergy += GreLu->readDynamicEnergy;
					*coreEnergyOther += GreLu->readDynamicEnergy;

				} else {
					Gsigmoid->CalculateLatency(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
					Gsigmoid->CalculatePower(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
					
					// 230920 update
					if (!param->sync_data_transfer) {
					*readLatency += Gsigmoid->readLatency;
					*coreLatencyOther += Gsigmoid->readLatency;
					}

					*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
					
					*coreEnergyOther += Gsigmoid->readDynamicEnergy;
				}
			}
			
			if (numTileEachLayer[0][l] > 1) { 
				// 230920 update  
				Gaccumulation->CalculateLatency(ceil(weightMatrixCol/numColPerSynapse  *(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l], 0);
				Gaccumulation->CalculatePower(ceil(weightMatrixCol/numColPerSynapse  *(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l]);
				
				*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
				
				
				// 230920 update
				if (!param->sync_data_transfer) {
					*readLatency += Gaccumulation->readLatency;
					*coreLatencyAccum += Gaccumulation->readLatency;
				}
							
				
				*coreEnergyAccum += Gaccumulation->readDynamicEnergy;
			}
			
			// if this layer is followed by Max Pool
			if (followedByMaxPool) {

				// 230920 update
				maxPool->CalculateLatency(1e20, 0, ceil((double) (numInVector * weightMatrixCol/numColPerSynapse/(double) maxPool->window)/maxPool->numMaxPooling));
				maxPool->CalculatePower(ceil((double) (numInVector * weightMatrixCol/numColPerSynapse/(double) maxPool->window)/maxPool->numMaxPooling));

				if (!param->sync_data_transfer) {
					*readLatency += maxPool->readLatency;
					*coreLatencyOther += maxPool->readLatency; 
				}

				*readDynamicEnergy += maxPool->readDynamicEnergy;
				
				*coreEnergyOther += maxPool->readDynamicEnergy;
			}							  
			
			double numBitToLoadOut = weightMatrixRow*param->numBitInput*numInVector;
			double numBitToLoadIn = ceil(weightMatrixCol/param->numColPerSynapse)*param->numBitInput*numInVector/(netStructure[l][6]? 4:1);
			
			// Anni update
			if (param->globalBusType) {
				
				// 230920 update
				double fraction = netStructure[l][2]*netStructure[l][3]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];

				if (param->novelMapping && param->sync_data_transfer) GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/floor(GhTree->busWidth*(fraction/totalInput))));
				else GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
				if (param->novelMapping && param->sync_data_transfer) GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil(GhTree->busWidth*(fraction/totalInput)), 
							ceil((numBitToLoadOut*param->inputtoggle)/floor(GhTree->busWidth*(fraction/totalInput) )) + ceil((numBitToLoadIn*param->outputtoggle)/floor(GhTree->busWidth*(fraction/totalInput) ) ) );
				else GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile)), 
							ceil((numBitToLoadOut*param->inputtoggle)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile) )) + ceil((numBitToLoadIn*param->outputtoggle)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile) ) ) );
				
				// GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/GhTree->busWidth));
				// GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], CMTileheight, CMTilewidth, GhTree->busWidth, 
				// 					ceil((numBitToLoadOut+numBitToLoadIn)/GhTree->busWidth));
			} else{
				GBus->CalculateLatency(ceil(totalnumTileCol/2), ceil(totalnumTileRow/2), CMTileheight, CMTilewidth, \
									ceil(numBitToLoadOut/GBus->busWidth*totalnumTileRow/numTileEachLayer[0][l]), ceil(numBitToLoadIn/GBus->busWidth*totalnumTileCol/numTileEachLayer[1][l]));
				GBus->CalculatePower(ceil(totalnumTileCol/2), ceil(totalnumTileRow/2), CMTileheight, CMTilewidth, numBitToLoadOut*param->inputtoggle, numBitToLoadIn*param->outputtoggle);
			}

			// 230920 update
			double chip_bufferclk = max(GhTree->critical_latency, 1.0); 
			
			globalBuffer->CalculateLatency(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
									globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
			
			globalBuffer->CalculatePower(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
									globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
			
			// since multi-core buffer has improve the parallelism

			// 230920 update
			if (param->sync_data_transfer) {
				double fraction = netStructure[l][2]*netStructure[l][3]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
				
				globalBuffer->readLatency *= chip_bufferclk * ceil(totalInput/fraction);
				globalBuffer->writeLatency *= chip_bufferclk * ceil(totalInput/fraction);
			
			} else {
				// since multi-core buffer has improve the parallelism
				globalBuffer->readLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
				globalBuffer->writeLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
								
			}				
			
			// Anni update: 
			// each time, only a part of the ic is used to transfer data to a part of the tiles
			// globalBuffer->readLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
			// globalBuffer->writeLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
		}
	} else {   // novel Mapping
		for (int i=0; i<ceil((double) netStructure[l][2]*(double) numRowPerSynapse/(double) desiredPESizeNM); i++) {       // # of tiles in row
			for (int j=0; j<ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredPESizeNM); j++) {   // # of tiles in Column
				double tileReadLatency = 0;
				double tileReadDynamicEnergy = 0;
				double tilebufferLatency = 0;
				double tilebufferDynamicEnergy = 0;
				double tileicLatency = 0;
				double tileicDynamicEnergy = 0;
				double tileLatencyADC = 0;
				double tileLatencyAccum = 0;
				double tileLatencyOther = 0;
				double tileEnergyADC = 0;
				double tileEnergyAccum = 0;
				double tileEnergyOther = 0;
				
				// novel mapping
				int numtileEachLayerRow = ceil((double) netStructure[l][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
				int numtileEachLayerCol = ceil((double) netStructure[l][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				
				int numRowMatrix = min(desiredPESizeNM*numPENM, weightMatrixRow-i*desiredPESizeNM*numPENM);
				int numColMatrix = min(desiredPESizeNM, weightMatrixCol-j*desiredPESizeNM);
				
				// assign weight and input to specific tile
				vector<vector<double> > tileMemory;
				tileMemory = ReshapeArray(newMemory, i*desiredPESizeNM, j*desiredPESizeNM, (int) netStructure[l][2]*numRowPerSynapse/numtileEachLayerRow, 
									(int) netStructure[l][5]*numColPerSynapse/numtileEachLayerCol, numPENM, (int) netStructure[l][2]*numRowPerSynapse);

				vector<vector<double> > tileInput;
				tileInput = ReshapeInput(inputVector, i*desiredPESizeNM, numInVector*param->numBitInput, 
									(int) netStructure[l][2]*numRowPerSynapse/numtileEachLayerRow, numPENM, (int) netStructure[l][2]*numRowPerSynapse);

				// Anni update: add &tileLeakageSRAMInUse
				TileCalculatePerformance(tileMemory, tileMemory, tileInput, markNM[l], numPENM, desiredPESizeNM, speedUpEachLayer[0][l], speedUpEachLayer[1][l],
									numRowMatrix, numColMatrix, numInVector*param->numBitInput, cell, 
									&tileReadLatency, &tileReadDynamicEnergy, &tileLeakage, &tileLeakageSRAMInUse, &tilebufferLatency, &tilebufferDynamicEnergy, &tileicLatency, &tileicDynamicEnergy,
									&tileLatencyADC, &tileLatencyAccum, &tileLatencyOther, &tileEnergyADC, &tileEnergyAccum, &tileEnergyOther, CalculateclkFreq, clkPeriod);
				
	
				*readLatency = MAX(tileReadLatency, (*readLatency));
				*readDynamicEnergy += tileReadDynamicEnergy;
				*bufferLatency = MAX(tilebufferLatency, (*bufferLatency));
				*bufferDynamicEnergy += tilebufferDynamicEnergy;
				*icLatency = MAX(tileicLatency, (*icLatency));
				*icDynamicEnergy += tileicDynamicEnergy;
				
				*coreLatencyADC = MAX(tileLatencyADC, (*coreLatencyADC));
				*coreLatencyAccum = MAX(tileLatencyAccum, (*coreLatencyAccum));
				*coreLatencyOther = MAX(tileLatencyOther, (*coreLatencyOther));
				
				*coreEnergyADC += tileEnergyADC;
				*coreEnergyAccum += tileEnergyAccum;
				*coreEnergyOther += tileEnergyOther;
			}
		}
		if(!CalculateclkFreq){
			if (param->chipActivation) {
				if (param->reLu) {
					GreLu->CalculateLatency(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
					GreLu->CalculatePower(ceil(numInVector*netStructure[l][5]/(double) GreLu->numUnit));
					// 230920 update
					if (!param->sync_data_transfer)	{
						*readLatency += GreLu->readLatency;
						*coreLatencyOther += GreLu->readLatency;
					}

					*readDynamicEnergy += GreLu->readDynamicEnergy;
					*coreEnergyOther += GreLu->readDynamicEnergy;

				} else {
					Gsigmoid->CalculateLatency(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
					Gsigmoid->CalculatePower(ceil(numInVector*netStructure[l][5]/Gsigmoid->numEntry));
					
					// 230920 update
					if (!param->sync_data_transfer)	{	
						*readLatency += Gsigmoid->readLatency;
						*coreLatencyOther += Gsigmoid->readLatency;
					}

					*readDynamicEnergy += Gsigmoid->readDynamicEnergy;
					*coreEnergyOther += Gsigmoid->readDynamicEnergy;
				}
			}
			
			if (numTileEachLayer[0][l] > 1) {   
				// 230920 update
				Gaccumulation->CalculateLatency(ceil(weightMatrixCol/numColPerSynapse *(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l], 0);
				Gaccumulation->CalculatePower(ceil(weightMatrixCol/numColPerSynapse*(numInVector/(double) Gaccumulation->numAdderTree)), numTileEachLayer[0][l]);

				// 230920 update 
				if (!param->sync_data_transfer) {
				*readLatency += Gaccumulation->readLatency;
				*coreLatencyAccum += Gaccumulation->readLatency;
				}

				*readDynamicEnergy += Gaccumulation->readDynamicEnergy;
				*coreEnergyAccum += Gaccumulation->readDynamicEnergy;
			}
			
			// if this layer is followed by Max Pool
			if (followedByMaxPool) {

				// 230920 update
				maxPool->CalculateLatency(1e20, 0, ceil((double) (numInVector * weightMatrixCol/numColPerSynapse/(double) maxPool->window)/maxPool->numMaxPooling));
				maxPool->CalculatePower(ceil((double) (numInVector * weightMatrixCol/numColPerSynapse/(double) maxPool->window)/maxPool->numMaxPooling));
				
				// 230920 update
				if (!param->sync_data_transfer) {
				*readLatency += maxPool->readLatency;
				*coreLatencyOther += maxPool->readLatency;
				}

				*readDynamicEnergy += maxPool->readDynamicEnergy;
				
				*coreEnergyOther += maxPool->readDynamicEnergy;
			}
			double numBitToLoadOut = weightMatrixRow*param->numBitInput*numInVector/netStructure[l][3];
			double numBitToLoadIn = ceil(weightMatrixCol/param->numColPerSynapse)*param->numBitInput*numInVector/(netStructure[l][6]? 4:1);
			
			// Anni update
			if (param->globalBusType) {

				// 230920 update
				double fraction = netStructure[l][2]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
				
				if (param->novelMapping && param->sync_data_transfer) GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/floor(GhTree->busWidth*(fraction/totalInput))));
				else GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));

				if (param->novelMapping && param->sync_data_transfer) GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil(GhTree->busWidth*(fraction/totalInput)), 
																ceil((numBitToLoadOut*param->inputtoggle)/floor(GhTree->busWidth*(fraction/totalInput))) + ceil((numBitToLoadIn*param->outputtoggle)/floor(GhTree->busWidth*(fraction/totalInput))));
				else GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile)), 
																ceil((numBitToLoadOut*param->inputtoggle)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))) + ceil((numBitToLoadIn*param->outputtoggle)/ceil(GhTree->busWidth*(numTileEachLayer[0][l]*numTileEachLayer[1][l]/totalNumTile))));
				// GhTree->CalculateLatency(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, ceil((numBitToLoadOut+numBitToLoadIn)/GhTree->busWidth));
				// GhTree->CalculatePower(0, 0, tileLocaEachLayer[0][l], tileLocaEachLayer[1][l], NMTileheight, NMTilewidth, GhTree->busWidth, 
				// 				ceil((numBitToLoadOut+numBitToLoadIn)/GhTree->busWidth));
			} else {
				GBus->CalculateLatency(ceil(totalnumTileCol/2), ceil(totalnumTileRow/2), NMTileheight, NMTilewidth, \
									ceil(numBitToLoadOut/GBus->busWidth*totalnumTileRow/numTileEachLayer[0][l]), ceil(numBitToLoadIn/GBus->busWidth*totalnumTileCol/numTileEachLayer[1][l]));
				GBus->CalculatePower(ceil(totalnumTileCol/2), ceil(totalnumTileRow/2), NMTileheight, NMTilewidth, numBitToLoadOut*param->inputtoggle, numBitToLoadIn*param->outputtoggle);
			}

			globalBuffer->CalculateLatency(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
									globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
			globalBuffer->CalculatePower(globalBuffer->interface_width, numBitToLoadOut/globalBuffer->interface_width,
									globalBuffer->interface_width, numBitToLoadIn/globalBuffer->interface_width);
			// since multi-core buffer has improve the parallelism
			// cout<<"before numCore globalBuffer->readLatency: "<<globalBuffer->readLatency<<"	globalBuffer->writeLatency: "<<globalBuffer->writeLatency<<endl;
			// cout<<"numBufferCore: "<<numBufferCore<<"	ceil(globalBusWidth/globalBuffer->interface_width): "<<ceil(globalBusWidth/globalBuffer->interface_width)<<endl;
			
			// 230920 update
			double chip_bufferclk = max(GhTree->critical_latency, 1.0); 

			// 230920 update
			if (param->sync_data_transfer) {				
				double fraction = netStructure[l][2]*netStructure[l][4] * (netStructure[l][0]-netStructure[l][3]+1)/netStructure[l][7]*(netStructure[l][1]-netStructure[l][4]+1)/netStructure[l][7];
				globalBuffer->readLatency *= chip_bufferclk * ceil(totalInput/fraction);
				globalBuffer->writeLatency *= chip_bufferclk * ceil(totalInput/fraction);
			}

			else {
				// since multi-core buffer has improve the parallelism
				globalBuffer->readLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
				globalBuffer->writeLatency /= MIN(numBufferCore, ceil(globalBusWidth/globalBuffer->interface_width));
				// each time, only a part of the ic is used to transfer data to a part of the tiles				
			}			

			// Anni update
			// each time, only a part of the ic is used to transfer data to a part of the tiles
			// globalBuffer->readLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));
			// globalBuffer->writeLatency *= ceil(totalNumTile/(numTileEachLayer[0][l]*numTileEachLayer[1][l]));	
		}
	}		
	if(!CalculateclkFreq){
		if (param->sync_data_transfer) {
			*bufferLatency += globalBuffer->readLatency + globalBuffer->writeLatency;
			*icLatency = 0;
			*readLatency += globalBuffer->readLatency + globalBuffer->writeLatency;
			*coreLatencyOther = globalBuffer->readLatency + globalBuffer->writeLatency;
		} 
		else {

			*bufferLatency += globalBuffer->readLatency + globalBuffer->writeLatency;
			*icLatency += GhTree->readLatency + GBus->readLatency;
			*readLatency += globalBuffer->readLatency + globalBuffer->writeLatency + GhTree->readLatency + GBus->readLatency;
			*coreLatencyOther += globalBuffer->readLatency + globalBuffer->writeLatency + GhTree->readLatency + GBus->readLatency;
		}

		*bufferDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy;
		//Anni update
		
		*icDynamicEnergy += GhTree->readDynamicEnergy + GBus->readDynamicEnergy;
		
		
		*readDynamicEnergy += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy + GBus->readDynamicEnergy;
	
		*coreEnergyOther += globalBuffer->readDynamicEnergy + globalBuffer->writeDynamicEnergy + GhTree->readDynamicEnergy + GBus->readDynamicEnergy;

		*leakage = tileLeakage;
		// Anni update
		*leakageSRAMInUse = tileLeakageSRAMInUse;
	}
	return 0;
}



vector<double> TileDesignCM(double tileSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double numTileTotal = 0;
	double matrixTotalCM = 0;
	double utilization = 0;
	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 0) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) tileSize) * ceil(netStructure[i][5]*numColPerSynapse/tileSize);
			matrixTotalCM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	utilization = matrixTotalCM/(numTileTotal*tileSize*tileSize);
	
	vector<double> tileDesignCM;
	tileDesignCM.push_back(numTileTotal);
	tileDesignCM.push_back(matrixTotalCM);
	tileDesignCM.push_back(utilization);
	return tileDesignCM;
	tileDesignCM.clear();
}

vector<double> TileDesignNM(double peSize, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM){
	double numTileTotal = 0;
	double matrixTotalNM = 0;
	double utilization = 0;
	for (int i=0; i<netStructure.size(); i++) {
		if (markNM[i] == 1) {
			numTileTotal += ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) peSize) * ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
			matrixTotalNM += netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
		}
	}
	utilization = matrixTotalNM/(numTileTotal*peSize*peSize*numPENM);
	vector<double> tileDesignNM;
	tileDesignNM.push_back(numTileTotal);
	tileDesignNM.push_back(matrixTotalNM);
	tileDesignNM.push_back(utilization);
	return tileDesignNM;
	tileDesignNM.clear();
}

vector<vector<double> > PEDesign(bool Design, double peSize, double desiredTileSize, double numTileTotal, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	double matrixTotalCM = 0;
	double utilization = 0;
	vector<double> peDupRow;
	vector<double> peDupCol;
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] ==0) {
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredTileSize)||(netStructure[i][5]*numColPerSynapse <= desiredTileSize) ) {
				int peForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) peSize);
				int peForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) peSize);
				int numPERow = ceil((double) desiredTileSize/(double) peSize);
				int numPECol = ceil((double) desiredTileSize/(double) peSize);
				actualDupRow = floor(numPERow/peForOneMatrixRow)==0? 1:floor(numPERow/peForOneMatrixRow);
				actualDupCol = floor(numPECol/peForOneMatrixCol)==0? 1:floor(numPECol/peForOneMatrixCol);
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
				matrixTotalCM += actualDupRow*actualDupCol*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse*netStructure[i][5]*numColPerSynapse;
			}
		} else {
			actualDupRow = 1;
			actualDupCol = 1;
		}
		peDupRow.push_back(actualDupRow);
		peDupCol.push_back(actualDupCol);
	}
	utilization = matrixTotalCM/(numTileTotal*desiredTileSize*desiredTileSize);
	
	vector<double> matrixTotal;
	matrixTotal.push_back(matrixTotalCM);
	vector<double> utiliz;
	utiliz.push_back(utilization);
	vector<vector<double> > peDesignCM;
	peDesignCM.push_back(matrixTotal);
	peDesignCM.push_back(utiliz);
	matrixTotal.clear();
	utiliz.clear();
	
	vector<vector<double> > peDup;
	peDup.push_back(peDupRow);
	peDup.push_back(peDupCol);
	peDupRow.clear();
	peDupCol.clear();
	if (Design) {
		return peDesignCM;
	} else {
		return peDup;
	}
	peDesignCM.clear();
	peDup.clear();
}

vector<vector<double> > SubArrayDup(double desiredPESizeCM, double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse) {
	vector<double> subArrayDupRow;
	vector<double> subArrayDupCol;
	
	for (int i=0; i<netStructure.size(); i++) {
		int actualDupRow = 0;
		int actualDupCol = 0;
		if (markNM[i] == 0){
			if ( (netStructure[i][2]*netStructure[i][3]*netStructure[i][4]*numRowPerSynapse <= desiredPESizeCM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeCM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeCM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeCM/(double) param->numColSubArray);
				actualDupRow = floor(numSubArrayRow/arrayForOneMatrixRow)==0? 1:floor(numSubArrayRow/arrayForOneMatrixRow);
				actualDupCol = floor(numSubArrayCol/arrayForOneMatrixCol)==0? 1:floor(numSubArrayCol/arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		} else {
			if ( (netStructure[i][2]*numRowPerSynapse <= desiredPESizeNM)||(netStructure[i][5]*numColPerSynapse <= desiredPESizeNM) ) {
				int arrayForOneMatrixRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) param->numRowSubArray);
				int arrayForOneMatrixCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) param->numColSubArray);
				int numSubArrayRow = ceil((double) desiredPESizeNM/(double) param->numRowSubArray);
				int numSubArrayCol = ceil((double) desiredPESizeNM/(double) param->numColSubArray);
				actualDupRow = floor(numSubArrayRow/arrayForOneMatrixRow)==0? 1:floor(numSubArrayRow/arrayForOneMatrixRow);
				actualDupCol = floor(numSubArrayCol/arrayForOneMatrixCol)==0? 1:floor(numSubArrayCol/arrayForOneMatrixCol);
			} else {
				actualDupRow = 1;
				actualDupCol = 1;
			}
		}
		subArrayDupRow.push_back(actualDupRow);
		subArrayDupCol.push_back(actualDupCol);
	}
	vector<vector<double> > subArrayDup;
	subArrayDup.push_back(subArrayDupRow);
	subArrayDup.push_back(subArrayDupCol);
	subArrayDupRow.clear();
	subArrayDupCol.clear();
	return subArrayDup;
	subArrayDup.clear();
}

vector<vector<double> > OverallEachLayer(bool utilization, bool speedUp, const vector<vector<double> > &peDup, const vector<vector<double> > &subArrayDup, const vector<int> &pipelineSpeedUp, double desiredTileSizeCM, 
										double desiredPESizeNM, const vector<int > &markNM, const vector<vector<double> > &netStructure, int numRowPerSynapse, int numColPerSynapse, double numPENM) {
	vector<double> numTileEachLayerRow;
	vector<double> numTileEachLayerCol;
	vector<vector<double> > utilizationEachLayer;	
	vector<double> speedUpEachLayerRow;
	vector<double> speedUpEachLayerCol;
	
	for (int i=0; i<netStructure.size(); i++) {
		vector<double> utilization;
		double numtileEachLayerRow, numtileEachLayerCol, utilizationEach;
		if (markNM[i] == 0) {
			// conventional mapping
			if (!param->pipeline) {
				// layer-by-layer process
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM);
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM);
				utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
											*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM*desiredTileSizeCM);

				utilization.push_back(utilizationEach);
			} else {
				// pipeline system
				// original design
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) netStructure[i][3]*(double) netStructure[i][4]*(double) numRowPerSynapse/desiredTileSizeCM)
										*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]));
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredTileSizeCM);

				utilizationEach = (MAX(pipelineSpeedUp[i], peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])*netStructure[i][2]*netStructure[i][3]*netStructure[i][4]
									*numRowPerSynapse*netStructure[i][5]*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredTileSizeCM*desiredTileSizeCM);

				utilization.push_back(utilizationEach);
			}
		} else {
			if (!param->pipeline) {
				// novel mapping
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM);
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				utilizationEach = (peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
											*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);
				
				utilization.push_back(utilizationEach);
			} else {
				// novel mapping
				numtileEachLayerRow = ceil((double) netStructure[i][2]*(double) numRowPerSynapse/(double) desiredPESizeNM)
										*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i]));
				numtileEachLayerCol = ceil((double) netStructure[i][5]*(double) numColPerSynapse/(double) desiredPESizeNM);
				utilizationEach = (MAX(pipelineSpeedUp[i], peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])*netStructure[i][2]*numPENM*numRowPerSynapse*netStructure[i][5]
											*numColPerSynapse)/(numtileEachLayerRow*numtileEachLayerCol*desiredPESizeNM*desiredPESizeNM*numPENM);
				
				utilization.push_back(utilizationEach);
			}
		}
		numTileEachLayerRow.push_back(numtileEachLayerRow);
		numTileEachLayerCol.push_back(numtileEachLayerCol);
		utilizationEachLayer.push_back(utilization);
		if (!param->pipeline) {
			speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]);
			speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		} else {
			speedUpEachLayerRow.push_back(peDup[0][i]*subArrayDup[0][i]*ceil(pipelineSpeedUp[i]/(peDup[0][i]*peDup[1][i]*subArrayDup[0][i]*subArrayDup[1][i])));
			speedUpEachLayerCol.push_back(peDup[1][i]*subArrayDup[1][i]);
		}
		utilization.clear();
	}

	vector<vector<double> > numTileEachLayer;
	numTileEachLayer.push_back(numTileEachLayerRow);
	numTileEachLayer.push_back(numTileEachLayerCol);
	numTileEachLayerRow.clear();
	numTileEachLayerCol.clear();
	
	vector<vector<double> > speedUpEachLayer;
	speedUpEachLayer.push_back(speedUpEachLayerRow);
	speedUpEachLayer.push_back(speedUpEachLayerCol);
	speedUpEachLayerRow.clear();
	speedUpEachLayerCol.clear();

	if (utilization) {
		return utilizationEachLayer;
	} else if (speedUp) {
		return speedUpEachLayer;
	} else {
		return numTileEachLayer;
	}
	utilizationEachLayer.clear();
	speedUpEachLayer.clear();
	numTileEachLayer.clear();
}



vector<vector<double> > LoadInWeightData(const string &weightfile, int numRowPerSynapse, int numColPerSynapse, double maxConductance, double minConductance) {
	
	ifstream fileone(weightfile.c_str());                           
	string lineone;
	string valone;
	
	int ROW = 0;
	int COL = 0;
	
	if (!fileone.good()) {                                       
		cerr << "Error: the fileone cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(fileone, lineone, '\n')) {                   
			ROW++;                                             
		}
		fileone.clear();
		fileone.seekg(0, ios::beg);                               
		if (getline(fileone, lineone, '\n')) {                      
			istringstream iss (lineone);                         
			while (getline(iss, valone, ',')) {                   
				COL++;
			}
		}	
	}
	fileone.clear();
	fileone.seekg(0, ios::beg);                   
	
	
	double NormalizedMin = 0;
	double NormalizedMax = pow(2, param->synapseBit);
	
	double RealMax = param->algoWeightMax;
	double RealMin = param->algoWeightMin;
	
	vector<vector<double> > weight;            
	// load the data into a weight matrix ...
	for (int row=0; row<ROW; row++) {	
		vector<double> weightrow;
		vector<double> weightrowb;
		getline(fileone, lineone, '\n');              
		istringstream iss;
		iss.str(lineone);
		for (int col=0; col<COL; col++) {       
			while(getline(iss, valone, ',')){	
				istringstream fs;
				fs.str(valone);
				double f=0;
				fs >> f;	
				//normalize weight to integer
				double newdata = ((NormalizedMax-NormalizedMin)/(RealMax-RealMin)*(f-RealMax)+NormalizedMax);
				if (newdata >= 0) {
					newdata += 0.5;
				}else {
					newdata -= 0.5;
				}
				// map and expend the weight in memory array
				int cellrange = pow(2, param->cellBit);
				vector<double> synapsevector(numColPerSynapse);       
				int value = newdata; 
				
				if (param->BNNparallelMode) {
					if (value == 1) {
						weightrow.push_back(maxConductance);
						weightrow.push_back(minConductance);
					} else {
						weightrow.push_back(minConductance);
						weightrow.push_back(maxConductance);
					}
				} else if (param->XNORparallelMode || param->XNORsequentialMode) {
					if (value == 1) {
						weightrow.push_back(maxConductance);
						weightrowb.push_back(minConductance);
					} else {
						weightrow.push_back(minConductance);
						weightrowb.push_back(maxConductance);
					}
				} else {
					int remainder;   
					for (int z=0; z<numColPerSynapse; z++) {   
						remainder = ceil((double)(value%cellrange));
						value = ceil((double)(value/cellrange));
						synapsevector.insert(synapsevector.begin(), remainder);
					}
					for (int u=0; u<numColPerSynapse; u++) {
						double cellvalue = synapsevector[u];
						double conductance = cellvalue/(cellrange-1) * (maxConductance-minConductance) + minConductance;
						weightrow.push_back(conductance);
					}
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			weight.push_back(weightrow);
			weightrow.clear();
			weight.push_back(weightrowb);
			weightrowb.clear();
		} else {
			weight.push_back(weightrow);
			weightrow.clear();
		}
	}
	fileone.close();
	
	return weight;
	weight.clear();
}



vector<vector<double> > CopyArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol) {
	
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



vector<vector<double> > ReshapeArray(const vector<vector<double> > &orginal, int positionRow, int positionCol, int numRow, int numCol, int numPE, int weightMatrixRow) {
	
	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numCol; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][positionCol+j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}
	
	return copy;
	copy.clear();
} 



vector<vector<double> > LoadInInputData(const string &inputfile) {
	
	ifstream infile(inputfile.c_str());     
	string inputline;
	string inputval;
	
	int ROWin=0, COLin=0;      
	if (!infile.good()) {       
		cerr << "Error: the input file cannot be opened!" << endl;
		exit(1);
	}else{
		while (getline(infile, inputline, '\n')) {      
			ROWin++;                               
		}
		infile.clear();
		infile.seekg(0, ios::beg);    
		if (getline(infile, inputline, '\n')) {        
			istringstream iss (inputline);      
			while (getline(iss, inputval, ',')) {       
				COLin++;
			}
		}	
	}
	infile.clear();
	infile.seekg(0, ios::beg);          

	vector<vector<double> > inputvector;              
	// load the data into inputvector ...
	for (int row=0; row<ROWin; row++) {	
		vector<double> inputvectorrow;
		vector<double> inputvectorrowb;
		getline(infile, inputline, '\n');             
		istringstream iss;
		iss.str(inputline);
		for (int col=0; col<COLin; col++) {
			while(getline(iss, inputval, ',')){	
				istringstream fs;
				fs.str(inputval);
				int val=0;
				fs >> val;
				int f=0;

				//1.5 update: inputs are stored as integers now, need to convert to binary
				for (int i = 0; i < param->numBitInput; i++) {
					f = (val >> (param->numBitInput - 1 - i)) & 1;

					if (param->BNNparallelMode) {
						if (f == 1) {
							inputvectorrow.push_back(1);
						} else {
							inputvectorrow.push_back(0);
						}
					} else if (param->XNORparallelMode || param->XNORsequentialMode) {
						if (f == 1) {
							inputvectorrow.push_back(1);
							inputvectorrowb.push_back(0);
						} else {
							inputvectorrow.push_back(0);
							inputvectorrowb.push_back(1);
						}
					} else {
						inputvectorrow.push_back(f);
					}
				}
			}
		}
		if (param->XNORparallelMode || param->XNORsequentialMode) {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
			inputvector.push_back(inputvectorrowb);
			inputvectorrowb.clear();
		} else {
			inputvector.push_back(inputvectorrow);
			inputvectorrow.clear();
		}
	}
	// close the input file ...
	infile.close();
	
	return inputvector;
	inputvector.clear();
}




vector<vector<double> > CopyInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow) {
	
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



vector<vector<double> > ReshapeInput(const vector<vector<double> > &orginal, int positionRow, int numInputVector, int numRow, int numPE, int weightMatrixRow) {
	
	vector<vector<double> > copy;

	for (int k=0; k<numPE; k++) {
		for (int i=0; i<numRow; i++) {
			vector<double> copyRow;
			for (int j=0; j<numInputVector; j++) {
				copyRow.push_back(orginal[positionRow+k*weightMatrixRow+i][j]);
			}
			copy.push_back(copyRow);
			copyRow.clear();
		}
	}
	
	return copy;
	copy.clear();
} 











