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
#include <vector>
#include "constant.h"
#include "formula.h"
#include "SubArray.h"
#include "Param.h"


using namespace std;

extern Param *param;

SubArray::SubArray(InputParameter& _inputParameter, Technology& _tech, MemCell& _cell):
						inputParameter(_inputParameter), tech(_tech), cell(_cell),
						wlDecoder(_inputParameter, _tech, _cell),
						wlDecoderOutput(_inputParameter, _tech, _cell),
						wlNewDecoderDriver(_inputParameter, _tech, _cell),
						wlNewSwitchMatrix(_inputParameter, _tech, _cell),
						rowCurrentSenseAmp(_inputParameter, _tech, _cell),
						mux(_inputParameter, _tech, _cell),
						muxDecoder(_inputParameter, _tech, _cell),
						slSwitchMatrix(_inputParameter, _tech, _cell),
						blSwitchMatrix(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						deMux(_inputParameter, _tech, _cell),
						readCircuit(_inputParameter, _tech, _cell),
						precharger(_inputParameter, _tech, _cell),
						senseAmp(_inputParameter, _tech, _cell),
						colDecoder(_inputParameter, _tech, _cell),
						wlDecoderDriver(_inputParameter, _tech, _cell),
						colDecoderDriver(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell),
						adder(_inputParameter, _tech, _cell),
						dff(_inputParameter, _tech, _cell),
						shiftAdd(_inputParameter, _tech, _cell),
						multilevelSenseAmp(_inputParameter, _tech, _cell),
						multilevelSAEncoder(_inputParameter, _tech, _cell),
						sarADC(_inputParameter, _tech, _cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
} 

void SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	
	numRow = _numRow;    //import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		if (relaxArrayCellWidth) {  //if want to relax the cell width
			lengthRow = (double)numCol * MAX(cell.widthInFeatureSize, MIN_CELL_WIDTH) * tech.featureSize;
		} else { //if not relax the cell width
			lengthRow = (double)numCol * cell.widthInFeatureSize * tech.featureSize;
		}
		if (relaxArrayCellHeight) {  //if want to relax the cell height
			lengthCol = (double)numRow * MAX(cell.heightInFeatureSize, MIN_CELL_HEIGHT) * tech.featureSize;
		} else {  //if not relax the cell height
			lengthCol = (double)numRow * cell.heightInFeatureSize * tech.featureSize;
		}
	
	} else if (cell.memCellType == Type::RRAM ||  cell.memCellType == Type::FeFET) {  //if array is RRAM
		double cellHeight = cell.heightInFeatureSize; 
		double cellWidth = cell.widthInFeatureSize;  
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
		
		//firstly calculate the CMOS resistance and capacitance
		resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
		capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech);
		cell.capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateDrainCap(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, cell.widthInFeatureSize * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellNMOS * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * tech.featureSize, tech);

		if (conventionalSequential) {
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			int adderBit = (int)ceil(log2(numRow)) + 1;	
			int numAdder = numCol/numCellPerSynapse;
			dff.Initialize((adderBit+1)*numAdder, clkFreq);	
			adder.Initialize(adderBit, numAdder);
			if (numReadPulse > 1) {
				shiftAdd.Initialize(numAdder, adderBit+1, clkFreq, spikingMode, numReadPulse);
			}
		} else if (conventionalParallel) {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resCellAccess/numRow/2, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numReadPulse > 1) {
				shiftAdd.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+1, clkFreq, spikingMode, numReadPulse);
			}
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			int adderBit = (int)ceil(log2(numRow)) + avgWeightBit;	
			int numAdder = numCol/numCellPerSynapse;
			dff.Initialize((adderBit+1)*numAdder, clkFreq);	
			adder.Initialize(adderBit, numAdder);
		} else if (BNNparallelMode || XNORparallelMode) {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
		} else {
			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numReadPulse > 1) {
				shiftAdd.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+1, clkFreq, spikingMode, numReadPulse);
			}
		}
		precharger.Initialize(numCol, resCol, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
		sramWriteDriver.Initialize(numCol, activityColWrite, numWriteCellPerOperationNeuro);

    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
		if (cell.accessType == CMOS_access) {	// 1T1R
			cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
			cell.widthAccessCMOS = CalculateOnResistance(tech.featureSize, NMOS, inputParameter.temperature, tech) * LINEAR_REGION_RATIO / cell.resCellAccess;   //get access CMOS width
			if (cell.widthAccessCMOS > cell.widthInFeatureSize) {	// Place transistor vertically
				printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
				exit(-1);
			}

			cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;        //calculate single memory cell resistance_ON
			cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF
			cell.resMemCellAvg = cell.resCellAccess + cell.resistanceAvg;      //calculate single memory cell resistance_AVG

			capRow2 += CalculateGateCap(cell.widthAccessCMOS * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
			capCol += CalculateDrainCap(cell.widthAccessCMOS * tech.featureSize, NMOS, cell.widthInFeatureSize * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit
		} else {	// Cross-point
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
		
		if (conventionalSequential) {  
			double capBL = lengthCol * 0.2e-15/1e-6;
			int numAdder = (int)ceil(numCol/numColMuxed);   // numCol is divisible by numCellPerSynapse
			int numInput = numAdder;        //XXX input number of MUX, 
			double resTg = cell.resMemCellOn;     //transmission gate resistance
			int adderBit = (int)ceil(log2(numRow)) + avgWeightBit;  
			
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);          
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.Initialize(numRow);          
			} else {
				wlDecoderDriver.Initialize(ROW_MODE, numRow, numCol);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);     //SL use switch matrix
			if (numColMuxed>1) {
				mux.Initialize(numInput, numColMuxed, resTg, FPGA);     
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, pow(2, avgWeightBit), clkFreq, numReadCellPerOperationNeuro, false, currentMode);
				if (avgWeightBit > 1) {
					multilevelSAEncoder.Initialize(pow(2, avgWeightBit), numCol/numColMuxed);
				}
			}
			
			dff.Initialize((adderBit+1)*numAdder, clkFreq); 
			adder.Initialize(adderBit, numAdder);
			if (numReadPulse > 1) {
				shiftAdd.Initialize(numAdder, adderBit+1, clkFreq, spikingMode, numReadPulse);
			}
		} else if (conventionalParallel) { 
		
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg * numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			
			if (numReadPulse > 1) {
				shiftAdd.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+1, clkFreq, spikingMode, numReadPulse);
			}
			
		} else if (BNNsequentialMode || XNORsequentialMode) {       
			double resTg = cell.resMemCellOn;
			int numAdder = (int)ceil(numCol/numColMuxed);  
			int numInput = numAdder;        
			int adderBit = (int)ceil(log2(numRow)) + 1; 
			
			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);           
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.Initialize(numRow);          
			} else {
				wlDecoderDriver.Initialize(ROW_MODE, numRow, numCol);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);     //SL use switch matrix
			if (numColMuxed>1) {
				mux.Initialize(numInput, numColMuxed, resTg, FPGA);      
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			rowCurrentSenseAmp.Initialize(numCol/numColMuxed, true, false, clkFreq, numReadCellPerOperationNeuro);
			dff.Initialize((adderBit+1)*numAdder, clkFreq); 
			adder.Initialize(adderBit, numAdder);
		} else if (BNNparallelMode || XNORparallelMode) {      
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed/2)), true, true);    
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
		} else {
			double resTg = cell.resMemCellOn / numRow;
			
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);         
			} else {
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resTg*numRow/numCol, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			}
			slSwitchMatrix.Initialize(COL_MODE, numCol, resTg*numRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);     
			if (numColMuxed>1) {
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resTg, FPGA);      
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			if (numReadPulse > 1) {
				shiftAdd.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+1, clkFreq, spikingMode, numReadPulse);
			}
		}
	} 
	initialized = true;  //finish initialization
}



void SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		usedArea = 0;
		if (cell.memCellType == Type::SRAM) {       
			// Array only
			heightArray = lengthCol;
			widthArray = lengthRow;
			areaArray = heightArray * widthArray;
			
			//precharger and writeDriver are always needed for all different designs
			precharger.CalculateArea(NULL, widthArray, NONE);
			sramWriteDriver.CalculateArea(NULL, widthArray, NONE);
			
			if (conventionalSequential) {
				wlDecoder.CalculateArea(heightArray, NULL, NONE);  
				senseAmp.CalculateArea(NULL, widthArray, MAGIC);
				adder.CalculateArea(NULL, widthArray, NONE);
				dff.CalculateArea(NULL, widthArray, NONE);
				if (numReadPulse > 1) {
					shiftAdd.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height + shiftAdd.height;
				width = wlDecoder.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area + shiftAdd.area;
				emptyArea = area - usedArea;
				
				areaADC = senseAmp.area + precharger.area;
				areaAccum = adder.area + dff.area + shiftAdd.area;
				areaOther = wlDecoder.area + sramWriteDriver.area;
			} else if (conventionalParallel) { 
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (numColMuxed>1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAdd.height + ((numColMuxed > 1)==true? (mux.height):0)+sarADC.height;
				width = MAX(wlSwitchMatrix.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAdd.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0)+sarADC.area;
				emptyArea = area - usedArea;
				
				areaADC = multilevelSenseAmp.area + precharger.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAdd.area;
				areaOther = wlSwitchMatrix.area + sramWriteDriver.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0);
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculateArea(heightArray, NULL, NONE);  
				senseAmp.CalculateArea(NULL, widthArray, MAGIC);
				adder.CalculateArea(NULL, widthArray, NONE);
				dff.CalculateArea(NULL, widthArray, NONE);
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height + shiftAdd.height;
				width = wlDecoder.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area + shiftAdd.area;
				emptyArea = area - usedArea;
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = wlSwitchMatrix.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			} else {
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAdd.height + sarADC.height;
				width = wlSwitchMatrix.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + shiftAdd.area + sarADC.area;
				emptyArea = area - usedArea;
			}
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			// Array only
			heightArray = lengthCol;
			widthArray = lengthRow;
			areaArray = heightArray * widthArray;
			
			if (conventionalSequential) {  
				wlDecoder.CalculateArea(heightArray, NULL, NONE);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
					}
				}
				
				dff.CalculateArea(NULL, widthArray, NONE);
				adder.CalculateArea(NULL, widthArray, NONE);
				if (numReadPulse > 1) {
					shiftAdd.CalculateArea(NULL, widthArray, NONE);
				}
				height = slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + multilevelSenseAmp.height + multilevelSAEncoder.height + adder.height + dff.height + shiftAdd.height + sarADC.height;
				width = MAX(wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + multilevelSenseAmp.area + multilevelSAEncoder.area + adder.area + dff.area + shiftAdd.area + sarADC.area;
				emptyArea = area - usedArea;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = adder.area + dff.area + shiftAdd.area;
				areaOther = wlDecoder.area + wlNewDecoderDriver.area + wlDecoderDriver.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0);
			} else if (conventionalParallel) { 
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAdd.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + multilevelSenseAmp.area  + multilevelSAEncoder.area + shiftAdd.area + sarADC.area;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAdd.area;
				areaOther = wlNewSwitchMatrix.area + wlSwitchMatrix.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0);
				
				area = height * width;
				emptyArea = area - usedArea;
			} else if (BNNsequentialMode || XNORsequentialMode) {       
				wlDecoder.CalculateArea(heightArray, NULL, NONE);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlDecoderDriver.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				rowCurrentSenseAmp.CalculateUnitArea();
				rowCurrentSenseAmp.CalculateArea(widthArray);
				
				dff.CalculateArea(NULL, widthArray, NONE);
				adder.CalculateArea(NULL, widthArray, NONE);
				
				height = slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + rowCurrentSenseAmp.height + adder.height + dff.height;
				width = MAX(wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + rowCurrentSenseAmp.area  + adder.area + dff.area;
				emptyArea = area - usedArea;
			} else if (BNNparallelMode || XNORparallelMode) {      
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			} else {   
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				} else {
					wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				}
				slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
				if (numColMuxed > 1) {
					mux.CalculateArea(NULL, widthArray, NONE);
					muxDecoder.CalculateArea(NULL, NULL, NONE);
					double minMuxHeight = MAX(muxDecoder.height, mux.height);
					mux.CalculateArea(minMuxHeight, widthArray, OVERRIDE);
				}
				if (SARADC) {
					sarADC.CalculateUnitArea();
					sarADC.CalculateArea(NULL, widthArray, NONE);
				} else {
					multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);
					multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);
				}
				
				height = slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height;
				width = MAX(wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + sarADC.area;
				emptyArea = area - usedArea;
			}
			
		} 
	}
}

void SubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		readLatency = 0;
		writeLatency = 0;

		if (cell.memCellType == Type::SRAM) {
			if (conventionalSequential) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				wlDecoder.CalculateLatency(1e20, capRow1, NULL, numRow*activityRowRead, numRow*activityRowWrite);
				
				precharger.CalculateLatency(1e20, capCol, numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				senseAmp.CalculateLatency(numReadOperationPerRow*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numReadOperationPerRow*numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numReadOperationPerRow*numRow*activityRowRead);
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(1);	
				}
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency += wlDecoder.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += senseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				readLatency += shiftAdd.readLatency;
				
				readLatencyADC = precharger.readLatency + colDelay + senseAmp.readLatency;
				readLatencyAccum = adder.readLatency + dff.readLatency + shiftAdd.readLatency;
				readLatencyOther = wlDecoder.readLatency;
				
				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);

				// writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
				// writeLatency += wlDecoder.writeLatency;
				// writeLatency += precharger.writeLatency;
				// writeLatency += sramWriteDriver.writeLatency;
				
			} else if (conventionalParallel) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(0, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(numColMuxed);	
				}
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlSwitchMatrix.rampOutput, &colRamp);

				readLatency = 0;
				readLatency += MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += precharger.readLatency;
				readLatency += colDelay/numReadPulse;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAdd.readLatency;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = precharger.readLatency + colDelay + multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = shiftAdd.readLatency;
				readLatencyOther = MAX(wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow1, NULL, numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				senseAmp.CalculateLatency(numReadOperationPerRow*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numReadOperationPerRow*numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numReadOperationPerRow*numRow*activityRowRead);
				
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency += wlDecoder.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += senseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				
				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
			} else if (BNNparallelMode || XNORparallelMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlSwitchMatrix.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency = 0;
				readLatency += wlSwitchMatrix.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += sarADC.readLatency;

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
			} else {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				precharger.CalculateLatency(1e20, capCol, numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(1);	
				}
				// Read
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double tau = (resCellAccess + resPullDown) * (capCellAccess + capCol) + resCol * capCol / 2;
				tau *= log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));   
				double gm = CalculateTransconductance(cell.widthAccessCMOS * tech.featureSize, NMOS, tech);
				double beta = 1 / (resPullDown * gm);
				double colRamp = 0;
				colDelay = horowitz(tau, beta, wlDecoder.rampOutput, &colRamp) * numReadOperationPerRow * numRow * activityRowRead;

				readLatency = 0;
				readLatency += wlSwitchMatrix.readLatency;
				readLatency += precharger.readLatency;
				readLatency += colDelay;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAdd.readLatency;
				readLatency += sarADC.readLatency;

				// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
				double resPull;
				resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
				tau = resPull * cell.capSRAMCell;
				gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
				beta = 1 / (resPull * gm);
				
			}
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			if (conventionalSequential) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp)*numColMuxed;	// Just to generate colRamp
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow2, NULL, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
				} else {
					wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed*numRow*activityRowRead);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, numRow*activityRowRead);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
					}
				}
				adder.CalculateLatency(1e20, dff.capTgDrain, numColMuxed*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(numColMuxed);	// There are numReadPulse times of shift-and-add
				}
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += colDelay/numReadPulse;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				readLatency += shiftAdd.readLatency;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = adder.readLatency + dff.readLatency + shiftAdd.readLatency;
				readLatencyOther = MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse) + colDelay/numReadPulse;
				
				// // Write
				// writeLatency = 0;
				// writeLatencyArray = 0;
				// writeLatencyArray += numWritePulse * param->writePulseWidthLTP;
				// writeLatencyArray += (-numErasePulse) * param->writePulseWidthLTD;
				// writeLatency += MAX(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency + wlDecoderDriver.writeLatency, slSwitchMatrix.writeLatency);
				// writeLatency += writeLatencyArray;
				
			} else if (conventionalParallel) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp)*numColMuxed;
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed>1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(numColMuxed);	
				}
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += colDelay/numReadPulse;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAdd.readLatency;
				readLatency += sarADC.readLatency;
				
				readLatencyADC = multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency;
				readLatencyAccum = shiftAdd.readLatency;
				readLatencyOther = MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse) + colDelay/numReadPulse;
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp)*numColMuxed;
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				
				wlDecoder.CalculateLatency(1e20, capRow2, NULL, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
				} else {
					wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				rowCurrentSenseAmp.CalculateLatency(columnResistance, numColMuxed, numRow*activityRowRead);
				adder.CalculateLatency(1e20, dff.capTgDrain, numColMuxed*numRow*activityRowRead);
				dff.CalculateLatency(1e20, numColMuxed*numRow*activityRowRead);
				
				// Read
				readLatency = 0;
				readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += colDelay/numReadPulse;
				readLatency += rowCurrentSenseAmp.readLatency;
				readLatency += adder.readLatency;
				readLatency += dff.readLatency;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp)*numColMuxed;
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}

				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += colDelay/numReadPulse;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += sarADC.readLatency;
				
			} else {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;
				double tau = (capCol)*(cell.resMemCellAvg/(numRow/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp)*numColMuxed;
				colDelay = tau * 0.2 * numColMuxed;  // assume the 15~20% voltage drop is enough for sensing
				
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculateLatency(colRamp, 0, numColMuxed);
					muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), numColMuxed, 0);
				}
				if (SARADC) {
					sarADC.CalculateLatency(numColMuxed);
				} else {
					multilevelSenseAmp.CalculateLatency(columnResistance, numColMuxed, 1);
					multilevelSAEncoder.CalculateLatency(1e20, numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculateLatency(numColMuxed);	
				}
				// Read
				readLatency = 0;
				readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ( ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) )/numReadPulse);
				readLatency += colDelay/numReadPulse;
				readLatency += multilevelSenseAmp.readLatency;
				readLatency += multilevelSAEncoder.readLatency;
				readLatency += shiftAdd.readLatency;
				readLatency += sarADC.readLatency;
			}
		}
	}
}

void SubArray::CalculatePower(const vector<double> &columnResistance) {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		readDynamicEnergyArray = 0;
		
		double numReadOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol > numReadCellPerOperationNeuro)
			numReadOperationPerRow = numCol / numReadCellPerOperationNeuro;
		else
			numReadOperationPerRow = 1;

		double numWriteOperationPerRow;   // average value (can be non-integer for energy calculation)
		if (numCol * activityColWrite > numWriteCellPerOperationNeuro)
			numWriteOperationPerRow = numCol * activityColWrite / numWriteCellPerOperationNeuro;
		else
			numWriteOperationPerRow = 1;

		if (cell.memCellType == Type::SRAM) {
			
			// Array leakage (assume 2 INV)
			leakage = 0;
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * tech.featureSize,
					cell.widthSRAMCellPMOS * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			leakage *= numRow * numCol;

			if (conventionalSequential) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar

				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				
				readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + senseAmp.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAdd.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy;

				// Write
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += senseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += shiftAdd.leakage;

			} else if (conventionalParallel) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numColMuxed);
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy/numReadPulse):0);
				readDynamicEnergy += ((numColMuxed > 1)==true? (muxDecoder.readDynamicEnergy/numReadPulse):0);
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = shiftAdd.readDynamicEnergy;
				readDynamicEnergyOther = wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				
				// Write
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAdd.leakage;
			
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse*(adder.numBit+1));
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar

				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				
				// Write				
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += senseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				// Write				
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				
			} else {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				precharger.CalculatePower(numColMuxed, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numColMuxed);
				}
				// Array
				readDynamicEnergyArray = 0; // Just BL discharging
				writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * 2 * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// Write				
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += precharger.writeDynamicEnergy;
				writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;			
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAdd.leakage;
			
			}
			
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			if (conventionalSequential) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				double numWriteCells = (int)ceil((double)numCol/*numWriteColMuxed*/); 
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
				
				wlDecoder.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numRow*activityRowRead);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numRow*activityRowRead);
					if (avgWeightBit > 1) {
						multilevelSAEncoder.CalculatePower(numRow*activityRowRead*numColMuxed);
					}
				}
				adder.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells);
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(adder.numBit+1)); 
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numColMuxed);	// There are numReadPulse times of shift-and-add
				}
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
				readDynamicEnergyArray *= numRow * activityRowRead * numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += wlNewDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += wlDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAdd.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy + wlDecoderDriver.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;

				// Write				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;
				
				// Leakage
				leakage = 0;
				leakage += wlDecoder.leakage;
				leakage += wlDecoderDriver.leakage;
				leakage += wlNewDecoderDriver.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += shiftAdd.leakage;
					
			} else if (conventionalParallel) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numColMuxed);
				}
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;
				
				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = shiftAdd.readDynamicEnergy;
				readDynamicEnergyOther = wlNewSwitchMatrix.readDynamicEnergy + wlSwitchMatrix.readDynamicEnergy + ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				
				// Write				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAdd.leakage;
				
			} else if (BNNsequentialMode || XNORsequentialMode) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				double numWriteCells = (int)ceil((double)numCol/*numWriteColMuxed*/); 
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				wlDecoder.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				if (cell.accessType == CMOS_access) {
					wlNewDecoderDriver.CalculatePower(numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				} else {
					wlDecoderDriver.CalculatePower(numReadCells, numWriteCells, numRow*activityRowRead*numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				rowCurrentSenseAmp.CalculatePower(columnResistance, numRow*activityRowRead);
				adder.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells);
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(adder.numBit+1)); 
				
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
				readDynamicEnergyArray *= numRow * activityRowRead * numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += wlNewDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += wlDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += rowCurrentSenseAmp.readDynamicEnergy;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;

				// Write				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				leakage += wlDecoder.leakage;
				leakage += wlDecoderDriver.leakage;
				leakage += wlNewDecoderDriver.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += rowCurrentSenseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// Write				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;

			} else {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;
			
				if (cell.accessType == CMOS_access) {
					wlNewSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
				} else {
					wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				}
				slSwitchMatrix.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, 1);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, 1);
					multilevelSAEncoder.CalculatePower(numColMuxed);
				}
				if (numReadPulse > 1) {
					shiftAdd.CalculatePower(numColMuxed);
				}
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;
				
				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += ( ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0) )/numReadPulse;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAdd.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				// Write				
				writeDynamicEnergyArray = writeDynamicEnergyArray;
				writeDynamicEnergy = 0;
				writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAdd.leakage;
			}
		} 
	}
}

void SubArray::PrintProperty() {

	if (cell.memCellType == Type::SRAM) {
		
		cout << endl << endl;
	    cout << "Array:" << endl;
	    cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	    cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	    cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
		
		precharger.PrintProperty("precharger");
		sramWriteDriver.PrintProperty("sramWriteDriver");
		
		if (conventionalSequential) {
			wlDecoder.PrintProperty("wlDecoder");			
			senseAmp.PrintProperty("senseAmp");
			dff.PrintProperty("dff"); 
			adder.PrintProperty("adder");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		} else if (conventionalParallel) {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.PrintProperty("wlDecoder");			
			senseAmp.PrintProperty("senseAmp");
			dff.PrintProperty("dff"); 
			adder.PrintProperty("adder");
		} else if (BNNparallelMode || XNORparallelMode) {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
		} else {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		}
		
	} else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
		
		cout << endl << endl;
	    cout << "Array:" << endl;
	    cout << "Area = " << heightArray*1e6 << "um x " << widthArray*1e6 << "um = " << areaArray*1e12 << "um^2" << endl;
	    cout << "Read Dynamic Energy = " << readDynamicEnergyArray*1e12 << "pJ" << endl;
	    cout << "Write Dynamic Energy = " << writeDynamicEnergyArray*1e12 << "pJ" << endl;
		cout << "Write Latency = " << writeLatencyArray*1e9 << "ns" << endl;

		if (conventionalSequential) {
			wlDecoder.PrintProperty("wlDecoder");
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.PrintProperty("wlNewDecoderDriver");
			} else {
				wlDecoderDriver.PrintProperty("wlDecoderDriver");
			} 
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp or single-bit SenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			adder.PrintProperty("adder");
			dff.PrintProperty("dff");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		} else if (conventionalParallel) {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		} else if (BNNsequentialMode || XNORsequentialMode) {
			wlDecoder.PrintProperty("wlDecoder");
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.PrintProperty("wlNewDecoderDriver");
			} else {
				wlDecoderDriver.PrintProperty("wlDecoderDriver");
			} 
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			rowCurrentSenseAmp.PrintProperty("currentSenseAmp");
			adder.PrintProperty("adder");
			dff.PrintProperty("dff");
		} else if (BNNparallelMode || XNORparallelMode) {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
		} else {
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.PrintProperty("wlNewSwitchMatrix");
			} else {
				wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			}
			slSwitchMatrix.PrintProperty("slSwitchMatrix");
			mux.PrintProperty("mux");
			muxDecoder.PrintProperty("muxDecoder");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAdd.PrintProperty("shiftAdd");
			}
		}
	} 
	FunctionUnit::PrintProperty("SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

