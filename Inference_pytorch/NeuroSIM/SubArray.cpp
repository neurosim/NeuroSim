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
						wllevelshifter(_inputParameter, _tech, _cell),
						sllevelshifter(_inputParameter, _tech, _cell),
						bllevelshifter(_inputParameter, _tech, _cell),
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
						wlDecoderDriver(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell),
						adder(_inputParameter, _tech, _cell),
						dff(_inputParameter, _tech, _cell),
						shiftAddInput(_inputParameter, _tech, _cell),
						shiftAddWeight(_inputParameter, _tech, _cell),
						multilevelSenseAmp(_inputParameter, _tech, _cell),
						multilevelSAEncoder(_inputParameter, _tech, _cell),
						sarADC(_inputParameter, _tech, _cell){
	initialized = false;
	readDynamicEnergyArray = writeDynamicEnergyArray = 0;
} 

void SubArray::Initialize(int _numRow, int _numCol, double _unitWireRes){  //initialization module
	
	numRow = _numRow;    // import parameters
	numCol = _numCol;
	unitWireRes = _unitWireRes;
	
	double MIN_CELL_HEIGHT = MAX_TRANSISTOR_HEIGHT;  //set real layout cell height
	double MIN_CELL_WIDTH = (MIN_GAP_BET_GATE_POLY + POLY_WIDTH) * 2;  //set real layout cell width
	double ISOLATION_REGION = MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY; // 1.4 update : new variable

	// 1.4 update : new cell dimension setting

	if (tech.featureSize == 14 * 1e-9){
	MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_14nm/MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_14nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 10 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_10nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_10nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 7 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_7nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_7nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 5 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_5nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_5nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 3 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_3nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_3nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 2 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_2nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_2nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else if (tech.featureSize == 1 * 1e-9){
    MIN_CELL_HEIGHT *= (MAX_TRANSISTOR_HEIGHT_1nm /MAX_TRANSISTOR_HEIGHT);
	ISOLATION_REGION *= (OUTER_HEIGHT_REGION_1nm/(MIN_POLY_EXT_DIFF*2 + MIN_GAP_BET_FIELD_POLY));}
    else{
    MIN_CELL_HEIGHT *= 1;
	ISOLATION_REGION *=1;}

	if (tech.featureSize == 14 * 1e-9)
	MIN_CELL_WIDTH  *= ((POLY_WIDTH_FINFET + MIN_GAP_BET_GATE_POLY_FINFET )/(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 10 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_10nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 7 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_7nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 5 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_5nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 3 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_3nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 2 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_2nm /(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else if (tech.featureSize == 1 * 1e-9)
    MIN_CELL_WIDTH  *= (CPP_1nm/(MIN_GAP_BET_GATE_POLY + POLY_WIDTH));
    else
    MIN_CELL_WIDTH  *= 1;

	// 1.4 update : think about the factor multiplied by the cell width/height for the relaxation 

	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		if (relaxArrayCellWidth) {  //if want to relax the cell width

			// 1.4 update: For SRAM, the height corresponds to the width of the logic layout 

			lengthRow = (double)numCol * MAX(cell.widthInFeatureSize, MIN_CELL_HEIGHT) * tech.featureSize;
		} else { //if not relax the cell width
			lengthRow = (double)numCol * cell.widthInFeatureSize * tech.featureSize;
		}
		if (relaxArrayCellHeight) {  //if want to relax the cell height

			// 1.4 update: For SRAM, the height corresponds to the width of the logic layout 

			lengthCol = (double)numRow * MAX(cell.heightInFeatureSize, MIN_CELL_WIDTH) * tech.featureSize;
		} else {  //if not relax the cell height
			lengthCol = (double)numRow * cell.heightInFeatureSize * tech.featureSize;
		}

		// 230920 update
		param->arraywidthunit = cell.widthInFeatureSize * tech.featureSize;
		param->arrayheight = (double)numRow * cell.heightInFeatureSize * cell.featureSize;


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

		// 230920 update
		param->arraywidthunit = cellWidth * cell.featureSize;
		param->arrayheight = (double)numRow * cellHeight * cell.featureSize;

	}      //finish setting array size
	
	// 1.4 update
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;

	resRow = lengthRow * param->Metal1_unitwireresis; 
	resCol = lengthCol * param->Metal0_unitwireresis;
	
	
	param->columncap = capCol;
	//start to initializing the subarray modules
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
		
		//firstly calculate the CMOS resistance and capacitance

		// 1.4 update : modified the code - no folding for SRAM

		// 1.4 update: needs check  - capCol 

		resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);

		capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech);
		cell.capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateDrainCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, PMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateGateCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech);

		// 1.4 update: for buffer insertion
		double unitcap= capRow1/param->numColSubArray;
		double unitres= resRow/param->numColSubArray;
		param->unitcap = unitcap;
		param->unitres = unitres;	

		if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;
		else capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;	
		param->columncap = capCol;
		if (conventionalSequential) {
		// 1.4 update: consider SRAM parasitic cap
		capRow1 += 2*CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap

			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			int adderBit = (int)ceil(log2(numRow)) + 1;				
			int numAdder = numCol/numCellPerSynapse; 
			// Anni update: no mux, so adder and shiftaddweight should be for all columns
			dff.Initialize(adderBit*numCol, clkFreq);	
			adder.Initialize(adderBit-1, numCol, clkFreq);
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit + numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}
			
		} else if (conventionalParallel) {
		// 1.4 update: consider SRAM parasitic cap - only one WL is activated for ADC 
		capRow1 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap

			// 1.4 update : buffer insertion
			if (param->buffernumber > 0) {

				sectionres = resRow / (param->buffernumber +1);
				targetdriveres = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize*param->buffersizeratio, NMOS, inputParameter.temperature, tech) ;

				widthInvN  = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, NMOS, inputParameter.temperature, tech) / targetdriveres * tech.featureSize;
				widthInvP = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, PMOS, inputParameter.temperature, tech) / targetdriveres * tech.featureSize ;

				if (tech.featureSize <= 14*1e-9){
					widthInvN = 2* ceil(widthInvN/tech.featureSize) * tech.featureSize;
					widthInvP = 2* ceil(widthInvP/tech.featureSize) * tech.featureSize;
				}				
					

				wlSwitchMatrix.Initialize(ROW_MODE, numRow, sectionres, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			} else {								  
				wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			} 
			if (numColMuxed>1) {
				// 1.4 update: store access resistance for multilevelsenseamp
				param->resCellAccess=resCellAccess;
				// 1.4 update: half turn on assume;	Anni update: numRow->numRowParallel
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resCellAccess/(numRowParallel/2), FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}		
			// Anni update: add partial sums	
			int adderBit = log2(levelOutput) + ceil(log2(numAdd));	
			int numAdder = ceil(numCol/numColMuxed);
			if (numAdd > 1) {				
				dff.Initialize(adderBit*numAdder, clkFreq);	
				adder.Initialize(adderBit-1, numAdder, clkFreq);
			}	
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit + numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
			}					
		} else if (BNNsequentialMode || XNORsequentialMode) {
		// 1.4 update: consider SRAM parasitic cap
		capRow1 += 2*CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap

			wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
			senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numReadCellPerOperationNeuro);
			// Anni update
			int adderBit = (int)ceil(log2(numRow)) + 1;	
			int numAdder = numCol;
			dff.Initialize(adderBit*numAdder, clkFreq);	
			adder.Initialize(adderBit-1, numAdder, clkFreq);
		} else if (BNNparallelMode || XNORparallelMode) {
		// 1.4 update: consider SRAM parasitic cap - only one WL is activated for ADC 
		capRow1 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap

			wlSwitchMatrix.Initialize(ROW_MODE, numRow, resRow, true, false, activityRowRead, activityColWrite, numWriteCellPerOperationMemory, numWriteCellPerOperationNeuro, 1, clkFreq);
			// Anni update: add mux
			if (numColMuxed>1) {
				// 1.4 update: half turn on assume;	Anni update: numRow->numRowParallel
				mux.Initialize(ceil(numCol/numColMuxed), numColMuxed, resCellAccess/(numRowParallel/2), FPGA);       
				muxDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numColMuxed)), true, false);
			}
			if (SARADC) {
				sarADC.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro);
			} else {
				multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, true, currentMode);
				multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);
			}
			// Anni update: add partial sums
			int adderBit = log2(levelOutput) + ceil(log2(numAdd));	
			int numAdder = ceil(numCol/numColMuxed);
			if (numAdd > 1) {				
				dff.Initialize(adderBit*numAdder, clkFreq);	
				adder.Initialize(adderBit-1, numAdder, clkFreq);
			}
		}
		precharger.Initialize(numCol, resCol, activityColWrite, numReadCellPerOperationNeuro, numWriteCellPerOperationNeuro);
		sramWriteDriver.Initialize(numCol, activityColWrite, numWriteCellPerOperationNeuro);
		
    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
		if (cell.accessType == CMOS_access) {	// 1T1R

			cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
			cell.widthAccessCMOS = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, NMOS, inputParameter.temperature, tech) * LINEAR_REGION_RATIO / cell.resCellAccess;   //get access CMOS width
			double widthAccessInFeatureSize = cell.widthAccessCMOS;
			if (tech.featureSize <= 14 * 1e-9){			
				widthAccessInFeatureSize = ((cell.widthAccessCMOS-1) * tech.PitchFin + tech.widthFin) / tech.featureSize;  //convert #fin to F
			}
			if (widthAccessInFeatureSize > cell.widthInFeatureSize) {	// Place transistor vertically
				printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
				exit(-1);
			}
			cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;        //calculate single memory cell resistance_ON
			cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF

			// 1.4 update; Anni update: numRow->numRowParallel
			cell.resMemCellAvg = 1/(1/(cell.resistanceOn + cell.resCellAccess ) * numRowParallel/2.0 + 1/(cell.resistanceOff + cell.resCellAccess )* numRowParallel/2.0) * numRowParallel;      //calculate single memory cell resistance_AVG

			// 1.4 update: needs check  - capCol : cell.widthInFeatureSize / 
			capRow2 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
			// 1.4 update
			double unitcap= capRow2/param->numColSubArray;
			double unitres= resRow/param->numColSubArray;
			param->unitcap = unitcap;
			param->unitres = unitres;			
			
			capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit
			param->columncap = capCol;
		} else {	// Cross-point
			cell.resMemCellOn = cell.resistanceOn;
			cell.resMemCellOff = cell.resistanceOff;
			cell.resMemCellOnAtHalfVw = cell.resistanceOn;
			cell.resMemCellOffAtHalfVw = cell.resistanceOff;
			cell.resMemCellOnAtVw = cell.resistanceOn;
			cell.resMemCellOffAtVw = cell.resistanceOff;

			// 1.4 update; Anni update: numRow->numRowParallel
			cell.resMemCellAvg = 1/(1/(cell.resistanceOn + cell.resCellAccess ) * numRowParallel/2.0 + 1/(cell.resistanceOff + cell.resCellAccess )* numRowParallel/2.0) * numRowParallel;      //calculate single memory cell resistance_AVG

			cell.resMemCellAvgAtHalfVw = cell.resistanceAvg;
			cell.resMemCellAvgAtVw = cell.resistanceAvg;
		}
		
		if (cell.writeVoltage > 1.5) {
			wllevelshifter.Initialize(numRow, activityRowRead, clkFreq);
			bllevelshifter.Initialize(numRow, activityRowRead, clkFreq);
			sllevelshifter.Initialize(numCol, activityColWrite, clkFreq);
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
			// Anni update
			dff.Initialize(adderBit*numAdder, clkFreq); 
			adder.Initialize(adderBit-1, numAdder, clkFreq);
			// Anni update: shift avgWeightBit for (numCellPerSynapse-1) times, +1 to compensate -1 inside shift-add which designed for general 1-bit cell
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, (numCellPerSynapse-1)*avgWeightBit+1);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit + (numCellPerSynapse-1)*avgWeightBit+1, clkFreq, spikingMode, numReadPulse);
			}
			
		} else if (conventionalParallel) { 
			// 1.4 update: needs check enabled rows?;	Anni update: numRow -> numRowParallel
			double resTg = cell.resMemCellAvg / numRowParallel;
			if (cell.accessType == CMOS_access) {
				wlNewSwitchMatrix.Initialize(numRow, activityRowRead, clkFreq);
				// 1.4 update: buffer insertion
				if (param->buffernumber>0) {
					sectionres = resRow / (param->buffernumber +1);
					targetdriveres = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize*param->buffersizeratio, NMOS, inputParameter.temperature, tech) ;

					widthInvN  = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, NMOS, inputParameter.temperature, tech) / targetdriveres * tech.featureSize;
					widthInvP = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, PMOS, inputParameter.temperature, tech) / targetdriveres * tech.featureSize ;

					if (tech.featureSize <= 14*1e-9){
						widthInvN = 2* ceil(widthInvN/tech.featureSize) * tech.featureSize;
						widthInvP = 2* ceil(widthInvP/tech.featureSize) * tech.featureSize;
					}															
				}
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
			// Anni update: add partial sums
			int adderBit = log2(levelOutput) + ceil(log2(numAdd));
			int numAdder = ceil(numCol/numColMuxed);
			if (numAdd > 1) {				
				dff.Initialize(adderBit*numAdder, clkFreq);	
				adder.Initialize(adderBit-1, numAdder, clkFreq);
			}
			// Anni update: shift avgWeightBit for (numCellPerSynapse-1) times, +1 to compensate -1 inside shift-add which designed for general 1-bit cell
			// needs check
			/*
			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(numAdder, adderBit, clkFreq, spikingMode, (numCellPerSynapse-1)*avgWeightBit+1);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(numAdder, adderBit + (numCellPerSynapse-1)*avgWeightBit+1, clkFreq, spikingMode, numReadPulse);
			}	
			*/					

			if (numCellPerSynapse > 1) {
				shiftAddWeight.Initialize(ceil(numCol/numColMuxed), log2(levelOutput), clkFreq, spikingMode, numCellPerSynapse);
			}
			if (numReadPulse > 1) {
				shiftAddInput.Initialize(ceil(numCol/numColMuxed), log2(levelOutput)+numCellPerSynapse, clkFreq, spikingMode, numReadPulse);
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

			// 1.4 update 230615
			multilevelSenseAmp.Initialize(numCol/numColMuxed, levelOutput, clkFreq, numReadCellPerOperationNeuro, false, currentMode);
			multilevelSAEncoder.Initialize(levelOutput, numCol/numColMuxed);

			// rowCurrentSenseAmp.Initialize(numCol/numColMuxed, true, false, clkFreq, numReadCellPerOperationNeuro);
			
			// Anni update
			dff.Initialize(adderBit*numAdder, clkFreq); 
			adder.Initialize(adderBit-1, numAdder, clkFreq);
		} else if (BNNparallelMode || XNORparallelMode) {      
			// 1.4 update: needs check enabled rows?;	Anni update: numRow->numRowParallel
			double resTg = cell.resMemCellAvg / numRowParallel;
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
			// Anni update: add partial sums
			int adderBit = log2(levelOutput) + ceil(log2(numAdd));	
			int numAdder = ceil(numCol/numColMuxed);
			if (numAdd > 1) {				
				dff.Initialize(adderBit*numAdder, clkFreq);	
				adder.Initialize(adderBit-1, numAdder, clkFreq);
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
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				// Anni update: DFF*2, for adder and shift-add weight pipeline
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height*2 + shiftAddInput.height + shiftAddWeight.height;
				width = wlDecoder.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area*2 + shiftAddInput.area + shiftAddWeight.area;
				emptyArea = area - usedArea;

				areaADC = senseAmp.area + precharger.area;
				areaAccum = adder.area + dff.area*2 + shiftAddInput.area + shiftAddWeight.area;
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
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				// Anni update: add partial sums, height & usedArea & areaAccum
				if (numAdd > 1) {
					adder.CalculateArea(NULL, widthArray, NONE);
					dff.CalculateArea(NULL, widthArray, NONE);
				}
				
				// 1.4 update : repeater implementation
				// buffer area
				if (param->buffernumber>0) {					
					CalculateGateArea(INV, 1, widthInvN , widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
					CalculateGateCapacitance(INV, 1, widthInvN , widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &drivecapin, &drivecapout);					
				} else {
					wInv = 0;
					hInv = 0;
				}
				// 1.4 update : repeater implementation
				double bufferarea= hInv * wInv * param->buffernumber * 2 * param->numRowSubArray;
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + \
						shiftAddInput.height + shiftAddWeight.height + ((numAdd > 1)==true? (adder.height+dff.height):0) + ((numColMuxed > 1)==true? (mux.height):0)+sarADC.height;
				width = MAX(wlSwitchMatrix.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray + bufferarea/lengthCol; // added for buffer area;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + \
						shiftAddInput.area + shiftAddWeight.area + ((numAdd > 1)==true? (adder.area+dff.area):0) + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0)+sarADC.area + bufferarea;
				emptyArea = area - usedArea;

				areaADC = multilevelSenseAmp.area + precharger.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAddInput.area + shiftAddWeight.area + ((numAdd > 1)==true? (adder.area+dff.area):0);
				areaOther = wlSwitchMatrix.area + sramWriteDriver.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + bufferarea;
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculateArea(heightArray, NULL, NONE);  
				senseAmp.CalculateArea(NULL, widthArray, MAGIC);
				adder.CalculateArea(NULL, widthArray, NONE);
				dff.CalculateArea(NULL, widthArray, NONE);
				height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height + adder.height + dff.height;
				width = wlDecoder.width + widthArray;
				area = height * width;
				usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area + adder.area + dff.area;
				emptyArea = area - usedArea;
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculateArea(heightArray, NULL, NONE);
				// Anni update: add mux
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
				// Anni update: add partial sums, height & usedArea
				if (numAdd > 1) {
					adder.CalculateArea(NULL, widthArray, NONE);
					dff.CalculateArea(NULL, widthArray, NONE);
				}
				height = precharger.height + sramWriteDriver.height + heightArray + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height + \
						((numColMuxed > 1)==true? (mux.height):0) + ((numAdd > 1)==true? (adder.height+dff.height):0);
				width = MAX(wlSwitchMatrix.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				area = height * width;
				usedArea = areaArray + wlSwitchMatrix.area + precharger.area + sramWriteDriver.area + multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area + \
							((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + ((numAdd > 1)==true? (adder.area+dff.area):0);
				emptyArea = area - usedArea;
			}
			
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			// Array only
			heightArray = lengthCol;
			widthArray = lengthRow;
			areaArray = heightArray * widthArray;
			
			// Level shifter for write
			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculateArea(heightArray, NULL, NONE);
				bllevelshifter.CalculateArea(heightArray, NULL, NONE);
				sllevelshifter.CalculateArea(NULL, widthArray, NONE);				
			}
			
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
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				height = ((cell.writeVoltage > 1.5)==true? (sllevelshifter.height):0) + slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + \
						multilevelSenseAmp.height + multilevelSAEncoder.height + adder.height + dff.height + shiftAddInput.height + shiftAddWeight.height + sarADC.height;
				width = MAX( ((cell.writeVoltage > 1.5)==true? (wllevelshifter.width + bllevelshifter.width):0) + wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, ((numColMuxed > 1)==true? (muxDecoder.width):0) ) + widthArray;
				area = height * width;
				usedArea = areaArray + ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + 
							((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + multilevelSenseAmp.area + multilevelSAEncoder.area + adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area + sarADC.area;
				emptyArea = area - usedArea;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = adder.area + dff.area + shiftAddInput.area + shiftAddWeight.area;
				areaOther = ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlDecoder.area + wlNewDecoderDriver.area + wlDecoderDriver.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0);
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
					shiftAddInput.CalculateArea(NULL, widthArray, NONE);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculateArea(NULL, widthArray, NONE);
				}
				// Anni update: add partial sums, height & usedArea & areaAccum
				if (numAdd > 1) {
					adder.CalculateArea(NULL, widthArray, NONE);
					dff.CalculateArea(NULL, widthArray, NONE);
				}
				
				// 1.4 update : buffer area
				// buffer area
				if (param->buffernumber>0) {
					CalculateGateArea(INV, 1, widthInvN , widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &hInv, &wInv);
					CalculateGateCapacitance(INV, 1, widthInvN , widthInvP, tech.featureSize * MAX_TRANSISTOR_HEIGHT, tech, &drivecapin, &drivecapout);					
				} else {
					wInv = 0;
					hInv = 0;
				}
				double bufferarea= hInv * wInv * param->buffernumber * 2 * param->numRowSubArray;
				
				height = ((cell.writeVoltage > 1.5)==true? (sllevelshifter.height):0) + slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + \
						multilevelSenseAmp.height + multilevelSAEncoder.height + shiftAddWeight.height + shiftAddInput.height + ((numAdd > 1)==true? (adder.height+dff.height):0) + sarADC.height;
				width = MAX( ((cell.writeVoltage > 1.5)==true? (wllevelshifter.width + bllevelshifter.width):0) + wlNewSwitchMatrix.width + wlSwitchMatrix.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray + bufferarea/lengthCol; // added;
				usedArea = areaArray + ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + 
							((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + multilevelSenseAmp.area  + multilevelSAEncoder.area + shiftAddWeight.area + shiftAddInput.area + ((numAdd > 1)==true? (adder.area+dff.area):0) + sarADC.area + bufferarea;
				
				areaADC = multilevelSenseAmp.area + multilevelSAEncoder.area + sarADC.area;
				areaAccum = shiftAddWeight.area + shiftAddInput.area + ((numAdd > 1)==true? (adder.area+dff.area):0);
				areaOther = ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlNewSwitchMatrix.area + wlSwitchMatrix.area + slSwitchMatrix.area + ((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + bufferarea;
				
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

				// 1.4 update 230615

				// rowCurrentSenseAmp.CalculateUnitArea();
				// rowCurrentSenseAmp.CalculateArea(widthArray);
				multilevelSenseAmp.CalculateArea(NULL, widthArray, NONE);	
				multilevelSAEncoder.CalculateArea(NULL, widthArray, NONE);

				dff.CalculateArea(NULL, widthArray, NONE);
				adder.CalculateArea(NULL, widthArray, NONE);
				
				height = ((cell.writeVoltage > 1.5)==true? (sllevelshifter.height):0) + slSwitchMatrix.height + heightArray + ((numColMuxed > 1)==true? (mux.height):0) + multilevelSAEncoder.height + multilevelSenseAmp.height + adder.height + dff.height;
				width = MAX( ((cell.writeVoltage > 1.5)==true? (wllevelshifter.width + bllevelshifter.width):0) + wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width, ((numColMuxed > 1)==true? (muxDecoder.width):0)) + widthArray;
				area = height * width;
				// 1.4 update 230615s
				usedArea = areaArray + ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area + 
							((numColMuxed > 1)==true? (mux.area + muxDecoder.area):0) + multilevelSenseAmp.area + multilevelSAEncoder.area  + adder.area + dff.area;
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
				// Anni update: add partial sums, height & usedArea
				if (numAdd > 1) {
					adder.CalculateArea(NULL, widthArray, NONE);
					dff.CalculateArea(NULL, widthArray, NONE);
				}
				height = ((cell.writeVoltage > 1.5)==true? (sllevelshifter.height):0) + slSwitchMatrix.height + heightArray + mux.height + multilevelSenseAmp.height + multilevelSAEncoder.height + sarADC.height + ((numAdd > 1)==true? (adder.height+dff.height):0);
				width = MAX( ((cell.writeVoltage > 1.5)==true? (wllevelshifter.width + bllevelshifter.width):0) + wlNewSwitchMatrix.width + wlSwitchMatrix.width, muxDecoder.width) + widthArray;
				area = height * width;
				usedArea = areaArray + ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area + bllevelshifter.area + sllevelshifter.area):0) + wlSwitchMatrix.area + wlNewSwitchMatrix.area + slSwitchMatrix.area + 
							mux.area + multilevelSenseAmp.area + muxDecoder.area + multilevelSAEncoder.area + sarADC.area + ((numAdd > 1)==true? (adder.area+dff.area):0);
				emptyArea = area - usedArea;
			}
		} 
	}
}

void SubArray::CalculateLatency(double columnRes, const vector<double> &columnResistance, bool CalculateclkFreq) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		readLatency = 0;
		readLatencyADC = 0;
		readLatencyAccum = 0;
		readLatencyOther = 0;
		writeLatency = 0;

		if (cell.memCellType == Type::SRAM) {
			if (conventionalSequential) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {

					// 1.4 update: new arguments for rowdecoder
					wlDecoder.CalculateLatency(1e20, capRow1, NULL, resRow, numCol, 1, numRow*activityRowWrite);				
					precharger.CalculateLatency(1e20, capCol, 1, numWriteOperationPerRow*numRow*activityRowWrite);					
					senseAmp.CalculateLatency(1);

					// Read
					// 1.4 update: SRAM column cap update (calibration with IMEC data)
					// 1.4 update: SRAM row delay update
					// needs check

					double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
					double BLCap_perCell = capCol / numRow + capCellAccess; // Anni update
					double BLRes_perCell = resCol / numRow;
					double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;

					colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));  

					if (CalculateclkFreq) {
						readLatency += wlDecoder.readLatency;
						readLatency += precharger.readLatency;
						readLatency += colDelay;
						readLatency += senseAmp.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				} 
				if (!CalculateclkFreq) {
					
					// Anni update: hide readLatencyAccum by pipeline
					adder.CalculateLatency(1e20, dff.capTgDrain, 1);	// numRead = numReadOperationPerRow*numRow*activityRowRead
					dff.CalculateLatency(1e20, 1); // numRead = numReadOperationPerRow*numRow*activityRowRead
					if (numCellPerSynapse > 1) {
						shiftAddWeight.CalculateLatency(1);	// numCellPerSynapse-1
					}
					if (numReadPulse > 1) {
						shiftAddInput.CalculateLatency(1);					
					}					
					if (param->synchronous) {
						readLatencyADC = numReadOperationPerRow*numRow*activityRowRead;
						// Anni update: hide readLatencyAccum by pipeline
						// adder is pipelined with ADC
						readLatencyAccum += numReadOperationPerRow*(numRow*activityRowRead-1) * (ceil(adder.readLatency*clkFreq)-1);		
						// shiftAddWeight and shiftAddInput are pipelined with ADC+Adder (no mux, only once computation for whole array)
						readLatencyAccum += MAX(ceil((shiftAddWeight.adder.readLatency*(numCellPerSynapse-1)+shiftAddInput.adder.readLatency)*clkFreq)-(readLatencyADC+readLatencyAccum), 0);	
					} else {
						readLatencyADC = (precharger.readLatency + colDelay + senseAmp.readLatency) * numReadOperationPerRow*numRow*activityRowRead * (validated==true? param->beta : 1);		
						readLatencyOther = wlDecoder.readLatency * numRow*activityRowRead * (validated==true? param->beta : 1);
						// Anni update
						readLatencyAccum = MAX(adder.readLatency*(numReadOperationPerRow*numRow*activityRowRead-1) + shiftAddWeight.adder.readLatency*(numCellPerSynapse-1) + \
											shiftAddInput.adder.readLatency - readLatencyADC - readLatencyOther, 0);
					}
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}	
					// // Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
					// double resPull;
					// resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
					// tau = resPull * cell.capSRAMCell;
					// gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
					// beta = 1 / (resPull * gm);
					// sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numWriteOperationPerRow*numRow*activityRowWrite);
					// // writeLatency += horowitz(tau, beta, 1e20, NULL) * numWriteOperationPerRow * numRow * activityRowWrite;
					// // writeLatency += wlDecoder.writeLatency;
					// // writeLatency += precharger.writeLatency;
					// // writeLatency += sramWriteDriver.writeLatency;
			} else if (conventionalParallel) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {
					double bufferlatency = 0;

					// 1.4 update : buffer for the latency
					int iterbuffnum = param->buffernumber-1;
					if (param->buffernumber>0) {
						double buffnum = param->buffernumber; // # of buffers
						double sectionnum = param->numColSubArray/(param->buffernumber+1); // # of cells in each section
						double unitcap= capRow1/param->numColSubArray;
						double unitres= resRow/param->numColSubArray;
						param->unitcap = unitcap;
						param->unitres = unitres;
						param->drivecapin = drivecapin;

						double capload_repeater1 = capRow1/(buffnum+1)+ drivecapin;
						double capload_repeater2 = capRow1/(buffnum+1);
						
						wlSwitchMatrix.CalculateLatency(1e20, capRow1/(buffnum+1)+ drivecapin, unitres * sectionnum, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);						
						
						double t1 =  targetdriveres * (drivecapout) * 0.69 
									+ unitcap * sectionnum * (0.69*targetdriveres + 0.38* unitres * sectionnum  )
									+ (unitcap * sectionnum * 0.69 + 0.69 *  targetdriveres  )* drivecapin;										
						double t2 =  targetdriveres * (drivecapout) * 0.69 
									+ unitcap * sectionnum * (0.69*targetdriveres + 0.38* unitres * sectionnum  );										
						double t3 = (drivecapout + drivecapin) * targetdriveres * 0.69;
						
						while (iterbuffnum >= 0){
							if (iterbuffnum  == 0) {
								bufferlatency += t2 + t3;
							}
							else {
								bufferlatency += t1+ t3;
							}
							iterbuffnum  = iterbuffnum  -1;
						}
					} else {
						bufferlatency =0;
						wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					}
					precharger.CalculateLatency(1e20, capCol, 1, numWriteOperationPerRow*numRow*activityRowWrite);
					if (SARADC) {
						sarADC.CalculateLatency(1);
					} else {
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						multilevelSAEncoder.CalculateLatency(1e20, 1);
					}					
					if (numColMuxed > 1) {
						mux.CalculateLatency(0, 0, 1);						
						// 1.4 update: more arguments for muxdecoder.calculatelatency
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}					

					// Read

					// 1.4 update: SRAM column cap update (calibration with IMEC data)
					// 1.4 update: parallel mode discharge characteristics should be accounted
					// 1.4 update: SRAM row delay update

					// tau = (resCellAccess + resPullDown / m) * (capCellAccess + capCol) + resCol * capCol / 2;
					// m = enabled rows && weight 1

					// 1.4 update : new bitline model - needs check for Parallel mode
					double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
					double BLCap_perCell = capCol / numRow + capCellAccess; // Anni update
					double BLRes_perCell = resCol / numRow;
					double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;

					colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));  

					if (CalculateclkFreq) {
						// 1.4 update - updated
						readLatency += MAX(wlSwitchMatrix.readLatency + bufferlatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) );
						readLatency += precharger.readLatency;
						// readLatency += colDelay;	   
						readLatency += multilevelSenseAmp.readLatency;

						param->rowdelay = wlSwitchMatrix.readLatency + bufferlatency;
						param->muxdelay = mux.readLatency+muxDecoder.readLatency;
						param->ADClatency = multilevelSenseAmp.readLatency;

						readLatency += multilevelSAEncoder.readLatency;
						readLatency += sarADC.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {
					// Anni update: add partial sums
					if (numAdd > 1) {	
						adder.CalculateLatency(1e20, dff.capTgDrain, 1); // numRead = numColMuxed*(numAdd-1)
						dff.CalculateLatency(1e20, 1);	// numRead = numColMuxed*(numAdd-1)
					}
					if (numCellPerSynapse > 1) {
						shiftAddWeight.CalculateLatency(1); // numRead = (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse)
					}
					if (numReadPulse > 1) {
						shiftAddInput.CalculateLatency(1);	// numRead = ceil(numColMuxed/numCellPerSynapse)
					}					
					if (param->synchronous) {
						readLatencyADC = numColMuxed * numAdd;	// Anni update	
						// Anni update: hide readLatencyAccum by pipeline
						if (numAdd > 1) {	// adder is pipelined with ADC
							readLatencyAccum += numColMuxed*(numAdd-1) * (ceil(adder.readLatency*clkFreq)-1);	
						}	
						if (numCellPerSynapse > 1) {	// shiftAddWeight is pipelined with adder+ADC of one column
							readLatencyAccum += (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddWeight.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/numColMuxed, 0);	
						} 
						if (numReadPulse > 1) {	 // shiftAddInput is pipelined with adder+ADC+shiftaddweight of numCellPerSynapse columns
							readLatencyAccum += ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddInput.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/ceil(numColMuxed/numCellPerSynapse), 0);	
						} 
					} else {
						readLatencyADC = (precharger.readLatency + colDelay + multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency) * numColMuxed * (validated==true? param->beta : 1) * numAdd;
						readLatencyOther = MAX(wlSwitchMatrix.readLatency * numAdd, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) ) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed*(numAdd-1) + shiftAddWeight.adder.readLatency * (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) + \
											shiftAddInput.adder.readLatency * ceil(numColMuxed/numCellPerSynapse) - readLatencyADC - readLatencyOther, 0);
					}					
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}
			} else if (BNNsequentialMode || XNORsequentialMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {

					// 1.4 update: new arguments for row decoder
					wlDecoder.CalculateLatency(1e20, capRow1, NULL, resRow, numCol, 1, numRow*activityRowWrite);
					precharger.CalculateLatency(1e20, capCol,1, numWriteOperationPerRow*numRow*activityRowWrite);				
					senseAmp.CalculateLatency(1);				
					
					// Read

					// 1.4 update : new bitline model - needs check for BNN/XNOR mode
					double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
					double BLCap_perCell = capCol / numRow + capCellAccess; // Anni update
					double BLRes_perCell = resCol / numRow;
					double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;

					colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));  


					if (CalculateclkFreq) {
						readLatency += wlDecoder.readLatency;
						readLatency += precharger.readLatency;
						readLatency += colDelay;
						readLatency += senseAmp.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {
					// Anni update: hide readLatencyAccum by pipeline
					adder.CalculateLatency(1e20, dff.capTgDrain, 1);
					dff.CalculateLatency(1e20, 1);
					if (param->synchronous) {
						readLatencyADC = numReadOperationPerRow*numRow*activityRowRead;						
						// adder is pipelined with ADC
						readLatencyAccum = numReadOperationPerRow*(numRow*activityRowRead-1) * (ceil(adder.readLatency*clkFreq)-1);		
					} else {
						readLatencyADC = (precharger.readLatency + colDelay + senseAmp.readLatency) * numReadOperationPerRow*numRow*activityRowRead * (validated==true? param->beta : 1);						
						readLatencyOther = wlDecoder.readLatency * numRow*activityRowRead * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numReadOperationPerRow*(numRow*activityRowRead-1) - readLatencyADC - readLatencyOther, 0);
					}				
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}				
			} else if (BNNparallelMode || XNORparallelMode) {
				int numReadOperationPerRow = (int)ceil((double)numCol/numReadCellPerOperationNeuro);
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {
					wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					precharger.CalculateLatency(1e20, capCol, 1, numWriteOperationPerRow*numRow*activityRowWrite);					
					if (SARADC) {
						sarADC.CalculateLatency(1);
					} else {
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						multilevelSAEncoder.CalculateLatency(1e20, 1);
					}
					if (numColMuxed > 1) {
						mux.CalculateLatency(0, 0, 1);

						// 1.4 update: new arguments for muxdecoder
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}				
					
					// Read
					// 1.4 update : new bitline model - needs check for BNN/XNOR parallel mode
					double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
					double BLCap_perCell = capCol / numRow + capCellAccess; // Anni update
					double BLRes_perCell = resCol / numRow;
					double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;

					colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));
					
					if (CalculateclkFreq) {
						readLatency += MAX(wlSwitchMatrix.readLatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0));
						readLatency += precharger.readLatency;
						// readLatency += colDelay;
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;
						readLatency += sarADC.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {
					// Anni update: add partial sums; hide readLatencyAccum by pipeline
					if (numAdd > 1) {	
						adder.CalculateLatency(1e20, dff.capTgDrain, 1);
						dff.CalculateLatency(1e20, 1);
					}
					if (param->synchronous) {
						readLatencyADC = numColMuxed * numAdd;
						// adder is pipelined with ADC
						readLatencyAccum = numColMuxed * (numAdd-1) * (ceil(adder.readLatency*clkFreq)-1);
					} else {
						readLatencyADC = (precharger.readLatency + colDelay + multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency) * numColMuxed * (validated==true? param->beta : 1) * numAdd;
						readLatencyOther = MAX(wlSwitchMatrix.readLatency * numAdd, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0) ) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed * (numAdd-1) - readLatencyADC - readLatencyOther, 0);
					}
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}			
			}
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			if (conventionalSequential) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;

				// 1.4 update: needs check
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp);	// Just to generate colRamp
				colDelay = tau * 0.2;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {		
					
					// 1.4 update
					wlDecoder.CalculateLatency(1e20, capRow2, NULL, resRow, numCol, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					if (cell.accessType == CMOS_access) {
						wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
					} else {
						wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);										
					}					
					if (numColMuxed > 1) {
						mux.CalculateLatency(colRamp, 0, 1);
						// 1.4 update
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}
					if (SARADC) {
						sarADC.CalculateLatency(1);
					} else {
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						if (avgWeightBit > 1) {
							multilevelSAEncoder.CalculateLatency(1e20, 1);
						}
					}					
					if (CalculateclkFreq) {
						readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0));
						readLatency += colDelay;
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;
						readLatency += sarADC.readLatency;
						readLatency *= (validated==true? param->beta : 1);		// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {
					// Anni update
					adder.CalculateLatency(1e20, dff.capTgDrain, 1);	// numRead = numColMuxed*(numRow*activityRowRead-1)
					dff.CalculateLatency(1e20, 1);
					if (numCellPerSynapse > 1) {
						shiftAddWeight.CalculateLatency(1);	// numRead = (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse)
					}
					if (numReadPulse > 1) {
						shiftAddInput.CalculateLatency(1);	// numRead = ceil(numColMuxed/numCellPerSynapse)
					}
					if (param->synchronous) {
						readLatencyADC = numRow*activityRowRead*numColMuxed;
						// Anni update: hide readLatencyAccum by pipeline
						// adder is pipelined with ADC
						readLatencyAccum += numColMuxed*(numRow*activityRowRead-1) * (ceil(adder.readLatency*clkFreq)-1);
						if (numCellPerSynapse > 1) {	// shiftAddWeight is pipelined with adder+ADC of one column
							readLatencyAccum += (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddWeight.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/numColMuxed, 0);	
						} 
						if (numReadPulse > 1) {	 // shiftAddInput is pipelined with adder+ADC+shiftaddweight of numCellPerSynapse columns
							readLatencyAccum += ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddInput.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/ceil(numColMuxed/numCellPerSynapse), 0);	
						} 
					} else {
						readLatencyADC = (multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency + colDelay) * (numRow*activityRowRead*numColMuxed) * (validated==true? param->beta : 1);
						readLatencyOther = MAX((wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency)*numRow*activityRowRead, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0)) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed*(numRow*activityRowRead-1) + shiftAddWeight.adder.readLatency * (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) + \
											shiftAddInput.adder.readLatency * ceil(numColMuxed/numCellPerSynapse) - readLatencyADC - readLatencyOther, 0);

					}
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}
					// // Write
					// wllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					// bllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					// sllevelshifter.CalculateLatency(1e20, 2*slSwitchMatrix.capTgDrain, slSwitchMatrix.resTg, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					// slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numWriteOperationPerRow*numRow*activityRowWrite); 
					// writeLatencyArray = numWritePulse * param->writePulseWidthLTP + (-numErasePulse) * param->writePulseWidthLTD;
					// writeLatency += MAX(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency + wlDecoderDriver.writeLatency, sllevelshifter.writeLatency + slSwitchMatrix.writeLatency + bllevelshifter.writeLatency);
					// writeLatency += writeLatencyArray;
					
			} else if (conventionalParallel) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;

				// 1.4 update: needs check - (capCol)*(cell.resMemCellAvg/(numRow));? 	Anni update: numRow->numRowParallel
				double tau = (capCol)*(cell.resMemCellAvg/(numRowParallel/2));

				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2;  // assume the 15~20% voltage drop is enough for sensing
				if (CalculateclkFreq || !param->synchronous) {				
					double bufferlatency=0;			 
					if (cell.accessType == CMOS_access) {
						// 1.4 update : buffer latency
						int iterbuffnum = param->buffernumber-1;
						if (param->buffernumber>0) {
							double buffnum = param->buffernumber; // # of buffers
							double sectionnum = param->numColSubArray/(param->buffernumber+1); // # of cells in each section
							double unitcap= capRow2/param->numColSubArray;
							double unitres= resRow/param->numColSubArray;

							param->unitcap = unitcap;
							param->unitres = unitres;
							param->drivecapin = drivecapin;

							double capload_repeater1 = capRow2/(buffnum+1)+ drivecapin;
							double capload_repeater2 = capRow2/(buffnum+1);
							
							wlNewSwitchMatrix.CalculateLatency(1e20, capload_repeater1, unitres * sectionnum, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
							
							double t1 =  targetdriveres * (drivecapout) * 0.69 
										+ unitcap * sectionnum * (0.69*targetdriveres + 0.38* unitres * sectionnum)
										+ (unitcap * sectionnum * 0.69 + 0.69 * targetdriveres)* drivecapin;											
							double t2 =  targetdriveres * (drivecapout) * 0.69 
										+ unitcap * sectionnum * (0.69*targetdriveres + 0.38* unitres * sectionnum);									
							double t3 = (drivecapout + drivecapin) * targetdriveres * 0.69;				
							while (iterbuffnum >= 0){
								if (iterbuffnum  == 0) {
									bufferlatency += t2 + t3;
								} else {
									bufferlatency += t1 + t3;
								}
								iterbuffnum  = iterbuffnum-1;
							}
						} else {
							bufferlatency=0;
							wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
						}
					} else {
						wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					}
					if (numColMuxed>1) {
						mux.CalculateLatency(colRamp, 0, 1);
						// 1.4 update
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}
					if (SARADC) {
						sarADC.CalculateLatency(1);
					} else {
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						multilevelSAEncoder.CalculateLatency(1e20, 1);
					}				
					if (CalculateclkFreq) {
						// 1.4 update - updated
						readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency + bufferlatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0));
						// readLatency += colDelay;
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;
						readLatency += sarADC.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default									
						param->rowdelay = wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency + bufferlatency;
						param->muxdelay = mux.readLatency+muxDecoder.readLatency;
						param->ADClatency = multilevelSenseAmp.readLatency;					
					
					
					}
				}
				if (!CalculateclkFreq) {
					// Anni update: add partial sums
					if (numAdd > 1) {	
						adder.CalculateLatency(1e20, dff.capTgDrain, 1); // numRead = numColMuxed*(numAdd-1)
						dff.CalculateLatency(1e20, 1);	// numRead = numColMuxed*(numAdd-1)
					}
					if (numCellPerSynapse > 1) {
						shiftAddWeight.CalculateLatency(1);	// numRead = (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse)
					}
					if (numReadPulse > 1) {
						shiftAddInput.CalculateLatency(1);	// numRead = ceil(numColMuxed/numCellPerSynapse)
					}
					if (param->synchronous) {
						readLatencyADC = numColMuxed * numAdd;	// Anni update	
						// Anni update: hide readLatencyAccum by pipeline
						if (numAdd > 1) {	// adder is pipelined with ADC
							readLatencyAccum += numColMuxed*(numAdd-1) * (ceil(adder.readLatency*clkFreq)-1);	
						}	
						if (numCellPerSynapse > 1) {	// shiftAddWeight is pipelined with adder+ADC of one column
							readLatencyAccum += (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddWeight.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/numColMuxed, 0);	
						} 
						if (numReadPulse > 1) {	 // shiftAddInput is pipelined with adder+ADC+shiftaddweight of numCellPerSynapse columns
							readLatencyAccum += ceil(numColMuxed/numCellPerSynapse) * MAX(ceil(shiftAddInput.adder.readLatency*clkFreq) - (readLatencyADC+readLatencyAccum)/ceil(numColMuxed/numCellPerSynapse), 0);	
						} 
					} else {
						readLatencyADC = (multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency + colDelay) * numColMuxed * (validated==true? param->beta : 1) * numAdd;
						readLatencyOther = MAX((wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency) * numAdd, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0)) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed*(numAdd-1) + shiftAddWeight.adder.readLatency * (numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse) + \
											shiftAddInput.adder.readLatency * ceil(numColMuxed/numCellPerSynapse) - readLatencyADC - readLatencyOther, 0);
					}
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}
			} else if (BNNsequentialMode || XNORsequentialMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				double colRamp = 0;

				// 1.4 update: needs check
				double tau = (capCol)*(cell.resMemCellAvg);
				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2 ;  // assume the 15~20% voltage drop is enough for sensing
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				if (CalculateclkFreq || !param->synchronous) {

					// 1.4 update
					wlDecoder.CalculateLatency(1e20, capRow2, NULL, resRow, numCol, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					if (cell.accessType == CMOS_access) {
						wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);	
					} else {
						wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					}
					if (numColMuxed > 1) {
						mux.CalculateLatency(colRamp, 0, 1);
						// 1.4 update
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}
						// 1.4 update 230615
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						multilevelSAEncoder.CalculateLatency(1e20, 1);

					if (CalculateclkFreq) {
						readLatency += MAX(wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0));
						// readLatency += colDelay;
						// 1.4 update 230615
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {
					// Anni update: hide readLatencyAccum by pipeline
					adder.CalculateLatency(1e20, dff.capTgDrain, 1);
					dff.CalculateLatency(1e20, 1);
					if (param->synchronous) {
						readLatencyADC = numRow*activityRowRead*numColMuxed;
						// adder is pipelined with ADC
						readLatencyAccum = numColMuxed*(numRow*activityRowRead-1) * (ceil(adder.readLatency*clkFreq)-1);		
					} else { 
						readLatencyADC = (rowCurrentSenseAmp.readLatency + colDelay) * numRow*activityRowRead*numColMuxed * (validated==true? param->beta : 1);
						readLatencyOther = MAX((wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency)*numRow*activityRowRead, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0)) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed*(numRow*activityRowRead-1) - readLatencyADC - readLatencyOther, 0);
					}					
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}
			} else if (BNNparallelMode || XNORparallelMode) {
				double capBL = lengthCol * 0.2e-15/1e-6;
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double colRamp = 0;

				// 1.4 update: needs check; Anni update: numRow->numRowParallel
				double tau = (capCol)*(cell.resMemCellAvg/(numRowParallel/2));
				colDelay = horowitz(tau, 0, 1e20, &colRamp);
				colDelay = tau * 0.2;  // assume the 15~20% voltage drop is enough for sensing
				if (CalculateclkFreq || !param->synchronous) {
					if (cell.accessType == CMOS_access) {
						wlNewSwitchMatrix.CalculateLatency(1e20, capRow2, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					} else {
						wlSwitchMatrix.CalculateLatency(1e20, capRow1, resRow, 1, 2*numWriteOperationPerRow*numRow*activityRowWrite);
					}
					if (numColMuxed > 1) {
						mux.CalculateLatency(colRamp, 0, 1);
						// 1.4 update 
						muxDecoder.CalculateLatency(1e20, mux.capTgGateN*ceil(numCol/numColMuxed), mux.capTgGateP*ceil(numCol/numColMuxed), 0, 0, 1, 0);
					}
					if (SARADC) {
						sarADC.CalculateLatency(1);
					} else {
						multilevelSenseAmp.CalculateLatency(columnResistance, 1, 1);
						multilevelSAEncoder.CalculateLatency(1e20, 1);
					}
					if (CalculateclkFreq) {
						readLatency += MAX(wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0));
						// readLatency += colDelay;
						readLatency += multilevelSenseAmp.readLatency;
						readLatency += multilevelSAEncoder.readLatency;
						readLatency += sarADC.readLatency;
						readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default
					}
				}
				if (!CalculateclkFreq) {					
					// Anni update: add partial sums; hide readLatencyAccum by pipeline
					if (numAdd > 1) {	
						adder.CalculateLatency(1e20, dff.capTgDrain, 1);
						dff.CalculateLatency(1e20, 1);
					}
					if (param->synchronous) {
						readLatencyADC = numColMuxed * numAdd;
						// adder is pipelined with ADC
						readLatencyAccum = numColMuxed * (numAdd-1) * (ceil(adder.readLatency*clkFreq)-1);
					} else { 
						readLatencyADC = (multilevelSenseAmp.readLatency + multilevelSAEncoder.readLatency + sarADC.readLatency + colDelay) * numColMuxed * (validated==true? param->beta : 1) * numAdd;
						readLatencyOther = MAX((wlNewSwitchMatrix.readLatency + wlSwitchMatrix.readLatency) * numAdd, ((numColMuxed > 1)==true? (mux.readLatency+muxDecoder.readLatency):0)) * numColMuxed * (validated==true? param->beta : 1);
						// Anni update: hide readLatencyAccum by pipeline
						readLatencyAccum = MAX(adder.readLatency * numColMuxed * (numAdd-1) - readLatencyADC - readLatencyOther, 0);
					}					
					readLatency = readLatencyADC + readLatencyAccum + readLatencyOther;
				}
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
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize,
					cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;
			// Anni update
			leakageSRAMInUse = leakage;
			leakage *= numRow * numCol;

			if (conventionalSequential) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse);				
				// Anni update
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro/numCellPerSynapse*(ceil(log2(numRow))/2+1), param->validated);
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				// Anni update:
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower(numCellPerSynapse-1);	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(1);					
				}

				// Array
				// 1.4 update: read energy update
				readDynamicEnergyArray = capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowRead;  // Just BL discharging // -added, wordline charging
				
				// // 1.4 update: WL energy for write + modification (assuming toggling of SRAM bit at every write, each Q/Qbar consumes half CVdd^2)
				// writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// writeDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowWrite;
				
				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				
				readDynamicEnergyADC = precharger.readDynamicEnergy + readDynamicEnergyArray + senseAmp.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy;

				// Write
				// writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				// writeDynamicEnergy += precharger.writeDynamicEnergy;
				// writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += senseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += shiftAddWeight.leakage + shiftAddInput.leakage;
				// Anni update
				leakageSRAMInUse *= (numRow-1) * numCol;

			} else if (conventionalParallel) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				// Anni update
				precharger.CalculatePower(numColMuxed*numAdd, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				
				// 1.4 update: ADC update
				param->reference_energy_peri = capRow1/param->numColSubArray * tech.vdd * tech.vdd * (numRow);				
				
				if (numColMuxed > 1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				// Anni update: numAdd
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numAdd);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numAdd);
					multilevelSAEncoder.CalculatePower(numColMuxed * numAdd);
				}
				if (numAdd > 1) {	
					adder.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)); 
					dff.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)*(log2(levelOutput) + ceil(log2(numAdd))/2), param->validated);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower((numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse));	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Array

				// 1.4 update: read energy update; Anni update: * numColMuxed
				readDynamicEnergyArray = capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowRead; // added for WL/BL discharging
				// 1.4 update: buffer energy
				readDynamicEnergyArray += (drivecapin + drivecapout) * tech.vdd * tech.vdd * param->buffernumber * 2 * numRow * activityRowRead;
				// 1.4 update: iterate for numColMuxed - needs check if this is necessary
				readDynamicEnergyArray *= numColMuxed;
				
				// // 1.4 update: WL energy for write + modification (assuming toggling of SRAM bit at every write, each Q/Qbar consumes half CVdd^2)
				// writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// writeDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowWrite;

				
				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				// readDynamicEnergy += precharger.readDynamicEnergy; -> precharger is not needed for SRAM parallel mode
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				// Anni update
				readDynamicEnergy += ((numColMuxed > 1)==true? mux.readDynamicEnergy:0);
				readDynamicEnergy += ((numColMuxed > 1)==true? muxDecoder.readDynamicEnergy:0);
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// 1.4 update : precharger not needed 230619
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;				
				// Anni update
				readDynamicEnergyAccum = shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy + adder.readDynamicEnergy + dff.readDynamicEnergy;
				readDynamicEnergyOther = wlSwitchMatrix.readDynamicEnergy + ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				
				// Write
				// writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += precharger.writeDynamicEnergy;
				// writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddWeight.leakage + shiftAddInput.leakage;
				// Anni update
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				// Anni update
				leakageSRAMInUse *= (numRow * numCol - numRowParallel * ceil(numCol/numColMuxed));
			
			} else if (BNNsequentialMode || XNORsequentialMode) {
				wlDecoder.CalculatePower(numRow*activityRowRead, numRow*activityRowWrite);
				precharger.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				// Anni update: 
				adder.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro);				
				dff.CalculatePower(numReadOperationPerRow*numRow*activityRowRead, numReadCellPerOperationNeuro*(ceil(log2(numRow))/2+1), param->validated);
				senseAmp.CalculatePower(numReadOperationPerRow*numRow*activityRowRead);
				
				// 1.4 update: read energy update : needs check for BNN/XNOR mode
				readDynamicEnergyArray = capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowRead; // added for WL/BL discharging
				
				// // 1.4 update: WL energy for write + modification (assuming toggling of SRAM bit at every write, each Q/Qbar consumes half CVdd^2)
				// // needs check for BNN/XNOR mode
				// writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// writeDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowWrite;

				// Read
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += senseAmp.readDynamicEnergy;
				
				// Write				
				// writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				// writeDynamicEnergy += precharger.writeDynamicEnergy;
				// writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlDecoder.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += senseAmp.leakage;
				leakage += dff.leakage;
				leakage += adder.leakage;
				// Anni update
				leakageSRAMInUse *= (numRow-1) * numCol;
				
			} else if (BNNparallelMode || XNORparallelMode) {
				wlSwitchMatrix.CalculatePower(numColMuxed, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead, activityColWrite);
				// Anni update
				precharger.CalculatePower(numColMuxed * numAdd, numWriteOperationPerRow*numRow*activityRowWrite);
				sramWriteDriver.CalculatePower(numWriteOperationPerRow*numRow*activityRowWrite);
				// Anni update: add mux; numAdd
				if (numColMuxed>1) {
					mux.CalculatePower(numColMuxed);	// Mux still consumes energy during row-by-row read
					muxDecoder.CalculatePower(numColMuxed, 1);
				}
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numAdd);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numAdd);
					multilevelSAEncoder.CalculatePower(numColMuxed*numAdd);
				}
				if (numAdd > 1) {	
					adder.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)); 
					dff.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)*(log2(levelOutput) + ceil(log2(numAdd))/2), param->validated);
				}
				// Array
				// 1.4 update: read energy update : needs check for BNN/XNOR mode; Anni update
				readDynamicEnergyArray = capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowRead * numColMuxed; // added for WL/BL discharging
				// 1.4 update: ADC update
				param->reference_energy_peri = capRow1/param->numColSubArray * tech.vdd * tech.vdd * numRow;
				
				// // 1.4 update: WL energy for write + modification (assuming toggling of SRAM bit at every write, each Q/Qbar consumes half CVdd^2)
				// // needs check for BNN/XNOR mode
				// writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * numCol * activityColWrite * numRow * activityRowWrite;    // flip Q and Q_bar
				// writeDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd * (numRow) * activityRowWrite;

				// Read
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += precharger.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				// Anni update:
				readDynamicEnergy += ((numColMuxed > 1)==true? mux.readDynamicEnergy:0);
				readDynamicEnergy += ((numColMuxed > 1)==true? muxDecoder.readDynamicEnergy:0);
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				
				// Write				
				// writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += precharger.writeDynamicEnergy;
				// writeDynamicEnergy += sramWriteDriver.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage += wlSwitchMatrix.leakage;
				leakage += precharger.leakage;
				leakage += sramWriteDriver.leakage;
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				// Anni update
				leakage += dff.leakage;
				leakage += adder.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				// Anni update
				leakageSRAMInUse *= (numRow * numCol - numRowParallel * ceil(numCol/numColMuxed));
				
			}		
	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET) {
			// Anni update
			leakageSRAMInUse = 0;
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
				// Anni update
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(ceil(log2(numRow))/2 + avgWeightBit), param->validated); 
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower((numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse));	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
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
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += sarADC.readDynamicEnergy;
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				readDynamicEnergyAccum = adder.readDynamicEnergy + dff.readDynamicEnergy + shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergyOther = wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy + wlDecoderDriver.readDynamicEnergy + ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);

				// Write					
				// writeDynamicEnergyArray = writeDynamicEnergyArray;
				// writeDynamicEnergy = 0;
				// if (cell.writeVoltage > 1.5) {
					// wllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// bllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// sllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);	
					// writeDynamicEnergy += wllevelshifter.writeDynamicEnergy + bllevelshifter.writeDynamicEnergy + sllevelshifter.writeDynamicEnergy; 
				// }
				// writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				// writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				// writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				// writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;
				
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
				leakage += shiftAddWeight.leakage + shiftAddInput.leakage;
					
			} else if (conventionalParallel) {
				double numReadCells = (int)ceil((double)numCol/numColMuxed);    // similar parameter as numReadCellPerOperationNeuro, which is for SRAM
				int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);
				double capBL = lengthCol * 0.2e-15/1e-6;

				// 1.4 update: ADC update
				// param->reference_energy_peri = capRow2/param->numColSubArray * tech.vdd * tech.vdd * numRow;
				// for RRAM, the reference columns can always be turned on

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
				// Anni update: numAdd
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numAdd);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numAdd);
					multilevelSAEncoder.CalculatePower(numColMuxed * numAdd);
				}
				if (numAdd > 1) {	
					adder.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)); 
					dff.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)*(log2(levelOutput) + ceil(log2(numAdd))/2), param->validated);
				}
				if (numCellPerSynapse > 1) {
					shiftAddWeight.CalculatePower((numCellPerSynapse-1)*ceil(numColMuxed/numCellPerSynapse));	
				}
				if (numReadPulse > 1) {
					shiftAddInput.CalculatePower(ceil(numColMuxed/numCellPerSynapse));		
				}
				// Read
				readDynamicEnergyArray = 0;
				// Anni update:  * numAdd
				// readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells * numAdd; // Selected BLs activityColWrite -> no need already considered in multilevelsenseamp
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				// 1.4 update: buffer insertion
				readDynamicEnergyArray += (drivecapin + drivecapout) * tech.vdd * tech.vdd * param->buffernumber * 2 * numRow * activityRowRead;
				readDynamicEnergyArray *= numColMuxed;
				
				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				// Anni update: adder, dff, mux
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;	
				
				readDynamicEnergyADC = readDynamicEnergyArray + multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy + sarADC.readDynamicEnergy;
				// Anni update: accum, other
				readDynamicEnergyAccum = shiftAddWeight.readDynamicEnergy + shiftAddInput.readDynamicEnergy + adder.readDynamicEnergy + dff.readDynamicEnergy;				
				readDynamicEnergyOther = wlNewSwitchMatrix.readDynamicEnergy + wlSwitchMatrix.readDynamicEnergy + ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				
				// Write				
				// writeDynamicEnergyArray = writeDynamicEnergyArray;
				// writeDynamicEnergy = 0;
				// if (cell.writeVoltage > 1.5) {
					// wllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// bllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// sllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// writeDynamicEnergy += wllevelshifter.writeDynamicEnergy + bllevelshifter.writeDynamicEnergy + sllevelshifter.writeDynamicEnergy;
				// }
				// writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				// 1.4 update: repeater leakage
				double repeater_leakage = CalculateGateLeakage(INV, 1, widthInvN, widthInvP, inputParameter.temperature, tech) * param->buffernumber *2 * param->numRowSubArray;
				leakage += repeater_leakage ;											   
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				leakage += shiftAddWeight.leakage + shiftAddInput.leakage;
				// Anni update
				leakage += dff.leakage;
				leakage += adder.leakage;
				
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
				
				// 1.4 update 230615
				// rowCurrentSenseAmp.CalculatePower(columnResistance, numRow*activityRowRead);
		
				multilevelSenseAmp.CalculatePower(columnResistance, numAdd);
				multilevelSAEncoder.CalculatePower(numColMuxed * numAdd);

				adder.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells);
				// Anni update
				dff.CalculatePower(numColMuxed*numRow*activityRowRead, numReadCells*(ceil(log2(numRow))/2+1), param->validated); 
				
				// Read
				readDynamicEnergyArray = 0;
				readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells; // Selected BLs activityColWrite
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
				readDynamicEnergyArray *= numRow * activityRowRead * numColMuxed;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlDecoder.readDynamicEnergy;
				readDynamicEnergy += wlNewDecoderDriver.readDynamicEnergy;
				readDynamicEnergy += wlDecoderDriver.readDynamicEnergy;
				// Anni update
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy + multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;

				// Write				
				// writeDynamicEnergyArray = writeDynamicEnergyArray;
				// writeDynamicEnergy = 0;
				// if (cell.writeVoltage > 1.5) {
					// wllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// bllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// sllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// writeDynamicEnergy += wllevelshifter.writeDynamicEnergy + bllevelshifter.writeDynamicEnergy + sllevelshifter.writeDynamicEnergy;
				// }
				// writeDynamicEnergy += wlDecoder.writeDynamicEnergy;
				// writeDynamicEnergy += wlNewDecoderDriver.writeDynamicEnergy;
				// writeDynamicEnergy += wlDecoderDriver.writeDynamicEnergy;
				// writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
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
				// Anni update: numAdd
				if (SARADC) {
					sarADC.CalculatePower(columnResistance, numAdd);
				} else {
					multilevelSenseAmp.CalculatePower(columnResistance, numAdd);
					multilevelSAEncoder.CalculatePower(numColMuxed * numAdd);
				}
				if (numAdd > 1) {	
					adder.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)); 
					dff.CalculatePower(numColMuxed*(numAdd-1), ceil(numCol/numColMuxed)*(log2(levelOutput) + ceil(log2(numAdd))/2), param->validated);
				}
				// Read
				readDynamicEnergyArray = 0;
				// Anni update

				
				// readDynamicEnergyArray += capBL * cell.readVoltage * cell.readVoltage * numReadCells * numAdd; // Selected BLs activityColWrite -> Already considered in multilevelsenseamp.cpp
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd * numRow * activityRowRead; // Selected WL
				readDynamicEnergyArray *= numColMuxed;
				// 1.4 update: ADC update
				// param->reference_energy_peri = capRow2/param->numColSubArray * tech.vdd * tech.vdd * numRow;

				readDynamicEnergy = 0;
				readDynamicEnergy += wlNewSwitchMatrix.readDynamicEnergy;
				readDynamicEnergy += wlSwitchMatrix.readDynamicEnergy;
				// Anni update
				readDynamicEnergy += ((numColMuxed > 1)==true? (mux.readDynamicEnergy + muxDecoder.readDynamicEnergy):0);
				readDynamicEnergy += adder.readDynamicEnergy;
				readDynamicEnergy += dff.readDynamicEnergy;
				readDynamicEnergy += multilevelSenseAmp.readDynamicEnergy;
				readDynamicEnergy += multilevelSAEncoder.readDynamicEnergy;
				readDynamicEnergy += readDynamicEnergyArray;
				readDynamicEnergy += sarADC.readDynamicEnergy;

				// Write				
				// writeDynamicEnergyArray = writeDynamicEnergyArray;
				// writeDynamicEnergy = 0;
				// if (cell.writeVoltage > 1.5) {
					// wllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// bllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// sllevelshifter.CalculatePower(0, 2*numWriteOperationPerRow*numRow*activityRowWrite, activityRowRead);
					// writeDynamicEnergy += wllevelshifter.writeDynamicEnergy + bllevelshifter.writeDynamicEnergy + sllevelshifter.writeDynamicEnergy;
				// }
				// writeDynamicEnergy += wlNewSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += wlSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += slSwitchMatrix.writeDynamicEnergy;
				// writeDynamicEnergy += writeDynamicEnergyArray;				
				
				// Leakage
				leakage = 0;
				leakage += wlSwitchMatrix.leakage;
				leakage += wlNewSwitchMatrix.leakage;
				leakage += slSwitchMatrix.leakage;
				leakage += ((numColMuxed > 1)==true? (mux.leakage):0);
				leakage += ((numColMuxed > 1)==true? (muxDecoder.leakage):0);
				leakage += multilevelSenseAmp.leakage;
				leakage += multilevelSAEncoder.leakage;
				// Anni update
				leakage += dff.leakage;
				leakage += adder.leakage;

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
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
			}
		} else if (conventionalParallel) {
			wlSwitchMatrix.PrintProperty("wlSwitchMatrix");
			multilevelSenseAmp.PrintProperty("multilevelSenseAmp");
			multilevelSAEncoder.PrintProperty("multilevelSAEncoder");
			if (numReadPulse > 1) {
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
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
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
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
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
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
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
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
				shiftAddWeight.PrintProperty("shiftAddWeight");
				shiftAddInput.PrintProperty("shiftAddInput");
			}
		}
	} 
	FunctionUnit::PrintProperty("SubArray");
	cout << "Used Area = " << usedArea*1e12 << "um^2" << endl;
	cout << "Empty Area = " << emptyArea*1e12 << "um^2" << endl;
}

