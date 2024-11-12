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
						slSwitchMatrix(_inputParameter, _tech, _cell),
						blSwitchMatrix(_inputParameter, _tech, _cell),
						wlSwitchMatrix(_inputParameter, _tech, _cell),
						deMux(_inputParameter, _tech, _cell),
						precharger(_inputParameter, _tech, _cell),
						senseAmp(_inputParameter, _tech, _cell),
						wlDecoderDriver(_inputParameter, _tech, _cell),
						sramWriteDriver(_inputParameter, _tech, _cell){
	initialized = false;
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


	lengthRow = (double)numCol * cell.widthInFeatureSize * tech.featureSize;
	lengthCol = (double)numRow * cell.heightInFeatureSize * tech.featureSize;
	
	capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
	capRow2 = lengthRow * 0.2e-15/1e-6;	// WL for 1T1R
	capCol = lengthCol * 0.2e-15/1e-6;
	capCol1 = lengthCol * 0.2e-15/1e-6;
	
	resRow = lengthRow * unitWireRes; 
	resCol = lengthCol * unitWireRes;
	
	if ( cell.memCellType == Type::SRAM ) {
		resRow = lengthRow * param->Metal1_unitwireresis; 
		resCol = lengthCol * param->Metal0_unitwireresis;
	}

	//start to initializing the subarray modules
	if (cell.memCellType == Type::SRAM) {  //if array is SRAM
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
		
		// 1.4 update: consider SRAM parasitic cap
		capRow1 += 2*CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
		
		if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;
		else capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;			

		wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
		senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numCol);
		precharger.Initialize(numCol, resCol, 1, numCol, numCol);
		sramWriteDriver.Initialize(numCol, 1, numCol);
	
	} else if (cell.memCellType == Type::eDRAM) {	
		cell.resCellAccessWrite = CalculateOnResistance(cell.widthAccessCMOSWrite * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);	
		cell.capCellAccessWrite = CalculateDrainCap(cell.widthAccessCMOSWrite * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech);
	
		capRow1 += CalculateGateCap(cell.widthAccessCMOSWrite * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;	//capWordlineWrite
		capCol1 += cell.capCellAccessWrite * numRow / 2;	/* Due to shared contact */		// capBitlineWrite

		cell.resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
		cell.capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech);

		capRow2 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;
		capCol  += cell.capCellAccess * numRow / 2;	/* Due to shared contact */

		wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);
		senseAmp.Initialize(numCol, false, cell.minSenseVoltage, 2*lengthRow/numCol, clkFreq, numCol);
		precharger.Initialize(numCol, resCol, 1, numCol, numCol);

		if (cell.writeVoltage > 1.5) {
			wllevelshifter.Initialize(numRow, 1, clkFreq);	
			bllevelshifter.Initialize(numRow, 1, clkFreq);
			sllevelshifter.Initialize(numCol, 1, clkFreq);
		}

    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET || cell.memCellType == Type::STTMRAM || cell.memCellType == Type::SOTMRAM) {

		double unitcap= capRow2/param->numColSubArray;
		double unitres= resRow/param->numColSubArray;
		param->unitcap = unitcap;
		param->unitres = unitres;
		cell.resCellAccess = cell.resistanceOn * IR_DROP_TOLERANCE;    //calculate access CMOS resistance
		cell.widthAccessCMOS = CalculateOnResistance(((tech.featureSize <= 14*1e-9)? 2:1)*tech.featureSize, NMOS, inputParameter.temperature, tech) * LINEAR_REGION_RATIO / cell.resCellAccess;   //get access CMOS width
		double widthAccessInFeatureSize = cell.widthAccessCMOS;
		if (tech.featureSize <= 14 * 1e-9){			
			widthAccessInFeatureSize = ((cell.widthAccessCMOS-1) * tech.PitchFin + tech.widthFin) / tech.featureSize;  //convert #fin to F
		}
		if (widthAccessInFeatureSize > cell.widthInFeatureSize) {	// Place transistor vertically
			// printf("Transistor width of 1T1R=%.2fF is larger than the assigned cell width=%.2fF in layout\n", cell.widthAccessCMOS, cell.widthInFeatureSize);
			// exit(-1);
			cell.widthAccessCMOS = cell.widthInFeatureSize;
		}
		cell.resMemCellOn = cell.resCellAccess + cell.resistanceOn;        //calculate single memory cell resistance_ON
		cell.resMemCellOff = cell.resCellAccess + cell.resistanceOff;      //calculate single memory cell resistance_OFF

		// 1.4 update; Anni update: numRow->numRowParallel
		cell.resMemCellAvg = 1/(1/(cell.resistanceOn + cell.resCellAccess ) * numRowParallel/2.0 + 1/(cell.resistanceOff + cell.resCellAccess )* numRowParallel/2.0) * numRowParallel;      //calculate single memory cell resistance_AVG

		// 1.4 update: needs check  - capCol : cell.widthInFeatureSize / 
		capRow2 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
		capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;	// If capCol is found to be too large, increase cell.widthInFeatureSize to relax the limit

		if (cell.memCellType == Type::SOTMRAM) {
		cell.resCellAccessWrite = CalculateOnResistance(cell.widthAccessCMOSWrite * ((tech.featureSize <= 14*1e-9)? 2:1)* tech.featureSize, NMOS, inputParameter.temperature, tech);
		capRow1 += CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;
		// for SOT-MRAM, BL connected to the drain of both read and write access transistor 
		capCol += CalculateDrainCap(cell.widthAccessCMOSWrite * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;
		}

		if (cell.writeVoltage > 1.5) {
			wllevelshifter.Initialize(numRow, 1, clkFreq);
			bllevelshifter.Initialize(numRow, 1, clkFreq);
			sllevelshifter.Initialize(numCol, 1, clkFreq);
		}
		  
		double capBL = lengthCol * 0.2e-15/1e-6;
		double resTg = cell.resMemCellOn;     //transmission gate resistance
					
		wlDecoder.Initialize(REGULAR_ROW, (int)ceil(log2(numRow)), false, false);          
		if (cell.accessType == CMOS_access) {
			wlNewDecoderDriver.Initialize(numRow);          
		} else {
			wlDecoderDriver.Initialize(ROW_MODE, numRow, numCol);
		}						
		slSwitchMatrix.Initialize(COL_MODE, numCol, resTg, true, false, 1, 1, numCol, numCol, 1, clkFreq);     //SL use switch matrix

		senseAmp.Initialize(numCol, false, cell.minSenseVoltage, lengthRow/numCol, clkFreq, numCol);
	} 
	initialized = true;  //finish initialization
}



void SubArray::CalculateArea() {  //calculate layout area for total design
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;  //ensure initialization first
	} else {  //if initialized, start to do calculation
		area = 0;
		usedArea = 0;
		// Array only
		heightArray = lengthCol;
		widthArray = lengthRow;
		areaArray = heightArray * widthArray;
		if (cell.memCellType == Type::SRAM) {       			
			precharger.CalculateArea(NULL, widthArray, NONE);
			sramWriteDriver.CalculateArea(NULL, widthArray, NONE);		

			wlDecoder.CalculateArea(heightArray, NULL, NONE);  
			senseAmp.CalculateArea(NULL, widthArray, MAGIC);

			height = precharger.height + sramWriteDriver.height + heightArray + senseAmp.height;
			width = wlDecoder.width + widthArray;
			area = height * width;
			usedArea = areaArray + wlDecoder.area + precharger.area + sramWriteDriver.area + senseAmp.area;
		
	    } else if (cell.memCellType == Type::eDRAM) {
	
			precharger.CalculateArea(NULL, widthArray, NONE);
			wlDecoder.CalculateArea(heightArray, NULL, NONE);  
			senseAmp.CalculateArea(NULL, widthArray, MAGIC);
			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculateArea(heightArray, NULL, NONE);	
			}
	
			height = precharger.height + heightArray + senseAmp.height;
			width = ((cell.writeVoltage > 1.5)==true? wllevelshifter.width:0) + wlDecoder.width + widthArray;
			area = height * width;
			usedArea = ((cell.writeVoltage > 1.5)==true? wllevelshifter.area:0) + wlDecoder.area + precharger.area + senseAmp.area;
			if (param->memCell3D == false) {	// 3D projection or IWO transistor, only CMOS at bottm
				usedArea += areaArray;			
			} else {			
				height = sqrt(area);
				width = area / height;
			}

		} else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET || cell.memCellType == Type::STTMRAM || cell.memCellType == Type::SOTMRAM) {			
			// Level shifter for write
			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculateArea(heightArray, NULL, NONE);
				bllevelshifter.CalculateArea(heightArray, NULL, NONE);
				sllevelshifter.CalculateArea(NULL, widthArray, NONE);				
			}
				
			wlDecoder.CalculateArea(heightArray, NULL, NONE);
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.CalculateArea(heightArray, NULL, NONE);
			} else {
				wlDecoderDriver.CalculateArea(heightArray, NULL, NONE);
			}				
			slSwitchMatrix.CalculateArea(NULL, widthArray, NONE);
			senseAmp.CalculateArea(NULL, widthArray, MAGIC);			

			height = slSwitchMatrix.height + heightArray + senseAmp.height;// + ((cell.writeVoltage > 1.5)==true? (sllevelshifter.height):0)
			width = ((cell.writeVoltage > 1.5)==true? (wllevelshifter.width):0) + wlDecoder.width + wlNewDecoderDriver.width + wlDecoderDriver.width + widthArray; // + bllevelshifter.width
			area = height * width;
			usedArea = ((cell.writeVoltage > 1.5)==true? (wllevelshifter.area):0) + wlDecoder.area + wlDecoderDriver.area + wlNewDecoderDriver.area + slSwitchMatrix.area; // + bllevelshifter.area + sllevelshifter.area

			if (param->memCell3D == false) {	// 3D projection, only CMOS at bottm
				usedArea += areaArray;
			} else {
				height = sqrt(area);
				width = area / height;
			}

		} 
	}
}

void SubArray::CalculateLatency(double columnRes) {   //calculate latency for different mode 
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		
		readLatency = 0;
		readLatencyADC = 0;
		readLatencyAccum = 0;
		readLatencyOther = 0;
		writeLatency = 0;

		if (cell.memCellType == Type::SRAM) {

			// 1.4 update: new arguments for rowdecoder
			wlDecoder.CalculateLatency(1e20, capRow1, NULL, resRow, numCol, 1, numRow);				
			precharger.CalculateLatency(1e20, capCol, 1, numRow);					
			senseAmp.CalculateLatency(1);

			// Read
			double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
			double BLCap_perCell = capCol / numRow + capCellAccess; // Anni update
			double BLRes_perCell = resCol / numRow;
			double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;

			colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2));  

			readLatency += wlDecoder.readLatency;
			readLatency += precharger.readLatency;
			readLatency += colDelay;
			readLatency += senseAmp.readLatency;
			readLatency *= (validated==true? param->beta : 1);	// latency factor of sensing cycle, beta = 1.4 by default

			// Write (assume the average delay of pullup and pulldown inverter in SRAM cell)
			double resPull;
			resPull = (CalculateOnResistance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, inputParameter.temperature, tech) + CalculateOnResistance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, inputParameter.temperature, tech)) / 2;    // take average
			double tau = resPull * cell.capSRAMCell;
			double gm = (CalculateTransconductance(cell.widthSRAMCellNMOS * tech.featureSize, NMOS, tech) + CalculateTransconductance(cell.widthSRAMCellPMOS * tech.featureSize, PMOS, tech)) / 2;   // take average
			double beta = 1 / (resPull * gm);
			sramWriteDriver.CalculateLatency(1e20, capCol, resCol, numRow);
			writeLatency += horowitz(tau, beta, 1e20, NULL) * numRow;
			writeLatency += wlDecoder.writeLatency;
			writeLatency += precharger.writeLatency;
			writeLatency += sramWriteDriver.writeLatency;
		
		} else if (cell.memCellType == Type::eDRAM) {
			double cap = (cell.capCellAccessWrite + cell.capDRAMCell) * capCol1 / (cell.capCellAccessWrite + cell.capDRAMCell + capCol1);
			double res = resCol + cell.resCellAccessWrite;
			double tau = 2.3 * res * cap;
			double colRamp = 0;
			wlDecoder.CalculateLatency(1e20, capRow1, NULL, resRow, numCol, 1, numRow);			
			colDelay = horowitz(tau, 0, wlDecoder.rampOutput, &colRamp);			
			senseAmp.CalculateLatency(1);
			precharger.CalculateLatency(1e20, lengthCol * 0.2e-15/1e-6, 1, numRow);	
			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numRow);
				bllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numRow);
				sllevelshifter.CalculateLatency(1e20, 2*slSwitchMatrix.capTgDrain, slSwitchMatrix.resTg, 0, 2*numRow);
			}

            /* Refresh operation does not pass sense amplifier. */
            refreshLatency = wlDecoder.readLatency + colDelay + senseAmp.readLatency + precharger.readLatency;
            refreshLatency *= numRow; // TOTAL refresh latency for subarray
			readLatency = wlDecoder.readLatency + colDelay + senseAmp.readLatency + precharger.readLatency;
			/* assume symmetric read/write for DRAM/eDRAM bitline delay */
			writeLatency = readLatency;

	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET || cell.memCellType == Type::STTMRAM || cell.memCellType == Type::SOTMRAM) {
			double capBL = lengthCol * 0.2e-15/1e-6;
			double colRamp = 0;

			// 1.4 update: needs check
			double tau = (capCol)*(cell.resMemCellAvg);
			colDelay = horowitz(tau, 0, 1e20, &colRamp);	// Just to generate colRamp
			colDelay = tau * 0.2;  // assume the 15~20% voltage drop is enough for sensing
			int numWriteOperationPerRow = (int)ceil((double)numCol*activityColWrite/numWriteCellPerOperationNeuro);

			// 1.4 update
			wlDecoder.CalculateLatency(1e20, capRow2, NULL, resRow, numCol, 1, 2*numRow);
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow2, resRow, 1, 2*numRow);	
			} else {
				wlDecoderDriver.CalculateLatency(wlDecoder.rampOutput, capRow1, capRow1, resRow, 1, 2*numRow);										
			}					
			senseAmp.CalculateLatency(1);			

			readLatency += wlDecoder.readLatency + wlNewDecoderDriver.readLatency + wlDecoderDriver.readLatency;
			readLatency += colDelay;
			readLatency += senseAmp.readLatency;
			readLatency *= (validated==true? param->beta : 1);		// latency factor of sensing cycle, beta = 1.4 by default

			// Write
			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numRow);
				bllevelshifter.CalculateLatency(1e20, 2*wlNewDecoderDriver.capTgDrain, wlNewDecoderDriver.resTg, 0, 2*numRow);
				sllevelshifter.CalculateLatency(1e20, 2*slSwitchMatrix.capTgDrain, slSwitchMatrix.resTg, 0, 2*numRow);
			}
			slSwitchMatrix.CalculateLatency(1e20, capCol, resCol, 0, 2*numRow); 
			// writeLatencyArray = numWritePulse * param->writePulseWidthLTP + (-numErasePulse) * param->writePulseWidthLTD;
			writeLatency += MAX(wlDecoder.writeLatency + wlNewDecoderDriver.writeLatency + wlDecoderDriver.writeLatency, ((cell.writeVoltage > 1.5)==true? (sllevelshifter.writeLatency + bllevelshifter.writeLatency):0) + slSwitchMatrix.writeLatency);
			writeLatency += writeLatencyArray;
											
		}				
	}
}

void SubArray::CalculatePower() {
	if (!initialized) {
		cout << "[Subarray] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		readDynamicEnergyArray = 0;
		writeDynamicEnergyArray = 0;
		refreshDynamicEnergy = 0;

		if (cell.memCellType == Type::SRAM) {
			
			// Array leakage (assume 2 INV)
			leakage = 0;
			leakage += CalculateGateLeakage(INV, 1, cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize,
					cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, inputParameter.temperature, tech) * tech.vdd * 2;

			leakage += CalculateGateLeakage(INV, 1, cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, 0,
					inputParameter.temperature, tech) * tech.vdd;	/* two accesses NMOS, but combined as one with vdd crossed */

			leakageSRAMInUse = leakage;
			leakage *= (numRow * numCol);

			wlDecoder.CalculatePower(1, 1);
			precharger.CalculatePower(1, 1);
			sramWriteDriver.CalculatePower(1);
			senseAmp.CalculatePower(1);
			// Array
			// 1.4 update: read energy update
			readDynamicEnergyArray = capRow1 * tech.vdd * tech.vdd; // * (numRow);  // Just BL discharging // -added, wordline charging
			
			// 1.4 update: WL energy for write + modification (assuming toggling of SRAM bit at every write, each Q/Qbar consumes half CVdd^2)
			writeDynamicEnergyArray = cell.capSRAMCell * tech.vdd * tech.vdd * numCol; // * numRow;    // flip Q and Q_bar
			writeDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd; // * (numRow);
			
			// Read
			readDynamicEnergy = wlDecoder.readDynamicEnergy + readDynamicEnergyArray + senseAmp.readDynamicEnergy + precharger.readDynamicEnergy;			

			// Write
			writeDynamicEnergy = wlDecoder.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy + writeDynamicEnergyArray + precharger.writeDynamicEnergy;
			
			// Leakage
			leakage += wlDecoder.leakage + sramWriteDriver.leakage + senseAmp.leakage + precharger.leakage;
			leakageSRAMInUse *= (numRow-1) * numCol;
		} else if (cell.memCellType == Type::eDRAM) {
			double capBL = lengthCol * 0.2e-15/1e-6;
			double capRead = (cell.capCellAccess + cell.capDRAMCell) * capCol / (cell.capCellAccess + cell.capDRAMCell + capCol);
			double senseVoltage = tech.vdd * cell.capDRAMCell / (cell.capDRAMCell + capBL);//capCol1
			readDynamicEnergy = capRead * senseVoltage * tech.vdd * numCol;
			leakage = readDynamicEnergy / cell.retentionTime * numRow;
			double cap = (cell.capCellAccessWrite + cell.capDRAMCell) * capCol1 / (cell.capCellAccessWrite + cell.capDRAMCell + capCol1);
			writeDynamicEnergy = cap * cell.writeVoltage * cell.writeVoltage * numCol;			
			// * for fresh, still need to use the periphery circuits), the above equation is not correct
			if (cell.writeVoltage > 1.5){
				wllevelshifter.CalculatePower(1, 2, 1);		
				writeDynamicEnergy += wllevelshifter.readDynamicEnergy;
			}			
			wlDecoder.CalculatePower(1, 1);
			senseAmp.CalculatePower(1);
			precharger.CalculatePower(1, 1);
			writeDynamicEnergy += wlDecoder.writeDynamicEnergy + senseAmp.writeDynamicEnergy + precharger.writeDynamicEnergy;
			readDynamicEnergy += wlDecoder.readDynamicEnergy + senseAmp.readDynamicEnergy + precharger.readDynamicEnergy;
			readDynamicEnergy += wllevelshifter.readDynamicEnergy; 

			leakage += wlDecoder.leakage + senseAmp.leakage + precharger.leakage;
			refreshDynamicEnergy = readDynamicEnergy * numRow; // Energy for this entire subarray 

	    } else if (cell.memCellType == Type::RRAM || cell.memCellType == Type::FeFET || cell.memCellType == Type::STTMRAM || cell.memCellType == Type::SOTMRAM) {

			leakageSRAMInUse = 0;

			double capBL = lengthCol * 0.2e-15/1e-6;
			wlDecoder.CalculatePower(1, 2);
			if (cell.accessType == CMOS_access) {
				wlNewDecoderDriver.CalculatePower(1, 2);
			} else {
				wlDecoderDriver.CalculatePower(numCol, numCol, 1, 2);
			}
			slSwitchMatrix.CalculatePower(0, 2, 1, 1);

			if (cell.writeVoltage > 1.5) {
				wllevelshifter.CalculatePower(1, 2, 1);
				bllevelshifter.CalculatePower(1, 2, 1);
				sllevelshifter.CalculatePower(1, 2, 1);	
			}

			senseAmp.CalculatePower(1);

			// Read
			readDynamicEnergyArray = capBL * cell.readVoltage * cell.readVoltage * numCol; // Selected BLs activityColWrite			
			if (cell.memCellType == Type::SOTMRAM) {
				readDynamicEnergyArray += capRow1 * tech.vdd * tech.vdd; // Selected WL read;
			} else {
				readDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd; // Selected WL
			}

			readDynamicEnergy = wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy + wlDecoderDriver.readDynamicEnergy;
			readDynamicEnergy += readDynamicEnergyArray + senseAmp.readDynamicEnergy;
			readDynamicEnergy += wllevelshifter.readDynamicEnergy + bllevelshifter.readDynamicEnergy + sllevelshifter.readDynamicEnergy; 
			
			readDynamicEnergyADC = readDynamicEnergyArray + senseAmp.readDynamicEnergy;
			readDynamicEnergyOther = wlDecoder.readDynamicEnergy + wlNewDecoderDriver.readDynamicEnergy + wlDecoderDriver.readDynamicEnergy;

			// Write					
			writeDynamicEnergyArray = 0;
			if (cell.memCellType == Type::FeFET) {
				writeDynamicEnergyArray += cell.writeVoltage * cell.setCurrent * cell.writePulseWidth * numCol;
				writeDynamicEnergyArray += param->polarization*2*0.01*cell.writeVoltage*(2*tech.featureSize*tech.featureSize) * numCol;
				// SET
				writeDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd;	
				writeDynamicEnergyArray += (capCol + param->gateCapFeFET) * cell.writeVoltage * cell.writeVoltage * numCol;
			} else {
				writeDynamicEnergyArray += pow(cell.setCurrent, 2) * resCol * cell.writePulseWidth * numCol; 
				// energy at the access transistor
				writeDynamicEnergyArray += pow(cell.setCurrent, 2) * cell.resCellAccess * cell.writePulseWidth * numCol; 

				writeDynamicEnergyArray += cell.writeVoltage * (cell.writeVoltage-0.25) * (param->maxConductance + param->minConductance)/2 * cell.writePulseWidth * numCol;
				// SET
				writeDynamicEnergyArray += capRow2 * tech.vdd * tech.vdd;													// Selected WL
				writeDynamicEnergyArray += capCol * cell.writeVoltage * cell.writeVoltage * numCol/2;	                    // Unselected SLs
				writeDynamicEnergyArray += capRow1 * cell.writeVoltage * cell.writeVoltage * numCol/2;					// Selected BL
			}
			if (cell.memCellType == Type::SOTMRAM) {
				// SET				
				writeDynamicEnergyArray += cell.setCurrent * cell.setCurrent * cell.resistanceSOT * cell.writePulseWidth * numCol;
				writeDynamicEnergyArray += cell.setCurrentMTJ * cell.setCurrentMTJ * (cell.resCellAccessWrite + 0.5 * (param->resistanceOn + param->resistanceOff)) * cell.writePulseWidth;                   				
				}
			writeDynamicEnergy = wllevelshifter.writeDynamicEnergy + bllevelshifter.writeDynamicEnergy + sllevelshifter.writeDynamicEnergy; 
			writeDynamicEnergy += wlDecoder.writeDynamicEnergy + wlNewDecoderDriver.writeDynamicEnergy + wlDecoderDriver.writeDynamicEnergy + slSwitchMatrix.writeDynamicEnergy;
			writeDynamicEnergy += writeDynamicEnergyArray;
			
			// Leakage
			leakage = wlDecoder.leakage + wlDecoderDriver.leakage + wlNewDecoderDriver.leakage + slSwitchMatrix.leakage + senseAmp.leakage;					

		} 
	}
}


