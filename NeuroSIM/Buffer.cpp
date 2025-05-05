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
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "Buffer.h"

using namespace std;

extern Param *param;


Buffer::Buffer(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), 
                      wlDecoder(_inputParameter, _tech, _cell), 
					  precharger(_inputParameter, _tech, _cell), 
					  sramWriteDriver(_inputParameter, _tech, _cell), 
					  senseAmp(_inputParameter, _tech, _cell), 
					  dff(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void Buffer::Initialize(int _numBit, int _interface_width, int _num_interface, double _unitWireRes, double _clkFreq, bool _SRAM){
	if (initialized)
		cout << "[Buffer] Warning: Already initialized!" << endl;
	
	numBit = _numBit;                             // # of bits that Buffer can store
	interface_width = _interface_width;           // # of bits in a "line", normally refered as # of column
	num_interface = _num_interface;               // # of interface
	unitWireRes = _unitWireRes;                   // Wire unit Resistance
	clkFreq = _clkFreq;                           // assigned clock frequency
	SRAM = _SRAM;                                 // SRAM based or DFF based?
	
	// Anni update: 1.4 update add variables
	numCol= interface_width;
	numRow= (double)ceil((double)numBit/(double)interface_width);
	
	if (SRAM) {
		lengthRow = (double)interface_width * param->widthInFeatureSizeSRAM * tech.featureSize;
		lengthCol = (double)ceil((double)numBit/(double)interface_width) * param->heightInFeatureSizeSRAM * tech.featureSize;

		capRow1 = lengthRow * 0.2e-15/1e-6;	// BL for 1T1R, WL for Cross-point and SRAM
		capCol = lengthCol * 0.2e-15/1e-6;

		// 1.4 update
		resCellAccess = CalculateOnResistance(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
		capCellAccess = CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech);
		capSRAMCell = capCellAccess + CalculateDrainCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateDrainCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, PMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) 
						+ CalculateGateCap(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) + CalculateGateCap(cell.widthSRAMCellPMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech);
		capRow1 += 2*CalculateGateCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, tech) * numCol;          //sum up all the gate cap of access CMOS, as the row cap
		if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;
		else capCol += CalculateDrainCap(cell.widthAccessCMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, MAX_TRANSISTOR_HEIGHT * tech.featureSize, tech) * numRow;	
		
		resRow = lengthRow * param->Metal0_unitwireresis;
		resCol = lengthCol * param->Metal1_unitwireresis;

		// 1.4 update: consider overlap capacitance for FinFET
		// if (tech.featureSize <= 14 * 1e-9) capCol += tech.cap_draintotal * cell.widthAccessCMOS * tech.effective_width * numRow;	
				
		precharger.Initialize(interface_width, lengthCol * unitWireRes, 1, interface_width, interface_width);
		sramWriteDriver.Initialize(interface_width, 1, interface_width);

		// 1.4 update: add senseamp
		senseAmp.Initialize(interface_width, false, cell.minSenseVoltage, lengthRow/interface_width, clkFreq, interface_width);

	} else {
		dff.Initialize(numBit, clkFreq);
	}
	
	wlDecoder.Initialize(REGULAR_ROW, (int)ceil((double)log2((double)ceil((double)numBit/(double)interface_width))), false, false);
	
	initialized = true;
}

void Buffer::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		area = 0;
		height = 0;
		width = 0;
		
		if (SRAM) {
			memoryArea = lengthRow * lengthCol;
			wlDecoder.CalculateArea(lengthCol, NULL, NONE);
			precharger.CalculateArea(NULL, lengthRow, NONE);
			sramWriteDriver.CalculateArea(NULL, lengthRow, NONE);
			area += memoryArea + wlDecoder.area + precharger.area + sramWriteDriver.area;
		} else {
			dff.CalculateArea(NULL, NULL, NONE);
			wlDecoder.CalculateArea(dff.hDff*ceil((double)numBit/(double)interface_width), NULL, NONE);
			area += dff.area + wlDecoder.area;
		}

		if (_newWidth && _option==NONE) {
			width = _newWidth;
			height = area/width;
		} else if (_newHeight && _option==NONE) {
			height = _newHeight;
			width = area/height;
		} else {
			cout << "[Buffer] Error: No width assigned for the buffer circuit" << endl;
			exit(-1);
		}
		
		// Modify layout
		newHeight = _newHeight;
		newWidth = _newWidth;
		switch (_option) {
			case MAGIC:
				MagicLayout();
				break;
			case OVERRIDE:
				OverrideLayout();
				break;
			default:    // NONE
				break;
		}
	}
}

void Buffer::CalculateLatency(double numAccessBitRead, double numRead, double numAccessBitWrite, double numWrite){
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		writeLatency = 0;
		readWholeLatency = 0;
		writeWholeLatency = 0;
		
		if (param->synchronous) {
			// 230920 updated 
			readLatency = numRead;		// read 1 line per cycle
			writeLatency = numWrite;
		} else {
			if (SRAM) {
				// 1.4 update
				// row decoder update, update to integer 
				wlDecoder.CalculateLatency(1e20, lengthRow * 0.2e-15/1e-6, NULL, resRow, interface_width, numRow, numRow);
				precharger.CalculateLatency(1e20, lengthCol * 0.2e-15/1e-6, numRow, numRow);
				sramWriteDriver.CalculateLatency(1e20, lengthCol * 0.2e-15/1e-6, lengthCol * unitWireRes, numRow);
				senseAmp.CalculateLatency(numRow);

				// 1.4 update
				double resPullDown = CalculateOnResistance(cell.widthSRAMCellNMOS * ((tech.featureSize <= 14*1e-9)? 2:1) * tech.featureSize, NMOS, inputParameter.temperature, tech);
				double BLCap_perCell = capCol / numRow;
				double BLRes_perCell = resCol / numRow;
				double Elmore_BL = (resCellAccess + resPullDown) * BLCap_perCell * numRow   + BLCap_perCell * BLRes_perCell * numRow  * ( numRow +1 )  /2;
				
				// 1.4 update: to integer 
				colDelay = Elmore_BL * log(tech.vdd / (tech.vdd - cell.minSenseVoltage / 2)) * numRow;

				// 1.4 update: sense amp delay should be included
				readWholeLatency += wlDecoder.readLatency + precharger.readLatency + colDelay +senseAmp.readLatency;

				// 1.4 update: write delay -> needs check
				writeWholeLatency += precharger.writeLatency + max (wlDecoder.writeLatency, sramWriteDriver.writeLatency);

			} else {
				// 1.4 update to integer
				wlDecoder.CalculateLatency(1e20, dff.hDff * interface_width * 0.2e-15/1e-6, resRow, numCol, NULL, numRow, numRow);
				readWholeLatency += wlDecoder.readLatency;
				readWholeLatency += ((double) 1/clkFreq/2)*numRow;  // assume dff need half clock cycle to access
				writeWholeLatency += wlDecoder.writeLatency + ((double) 1/clkFreq/2)*numRow;
			}		
			// 1.4 update to integer		
			avgBitReadLatency = (double) readWholeLatency/numRow;     // average latency per line(sec/line)
			avgBitWriteLatency = (double) writeWholeLatency/numRow;
			readLatency = avgBitReadLatency*numRead;
			writeLatency = avgBitWriteLatency*numWrite;
		}
	}
}

void Buffer::CalculatePower(double numAccessBitRead, double numRead, double numAccessBitWrite, double numWrite) {
	if (!initialized) {
		cout << "[Buffer] Error: Require initialization first!" << endl;
	} else {
		readDynamicEnergy = 0;
		writeDynamicEnergy = 0;
		leakage = 0;
		readWholeDynamicEnergy = 0;
		writeWholeDynamicEnergy = 0;
		
		if (SRAM) {

			// 1.4 update: change to integer 
			wlDecoder.CalculatePower(numRow, numRow);
			precharger.CalculatePower(numRow, numRow);
			sramWriteDriver.CalculatePower(numRow);
			senseAmp.CalculatePower(numRow);

			// 1.4 update
			readWholeDynamicEnergy += wlDecoder.readDynamicEnergy + precharger.readDynamicEnergy + senseAmp.readDynamicEnergy;
			writeWholeDynamicEnergy += wlDecoder.writeDynamicEnergy + precharger.writeDynamicEnergy + sramWriteDriver.writeDynamicEnergy;
			leakage += wlDecoder.leakage + precharger.leakage + sramWriteDriver.leakage + senseAmp.leakage;
			
			// 1.4 update: read/write energy update
			readWholeDynamicEnergy += capRow1 * tech.vdd * tech.vdd * numRow; // added for WL/BL discharging
			writeWholeDynamicEnergy  += cell.capSRAMCell * tech.vdd * tech.vdd * numBit;    // flip Q and Q_bar
			writeWholeDynamicEnergy  += capRow1 * tech.vdd * tech.vdd * numRow;	

		} else {
			wlDecoder.CalculatePower(numBit/interface_width, numBit/interface_width);
			// 230920 update
			dff.CalculatePower(1, numBit, true);
			
			readWholeDynamicEnergy += wlDecoder.readDynamicEnergy + dff.readDynamicEnergy;
			writeWholeDynamicEnergy += wlDecoder.writeDynamicEnergy + dff.writeDynamicEnergy;
			
			// 230920 updated 
			readWholeDynamicEnergy = readWholeDynamicEnergy;
			writeWholeDynamicEnergy = writeWholeDynamicEnergy;

			leakage += dff.leakage;
			leakage += wlDecoder.leakage;
		}
		avgBitReadDynamicEnergy = readWholeDynamicEnergy/numBit;
		avgBitWriteDynamicEnergy = writeWholeDynamicEnergy/numBit;
		
		readDynamicEnergy = avgBitReadDynamicEnergy*numAccessBitRead*numRead;
		writeDynamicEnergy = avgBitWriteDynamicEnergy*numAccessBitWrite*numWrite;
	}
}

void Buffer::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}











