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

#include <cstdio>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "constant.h"
#include "formula.h"
#include "Param.h"
#include "ProcessingUnit.h"
#include "SubArray.h"
#include "Definition.h"

using namespace std;

int main(int argc, char * argv[]) {   
	setting(argv);
	int numSubArrayRow = 4; //4;	// 256KB; each subarray 256*512, 4*4 subarrays
	int numSubArrayCol = 4;
	param->numRowSubArray = 256; //256;
	param->numColSubArray = 512;
	SubArray *subArrayInPE;
	subArrayInPE = new SubArray(inputParameter, tech, cell);

	ProcessingUnitInitialize(subArrayInPE, inputParameter, tech, cell, numSubArrayRow, numSubArrayCol); 
			
	double Area = ProcessingUnitCalculateArea(subArrayInPE, numSubArrayRow, numSubArrayCol);	
	
	double readLatency, readDynamicEnergy, writeLatency, writeDynamicEnergy, leakage, refreshDynamicEnergy, refreshLatency;
	ProcessingUnitCalculatePerformance(subArrayInPE, numSubArrayRow, numSubArrayCol, cell, &readLatency, &readDynamicEnergy, &writeLatency, &writeDynamicEnergy, &leakage, &refreshDynamicEnergy, &refreshLatency);
	
	cout<<"----------------------NeuroSIM results------------------------"<<endl;
	cout<<"Area: "<<Area*1e12<<"um2"<<endl;
	// cout<<"readLatency: "<<readLatency*1e9<<"ns"<<endl;
	cout<<"readDynamicEnergy: "<<readDynamicEnergy*1e12<<"pJ"<<endl;
	// cout<<"writeLatency: "<<writeLatency*1e9<<"ns"<<endl;
	cout<<"writeDynamicEnergy: "<<writeDynamicEnergy*1e12<<"pJ"<<endl;
	cout<<"leakage: "<<leakage*1e6<<"uW"<<endl;
	cout<<"refreshDynamicEnergy: "<<refreshDynamicEnergy*1e9<<"nJ"<<endl;
	cout<<endl;

	ofstream breakdownfile;
	breakdownfile.open ("./neurosim.csv");
	if (breakdownfile.is_open()) {
		breakdownfile << Area<< "," << readDynamicEnergy << "," << writeDynamicEnergy << "," <<leakage << "," << param->retentionTime << "," << refreshDynamicEnergy << endl;
	}
	breakdownfile.close();
}

void setting(char * argv[]) {
	string memcelltype = argv[1];
	param->memCell3D = false;
	
	if (memcelltype == "3nm_SRAM") {
		param->memcelltype = 1;
		param->technode = 3;
		param->widthSRAMCellNMOS = 1;                            
		param->widthSRAMCellPMOS = 1;
		param->widthAccessCMOS = 1;
		param->heightInFeatureSizeSRAM = 30;        // SRAM Cell height in feature size  
		param->widthInFeatureSizeSRAM = 68.26;        // SRAM Cell width in feature size  
	} else if (memcelltype == "7nm_SRAM") {
		param->memcelltype = 1;
		param->technode = 7;
		param->widthSRAMCellNMOS = 1;                            
		param->widthSRAMCellPMOS = 1;
		param->widthAccessCMOS = 1;
		param->heightInFeatureSizeSRAM = 16;        // SRAM Cell height in feature size  
		param->widthInFeatureSizeSRAM = 34.43;        // SRAM Cell width in feature size  
	} else if (memcelltype == "22nm_SRAM") {
		param->memcelltype = 1;
		param->technode = 22;
		param->heightInFeatureSizeSRAM = 8;        // SRAM Cell height in feature size  
		param->widthInFeatureSizeSRAM = 24;        // SRAM Cell width in feature size  
		param->widthSRAMCellNMOS = 2;                            
		param->widthSRAMCellPMOS = 1;
		param->widthAccessCMOS = 1;
	} else if (memcelltype == "22nm_eDRAM") {
		param->memcelltype = 6;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 48;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 16;         // 1T1R Cell width in feature size
		param->capDRAMCell = 0.5e-15; 				// F
		param->retentionTime = 30e-6;				// s
		param->writeVoltage = tech.vdd;
		param->widthAccessCMOS = 15;
		param->widthAccessCMOSWrite = 6;
	} else if (memcelltype == "22nm_eDRAM_IWO_slow") {
		param->memcelltype = 6;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 5;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 23;         // 1T1R Cell width in feature size
		param->capDRAMCell = 1e-15; 				// F
		param->retentionTime = 1;				// s
		param->writeVoltage = tech.vdd;
		param->widthAccessCMOS = 23;
		param->widthAccessCMOSWrite = 6;
		param->memCell3D = true;
	} else if (memcelltype == "22nm_eDRAM_IWO_fast") {
		param->memcelltype = 6;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 5;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 23;         // 1T1R Cell width in feature size
		param->capDRAMCell = 1e-15; 				// F
		param->retentionTime = 1;				// s
		param->writeVoltage = 1.6;
		param->widthAccessCMOS = 23;
		param->widthAccessCMOSWrite = 6;
		param->memCell3D = true;
	} else if (memcelltype == "3D_eDRAM_IWO_slow") {
		param->memcelltype = 6;
		param->technode = 3;	// 7
		param->heightInFeatureSize1T1R = 5;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 23;         // 1T1R Cell width in feature size
		param->capDRAMCell = 1e-15; 				// F
		param->retentionTime = 1;				// s
		param->writeVoltage = tech.vdd;
		param->widthAccessCMOS = 23;
		param->widthAccessCMOSWrite = 6;
		param->memCell3D = true;
	} else if (memcelltype == "3D_eDRAM_IWO_fast") {
		param->memcelltype = 6;
		param->technode = 3;	// 7
		param->heightInFeatureSize1T1R = 5;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 23;         // 1T1R Cell width in feature size
		param->capDRAMCell = 1e-15; 				// F
		param->retentionTime = 1;				// s
		param->writeVoltage = 1.6;
		param->widthAccessCMOS = 23;
		param->widthAccessCMOSWrite = 6;
		param->memCell3D = true;
	} else if (memcelltype == "22nm_SOTMRAM") {
		param->memcelltype = 5;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 14.3;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 9.8;         // 1T1R Cell width in feature size
		param->resistanceOn = 10e3;              
		param->resistanceOff = 20.4e3;        
		param->resistanceSOT = 670;
		param->readVoltage = 0.5;
		param->widthAccessCMOS = 8.27;
		param->widthAccessCMOSWrite = 33.09;
		param->writeVoltage = tech.vdd;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 330e-6;					// A
		// param->setCurrentMTJ = 330e-6;					// A
		param->writePulseWidth = 1e-9;				// s
	} else if (memcelltype == "7nm_SOTMRAM") {
		param->memcelltype = 5;
		param->technode = 7;
		param->heightInFeatureSize1T1R = 25.7;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 16;         // 1T1R Cell width in feature size
		param->resistanceOn = 3910;              
		param->resistanceOff = 8200;        
		param->resistanceSOT = 780;
		param->readVoltage = 0.5;
		param->widthAccessCMOS = 8.57;
		param->widthAccessCMOSWrite = 17.14;
		param->writeVoltage = tech.vdd;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 500e-6;					// A
		// param->setCurrentMTJ = 330e-6;					// A
		param->writePulseWidth = 1e-9;				// s
	} else if (memcelltype == "3nm_SOTMRAM") {
		param->memcelltype = 5;
		param->technode = 3;
		param->heightInFeatureSize1T1R = 25.7*2;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 16*2;         // 1T1R Cell width in feature size
		param->resistanceOn = 3910;              
		param->resistanceOff = 8200;        
		param->resistanceSOT = 780;
		param->readVoltage = 0.5;
		param->widthAccessCMOS = 8.57;
		param->widthAccessCMOSWrite = 17.14;
		param->writeVoltage = tech.vdd;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 500e-6;					// A
		// param->setCurrentMTJ = 330e-6;					// A
		param->writePulseWidth = 1e-9;				// s
	} else if (memcelltype == "22nm_STTMRAM_eFlash") {
		param->memcelltype = 4;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 10;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 10;         // 1T1R Cell width in feature size
		param->resistanceOn = 1.4e3;              
		param->resistanceOff = 3.9e3;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 11.25;
		param->writeVoltage = 1.5;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 321e-6;					// A
		param->writePulseWidth = 20e-9;				// s
	} else if (memcelltype == "7nm_STTMRAM_eFlash") {
		param->memcelltype = 4;
		param->technode = 7;
		param->heightInFeatureSize1T1R = 16;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 13;         // 1T1R Cell width in feature size
		param->resistanceOn = 2920;              
		param->resistanceOff = 8180;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 17.14;
		param->writeVoltage = 1.0;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 98.7e-6;					// A
		param->writePulseWidth = 20e-9;				// s
	} else if (memcelltype == "3nm_STTMRAM_eFlash") {
		param->memcelltype = 4;
		param->technode = 3;
		param->heightInFeatureSize1T1R = 16*2;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 13*2;         // 1T1R Cell width in feature size
		param->resistanceOn = 2920;              
		param->resistanceOff = 8180;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 17.14;
		param->writeVoltage = 1.0;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 98.7e-6;					// A
		param->writePulseWidth = 20e-9;				// s
	} else if (memcelltype== "22nm_STTMRAM_LLC") {
		param->memcelltype = 4;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 10;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 10;         // 1T1R Cell width in feature size
		param->resistanceOn = 2.5e3;              
		param->resistanceOff = 7e3;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 6;
		param->writeVoltage = 1.0;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 215e-6;					// A
		param->writePulseWidth = 10e-9;				// s
	} else if (memcelltype == "7nm_STTMRAM_LLC") {
		param->memcelltype = 4;
		param->technode = 7;
		param->heightInFeatureSize1T1R = 16;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 13;         // 1T1R Cell width in feature size
		param->resistanceOn = 4440;              
		param->resistanceOff = 12440;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 17.14;
		param->writeVoltage = 1.0; //tech.vdd;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 80.75e-6;					// A
		param->writePulseWidth = 10e-9;				// s
	} else if (memcelltype == "3nm_STTMRAM_LLC") {
		param->memcelltype = 4;
		param->technode = 3;
		param->heightInFeatureSize1T1R = 16*2;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 13*2;         // 1T1R Cell width in feature size
		param->resistanceOn = 4440;              
		param->resistanceOff = 12440;    
		param->readVoltage = 0.5;   
		param->widthAccessCMOS = 17.14;
		param->writeVoltage = 1.0; //tech.vdd;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 80.75e-6;					// A
		param->writePulseWidth = 10e-9;				// s
	} else if (memcelltype == "22nm_FeFET_IWO") {
		param->memcelltype = 3;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 4;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 6;         // 1T1R Cell width in feature size
		param->resistanceOn = 274725;              
		param->resistanceOff = 35.7e9;    
		param->readVoltage = 0.2;   
		param->widthAccessCMOS = 25;
		param->writeVoltage = 1.6;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 10e-6;					// A
		param->writePulseWidth = 20e-9;				// s
		param->gateCapFeFET = 0.17e-15;	        // Gate capacitance of FeFET (F)
		param->polarization = 20;                  // polarization of FeFET (uC/cm^2)
		param->memCell3D = true;
	} else if (memcelltype == "3D_FeFET_IWO") {
		param->memcelltype = 3;
		param->technode = 3;
		param->heightInFeatureSize1T1R = 4;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 6;         // 1T1R Cell width in feature size
		param->resistanceOn = 274725;              
		param->resistanceOff = 35.7e9;    
		param->readVoltage = 0.2;   
		param->widthAccessCMOS = 25;
		param->writeVoltage = 1.6;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 10e-6;					// A
		param->writePulseWidth = 20e-9;				// s
		param->gateCapFeFET = 0.17e-15;	        // Gate capacitance of FeFET (F)
		param->polarization = 20;                  // polarization of FeFET (uC/cm^2)
		param->memCell3D = true;
	} else if (memcelltype == "22nm_FeMFET") {
		param->memcelltype = 3;
		param->technode = 22;
		param->heightInFeatureSize1T1R = 20;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 10;         // 1T1R Cell width in feature size
		param->resistanceOn = 50e3;              
		param->resistanceOff = 500e3;    
		param->readVoltage = 0.1;   
		param->widthAccessCMOS = 25;
		param->writeVoltage = 1.8;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 1.15e-6;					// A
		param->writePulseWidth = 100e-9;				// s
		param->gateCapFeFET = 210e-15;	        // Gate capacitance of FeFET (F)
		param->polarization = 10;                  // polarization of FeFET (uC/cm^2)
	} else if (memcelltype == "3D_FeMFET") {
		param->memcelltype = 3;
		param->technode = 3;
		param->heightInFeatureSize1T1R = 20;        // 1T1R Cell height in feature size
		param->widthInFeatureSize1T1R = 10;         // 1T1R Cell width in feature size
		param->resistanceOn = 50e3;              
		param->resistanceOff = 500e3;    
		param->readVoltage = 0.1;   
		param->widthAccessCMOS = 25;
		param->writeVoltage = 1.8;			//VoltageDropAccessDevice? = 0.25V for all
		param->setCurrent = 1.15e-6;					// A
		param->writePulseWidth = 100e-9;				// s
		param->gateCapFeFET = 210e-15;	        // Gate capacitance of FeFET (F)
		param->polarization = 10;  
		param->memCell3D = true;
	} else{
		exit(-1);
	}
}