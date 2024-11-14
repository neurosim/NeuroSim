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
#include "AdderTree_DCIM.h"
#include "Param.h"

extern Param *param;

using namespace std;

AdderTree_DCIM::AdderTree_DCIM(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), adder(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void AdderTree_DCIM::Initialize(int _numSubcoreRow, int _numAdderBit, int _numAdderTree, double _clkFreq) {
	if (initialized)
		cout << "[AdderTree_DCIM] Warning: Already initialized!" << endl;
	
	numSubcoreRow = _numSubcoreRow;                  // # of row of subcore in the synaptic core
	numStage = ceil(log2(numSubcoreRow));            // # of stage of the adder tree, used for CalculateLatency ...
	numAdderBit = _numAdderBit;                      // # of input bits of the Adder
	numAdderTree = _numAdderTree;                    // # of Adder Tree
	clkFreq = _clkFreq;
	
	initialized = true;
}

void AdderTree_DCIM::CalculateArea(double _newHeight, double _newWidth, AreaModify _option) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		double hInv, wInv, hNand, wNand;
		area = 0;
		height = 0;
		width = 0;
		// Adder
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = ceil(log2(numSubcoreRow));
		int j = numSubcoreRow;
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			numAdderEachTree += numBitEachStage*numAdderEachStage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
		}
		adder.Initialize(numAdderEachTree, numAdderTree, clkFreq);   
		// dff.Initialize((numAdderBit+numStage)*numAdderTree, clkFreq);
		
		if (_newWidth && _option==NONE) {
			adder.CalculateArea(NULL, _newWidth, NONE);
			width = _newWidth;
			height = adder.area/width;
		} else if (_newHeight && _option==NONE) {
			adder.CalculateArea(_newHeight, NULL, NONE);
			height = _newHeight;
			width = adder.area/height;
		} else {
			cout << "[AdderTree] Error: No width assigned for the adder tree circuit" << endl;
			exit(-1);
		}
		area = height*width;
		adder.initialized = false;
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

void AdderTree_DCIM::CalculateLatency(double numRead, int numUnitAdd, double _capLoad) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}

		numAdderEachStage = ceil(j/2);
		int last =0;

		adder.initialized = false;
		adder.Initialize(numBitEachStage, numAdderEachStage, clkFreq); 
		adder.CalculateLatency(1e20, _capLoad, 1, last);
		readLatency += adder.readLatency;
		numBitEachStage += 1;
		j = ceil(j/2);
		i -= 1;

		adder.initialized = false;
		if (i>0) {
		while (i != 0) {   // calculate the total # of full adder in each Adder Tree

			if (i ==1) {last =1;}
			numAdderEachStage = ceil(j/2);
			
			adder.Initialize(2, numAdderEachStage, clkFreq); 
			
			adder.CalculateLatency(1e20, _capLoad, 1, last);
	
			readLatency += adder.readLatency;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
			adder.initialized = false;
			
		}

		}



        readLatency *= numRead;		
	}
}

void AdderTree_DCIM::CalculatePower(double numRead, int numUnitAdd, double togglingrate) {
	if (!initialized) {
		cout << "[AdderTree] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		int numAdderEachStage = 0;                          // define # of adder in each stage
		int numBitEachStage = numAdderBit;                  // define # of bits of the adder in each stage
		int numAdderEachTree = 0;                           // define # of Adder in each Adder Tree
		int i = 0;
		int j = 0;
		

				
		if (!numUnitAdd) {
			i = ceil(log2(numSubcoreRow));
			j = numSubcoreRow;
		} else {
			i = ceil(log2(numUnitAdd));
			j = numUnitAdd;
		}
		
		while (i != 0) {  // calculate the total # of full adder in each Adder Tree
			numAdderEachStage = ceil(j/2);
			adder.Initialize(numBitEachStage, numAdderEachStage, clkFreq);     
			adder.CalculatePower(1, numAdderEachStage);	
			readDynamicEnergy += adder.readDynamicEnergy;	
			leakage += adder.leakage;
			numBitEachStage += 1;
			j = ceil(j/2);
			i -= 1;
			
			adder.initialized = false;
		}

		// DCIM part
	

		if (param->DCIM_energy_recalculated) {
		// assumes 50 % sparsity for macro-mode
			double energytemp[5];
			double energytemp_system;
			double a,b,c,d; // fitting coefficient

			if (tech.featureSize == 32 * 1e-9){
			a= 1.7954* 1.8;
			b= 4.2176* 1.8;
			c= 0.7073* 1.8;
			d= 2.9301* 1.8;
			if (param->realtime_toggle){
			a= 1.7954* 1.8;
			b= 4.2176* 1.8;
			c= 0.7073* 1.8;
			d= 2.9301* 1.8;

			}

			else {
			energytemp [0]= 0.9e-12 * 1.8;
			energytemp [1]= 0.708e-12 * 1.8;
			energytemp [2]= 1.43e-12 * 1.8;
			energytemp [3]= 1.887e-12 * 1.8;
			energytemp [4]= 0;

			}

			}

			else if (tech.featureSize == 22 * 1e-9){
			a= 1.7954;
			b= 4.2176;
			c= 0.7073;
			d= 2.9301;
			if (param->realtime_toggle){
			a= 1.7954;
			b= 4.2176;
			c= 0.7073;
			d= 2.9301;

			}

			else {
			energytemp [0]= 0.9e-12;
			energytemp [1]= 0.708e-12;
			energytemp [2]= 1.43e-12;
			energytemp [3]= 1.887e-12;
			energytemp [4]= 0;

			}

			}
			else if (tech.featureSize == 14 * 1e-9){
				a= 0.6975;
			b= 1.6302;
			c= 0.2584;
			d= 1.0989;
			if (param->realtime_toggle){

			a= 0.6975;
			b= 1.6302;
			c= 0.2584;
			d= 1.0989;
			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.411e-12;
			energytemp [2]= 0.549e-12;
			energytemp [3]= 0.671e-12;
			energytemp [4]= 0;
			}
			}
			else if (tech.featureSize == 10 * 1e-9){
			a=0.5709;
			b=1.3432;
			c=0.2136;
			d=0.9006;
			if (param->realtime_toggle){

			a=0.5709;
			b=1.3432;
			c=0.2136;
			d=0.9006;
			}

			else {				
			energytemp [0]= 0;
			energytemp [1]= 0.328e-12;
			energytemp [2]= 0.40e-12;
			energytemp [3]= 0.502e-12;
			energytemp [4]= 0;
			}
			}
			else if (tech.featureSize == 7 * 1e-9){

			a= 0.4998;
			b= 1.1733;
			c= 0.1917;
			d= 0.7917;
			if (param->realtime_toggle){
			a= 0.4998;
			b= 1.1733;
			c= 0.1917;
			d= 0.7917;

			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.296e-12;
			energytemp [2]= 0.309e-12;
			energytemp [3]= 0.397e-12;
			energytemp [4]= 0;
			}
			}
			else if (tech.featureSize == 5 * 1e-9){
			a= 0.4335;
			b= 1.0154;
			c= 0.1516;
			d=0.6499;
			if (param->realtime_toggle){

			a= 0.4335;
			b= 1.0154;
			c= 0.1516;
			d=0.6499;
			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.248e-12;
			energytemp [2]= 0.294e-12;
			energytemp [3]= 0.364e-12;
			energytemp [4]= 0;

			}
			}
			else if (tech.featureSize == 3 * 1e-9){
			a= 0.3641;
			b= 0.8561;
			c= 0.1364;
			d= 0.5732;
			if (param->realtime_toggle){
			a= 0.3641;
			b= 0.8561;
			c= 0.1364;
			d= 0.5732;

			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.205e-12;
			energytemp [2]= 0.255e-12;
			energytemp [3]= 0.327e-12;
			energytemp [4]= 0;

			}
			}
			else if (tech.featureSize == 2 * 1e-9){
			/* old values
				a= 0.1596;
				b= 0.373;
				c= 0.0565;
				d=0.2417;
			*/

			a= 0.2274;
			b= 0.5354;
			c= 0.0898;
			d=0.371;
			if (param->realtime_toggle){

			a= 0.2274;
			b= 0.5354;
			c= 0.0898;
			d=0.371;
			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.084e-12;
			energytemp [2]= 0.200e-12;
			energytemp [3]= 0.227e-12;
			energytemp [4]= 0;
			}
			}
			else if (tech.featureSize == 1 * 1e-9){
				/* old values
					a=0.1096;
					b=0.2569;
					c=0.04;
					d=0.1696;
				*/
			a=0.1653;
			b=0.3894;
			c=0.0633;
			d=0.2642;
			if (param->realtime_toggle){

			a=0.1653;
			b=0.3894;
			c=0.0633;
			d=0.2642;
			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0.065e-12;
			energytemp [2]= 0.100e-12;
			energytemp [3]= 0.128e-12;
			energytemp [4]= 0;

			}
			}
			else{

			if (param->realtime_toggle){

			energytemp_system=0;
			}

			else {
			energytemp [0]= 0;
			energytemp [1]= 0;
			energytemp [2]= 0;
			energytemp [3]= 0;
			energytemp [4]= 0;
			}
			}

			if (param->realtime_toggle){

				if (togglingrate<0.1) {
				energytemp_system = a*log(togglingrate+0.1) +b; 
				}

				else if ((togglingrate>=0.1) && (togglingrate<0.8))
				{
				energytemp_system = c*log(togglingrate) +d; 
				} 
				
				else 
				{
				energytemp_system = c*log(0.8) +d; 
				}

			readDynamicEnergy = energytemp_system * pow(10,-12);
			} else {

				if (togglingrate<0.1) {
				energytemp_system = a*log(togglingrate+0.1) +b; 
				}

				else if ((togglingrate>=0.1) && (togglingrate<0.8))
				{
				energytemp_system = c*log(togglingrate) +d; 
				} 
				
				else 
				{
				energytemp_system = c*log(0.8) +d; 
				}
			// readDynamicEnergy = energytemp [param->toggleindex];
			readDynamicEnergy = energytemp_system * pow(10,-12);
			}

			
			

		}

		else{
			readDynamicEnergy *= numAdderTree;	
			readDynamicEnergy *= numRead;
		}

		leakage *= numAdderTree;
	}
}

void AdderTree_DCIM::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}

