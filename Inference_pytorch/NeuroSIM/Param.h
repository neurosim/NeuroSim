/******************out*************************************************************
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

#ifndef PARAM_H_
#define PARAM_H_

class Param {
public:
	Param();

	int operationmode, operationmodeBack, memcelltype, accesstype, transistortype, deviceroadmap;      		
	
	double heightInFeatureSizeSRAM, widthInFeatureSizeSRAM, widthSRAMCellNMOS, widthSRAMCellPMOS, widthAccessCMOS, minSenseVoltage;
	 
	double heightInFeatureSize1T1R, widthInFeatureSize1T1R, heightInFeatureSizeCrossbar, widthInFeatureSizeCrossbar;
	
	int relaxArrayCellHeight, relaxArrayCellWidth;
	// Anni update
	bool globalBusType, globalBufferType, tileBufferType, peBufferType, chipActivation, reLu, novelMapping, pipeline, SARADC, currentMode, validated, synchronous;
	int globalBufferCoreSizeRow, globalBufferCoreSizeCol, tileBufferCoreSizeRow, tileBufferCoreSizeCol;																								
	
	double clkFreq, featuresize, readNoise, resistanceOn, resistanceOff, maxConductance, minConductance;
	int temp, technode, wireWidth, multipleCells;
	double maxNumLevelLTP, maxNumLevelLTD, readVoltage, readPulseWidth, writeVoltage;
	double accessVoltage, resistanceAccess;
	double nonlinearIV, nonlinearity;
	double writePulseWidth, numWritePulse;
	double globalBusDelayTolerance, localBusDelayTolerance;
	double treeFoldedRatio, maxGlobalBusWidth;
	double algoWeightMax, algoWeightMin;
	
	int neuro, multifunctional, parallelWrite, parallelRead;
	int numlut, numColMuxed, numWriteColMuxed, levelOutput, avgWeightBit, numBitInput;
	int numRowSubArray, numColSubArray;
	int cellBit, synapseBit;
	int speedUpDegree;
	
	int XNORparallelMode, XNORsequentialMode, BNNparallelMode, BNNsequentialMode, conventionalParallel, conventionalSequential; 
	int numRowPerSynapse, numColPerSynapse;
	double AR, Rho, wireLengthRow, wireLengthCol, unitLengthWireResistance, wireResistanceRow, wireResistanceCol;
	
	double alpha, beta, gamma, delta, epsilon, zeta;

	// 1.4 update: BEOL related parameters added
	
	double Metal0=0;
	double Metal1=0;
	double AR_Metal0=0;
	double AR_Metal1=0;
	double Rho_Metal0=0;
	double Rho_Metal1=0;
	double Metal0_unitwireresis=0;
	double Metal1_unitwireresis=0;

	// 1.4 update: add activation implementation option

	bool Activationtype; // true: SRAM, False: RRAM

	// 1.4 update: Final driver sizing for row decoder conventional parallel mode (SRAM, RRAM)
	// multiplied by the driver width
	double sizingfactor_MUX= 1; 
	double sizingfactor_WLdecoder= 1; 

	// 1.4 update: switchmatrix parameter tuning
	double newswitchmatrixsizeratio=6;
	double switchmatrixsizeratio=1;
	
	// 1.4 update: Special layout
	double speciallayout;
	
	// 1.4 update: added parameters for buffer insertion
	double unitcap;
	double unitres;
	double drivecapin; 
	double buffernumber=0;
	double buffersizeratio=0;
	
	// 1.4 update: barrier thickness
	double barrierthickness= 0;
	
	// 1.4 update: new ADC modeling related parameters
	double dumcolshared;
	double columncap;
	double reference_energy_peri=0;
	
	// 1.4 update: array dimension/SRAM access resistance for multilevelsenseamp
	double arrayheight;
	double arraywidthunit;
	double resCellAccess;

	// 1.4 update 
	double inputtoggle;
	double outputtoggle;

	// 1.4 debug
	double ADClatency;
	double rowdelay;
	double muxdelay;
	
	// 1.4 update: technology node
	int technologynode;

	// Anni update: partial parallel mode
	int numRowParallel;

	// 230920 update
	double totaltile_num;
	int sync_data_transfer;
};

#endif