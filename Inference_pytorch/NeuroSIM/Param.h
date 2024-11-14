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

using namespace std;

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

	// 1.4+ DCIM

	// file name 
	string filename;

	// parasitics, buffer
	double addedbufferlatency=0;
	double switchmatrixsizeratio_DCIM=0; 
	double DCIMline_R1 = 0;
	double DCIMline_C1 = 0;
	double DCIMline_R2 = 0;
	double DCIMline_C2 = 0;
	double resRow_DCIM = 0;
	double resCol_DCIM = 0;
	double finalCellWidth=0;
	double finalCellHeight=0;
	double DCIM_wsparsity=0.5;
	
	double parallel_weightprecision;
	double DCIM_energy_recalculated=0;

	int toggleindex=0;
	int toggle_enforce=0;
	int realtime_toggle =0;

	double togglerows [6]= {16.0, 32.0, 64.0, 128.0, 192.0, 256.0};
	double repeaterfinnum = 0; 
	double dff_cap;
	double targetdriveres=0;
	double sectionresistance=0;
	double finfetonresistance = 0;
	double finfetswitchmatrixdrive = 0;
	
	double capNORInput = 0;
	double capNOROutput = 0;
	
	double zeroweightcount =0;
	double weightcount =0;
	double sparsitycalc=0;
	double numcount=0;
	double unitactivity=0;
	double averageTR=0;


	// 1.4+ DCIM
	// energy breakdown extraction
	double totalsubarrayoperations=0;
	double Macrototaldynamic=0;
	double Macrototallatency=0;
	double Macroleakage=0;
	double subarray_leakage;
	double SRAM_only_leakage;

	double macro_readDynamicEnergyShiftAdd ;
	double macro_readDynamicEnergyAccum;
	double macro_readDynamicEnergyAdder;
	double macro_readDynamicEnergyBuffer;
	double macro_readDynamicSwitchmatrix;
	double macro_readDynamicInterconnect;
	double macro_readDynamicDFFinShiftAdd;
	double macro_readDynamicADC;

	double SRAM_peri_leakage;
	double DCIM_array_leakage;
	double ShiftAdd_leakage;
	double Adder_leakage;
	double Buffer_leakage;
	double Switchmatrix_leakage;
	double repeater_leakage=0;
	double subarray_selfleakage =0;
	double PE_selfleakage =0;
	double Tile_selfleakage =0; // can later be extended to have values for each lyaer
	double busleakage;

	// adder cap monitoring for external energy simulation

	double adderinputcap;
	double adderoutputcap;

	// area benchmarking
	double subarray_height;
	double subarray_width;
	double subarray_usedarea;
	double addertree_width;
	double wlSwitchMatrix_width ;
	double wlDecoder_width  ;
	double shiftAddInput_width ;
	double sramWriteDriver_height ;
	double precharger_height  ;
	double senseAmp_height ;
	double dff_width  ;
	double Inputdff_width;
	double addertree_area;

	// latency benchmarking
	double addertree_delay =0  ;
	double precharge_delay=0;
	double ADC_delay=0;
	double multisense_delay=0;
	double WL_delay=0;
	double multiencoder_delay=0;
	double mux_delay=0;
	double col_delay;
	double NOR_delay=0;
	double row_delay=0;

	// others
	double capcell_SRAM=0;
	double capcell_row_SRAM=0;
	double rescell_SRAM=0;
	double mux_res=0;
	

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
