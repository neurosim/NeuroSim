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
#include "Param.h"
#include "MultilevelSenseAmp.h"

using namespace std;

extern Param *param;

MultilevelSenseAmp::MultilevelSenseAmp(const InputParameter& _inputParameter, const Technology& _tech, const MemCell& _cell): inputParameter(_inputParameter), tech(_tech), cell(_cell), currentSenseAmp(_inputParameter, _tech, _cell), FunctionUnit() {
	initialized = false;
}

void MultilevelSenseAmp::Initialize(int _numCol, int _levelOutput, double _clkFreq, int _numReadCellPerOperationNeuro, bool _parallel, bool _currentMode) {
	if (initialized) {
		cout << "[MultilevelSenseAmp] Warning: Already initialized!" << endl;
    } else {
		numCol = _numCol;
		levelOutput = _levelOutput;                // # of bits for A/D output ... 
		clkFreq = _clkFreq;
		numReadCellPerOperationNeuro = _numReadCellPerOperationNeuro;
		parallel = _parallel;
		currentMode = _currentMode;
		
		// 1.4 update - updated
		if (parallel) {
			for (int i=0; i<levelOutput-1; i++){
				// 1.4 update - mincon

					if (cell.memCellType == Type::SRAM) {

					if (param->operationmode == 2) {
					double R_start = (double) param->resCellAccess / param->numRowParallel;
					
					double G_start = 1/R_start;
					
					double G_index = 0 - G_start;
					double R_this = 1/ (G_start + (double) (i)*G_index/levelOutput);
					Rref.push_back(R_this);
					}

	
					} else {

					if (param->operationmode == 2) {

					double R_start = (double) param->resistanceOn / param->numRowParallel;
					
					double G_start = 1/R_start;
					
					double G_index = 0 - G_start;
					double R_this = 1/ (G_start + (double) (i)*G_index/levelOutput);
					Rref.push_back(R_this);

					} else {

					double R_start = (double) param->resistanceOn ;
					double R_end = (double) param->resistanceOff ;
					double G_start = 1/R_start;
					double G_end = 1/R_end;
					double G_index = G_end - G_start;
					double R_this = 1/ (G_start + (double) (i)*G_index/levelOutput);
					Rref.push_back(R_this);

					}
				}

			} // TODO: Nonlinear Quantize
		} else {
			for (int i=0; i<levelOutput-1; i++){
				
				// 1.4 update
				double R_start = (double) param->resistanceOn ;
				double R_end = (double) param->resistanceOff ;
				double G_start = 1/R_start;
				double G_end = 1/R_end;
				double G_index = G_end - G_start;
				double G_offset = G_index/(2*(levelOutput-1));
				double R_this = 1/ (G_start + G_offset  + (double) (i)*G_index/(levelOutput-1));
				Rref.push_back(R_this);

			} // TODO: Nonlinear Quantize
		}
		widthNmos = MIN_NMOS_SIZE * tech.featureSize;
		widthPmos = tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

		// 1.4 update - updated 
		double widthNandN = 4 * MIN_NMOS_SIZE * tech.featureSize;
		double widthNandP = 2 * tech.pnSizeRatio * MIN_NMOS_SIZE * tech.featureSize;

		// Initialize SenseAmp
		currentSenseAmp.Initialize((levelOutput-1)*numCol, false, false, clkFreq, numReadCellPerOperationNeuro);        // use real-traced mode ... 


		if ((tech.featureSize == 2e-9) && param->speciallayout) { 
			CalculateGateCapacitance_GAA(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &gatecap_senseamp_P, &junctioncap_senseamp_P, 1.0/2.0,  4.5/15.0,  4.5/15.0);
			CalculateGateCapacitance_GAA(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &gatecap_senseamp_N, &junctioncap_senseamp_N, 1.0/2.0,  4.5/15.0,  4.5/15.0);
		}
	    else if ((tech.featureSize == 1e-9) && param->speciallayout) {
			CalculateGateCapacitance_GAA(INV, 1,0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &gatecap_senseamp_P, &junctioncap_senseamp_P, 10.5/16.0,  4.5/10.0,  4.5/10.0); 
			CalculateGateCapacitance_GAA(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &gatecap_senseamp_N,  &junctioncap_senseamp_N, 10.5/16.0,  4.5/10.0,  4.5/10.0); 
		}
		else {
			CalculateGateCapacitance(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech,  &gatecap_senseamp_P, &junctioncap_senseamp_P);
			CalculateGateCapacitance(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech,  &gatecap_senseamp_N, &junctioncap_senseamp_N);
		}

		if ((tech.featureSize == 2e-9) && param->speciallayout) { 
			CalculateGateCapacitance_GAA(NAND, 2, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &capNandInput, &capNandOutput, 1.0, 22.0/15.0, 8.0/15.0); }
	    else if ((tech.featureSize == 1e-9) && param->speciallayout) {
			CalculateGateCapacitance_GAA(NAND, 2, MIN_NMOS_SIZE * tech.featureSize, MIN_NMOS_SIZE * tech.featureSize, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &capNandInput, &capNandOutput, 1.0, 23.0/15.0, 7.0/15.0); }
		else {
			CalculateGateCapacitance(NAND, 2, widthNandN, widthNandP, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &capNandInput, &capNandOutput);
		}

		initialized = true;
	}
}

void MultilevelSenseAmp::CalculateArea(double heightArray, double widthArray, AreaModify _option) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		
		area = 0;
		height = 0;
		width = 0;
		
		double hNmos, wNmos, hPmos, wPmos;
		CalculateGateArea(INV, 1, widthNmos, 0, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hNmos, &wNmos);
		CalculateGateArea(INV, 1, 0, widthPmos, tech.featureSize*MAX_TRANSISTOR_HEIGHT, tech, &hPmos, &wPmos);
		
		if (widthArray && _option==NONE) {
			if (currentMode) {
				area += ((hNmos*wNmos)*9)*(levelOutput-1)*numCol;
				area += param->arrayheight * param->arraywidthunit *(levelOutput-1)*numCol/param->dumcolshared;
			} else {
				area += ((hNmos*wNmos)*9)*(levelOutput-1)*numCol;
				area += param->arrayheight * param->arraywidthunit *(levelOutput-1)*numCol/param->dumcolshared;
			}
			width = widthArray;
			height = area / width;
		} else if (heightArray && _option==NONE) {
			if (currentMode) {
				area += ((hNmos*wNmos)*9)*(levelOutput-1)*numCol;
				area += param->arrayheight * param->arraywidthunit *(levelOutput-1)*numCol/param->dumcolshared;
			} else {
				area += ((hNmos*wNmos)*9)*(levelOutput-1)*numCol;
				area += param->arrayheight * param->arraywidthunit *(levelOutput-1)*numCol/param->dumcolshared;
			}
			height = heightArray;
			width = area / height;
		} else {
			cout << "[MultilevelSenseAmp] Error: No width or height assigned for the multiSenseAmp circuit" << endl;
			exit(-1);
		}
		// Assume the Current Mirrors are on the same row and the total width of them is smaller than the adder or DFF
		
		// Modify layout
		newHeight = heightArray;
		newWidth = widthArray;
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

void MultilevelSenseAmp::CalculateLatency(const vector<double> &columnResistance, double numColMuxed, double numRead) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		readLatency = 0;
		double LatencyCol = 0;

		double latency_minrefcon = ColumnLatency_Table(Rref [Rref.size()-1], param->technode);
		double latency_maxrefcon = ColumnLatency_Table(Rref [1], param->technode) ;

		LatencyCol = max(latency_minrefcon, latency_maxrefcon);

		if (currentMode) {
			readLatency = LatencyCol*numColMuxed;
			readLatency *= numRead;
		} else {
			readLatency = (1e-9)*numColMuxed;
			readLatency *= numRead;
		}
		
	}
}

void MultilevelSenseAmp::CalculatePower(const vector<double> &columnResistance, double numRead) {
	if (!initialized) {
		cout << "[MultilevelSenseAmp] Error: Require initialization first!" << endl;
	} else {
		leakage = 0;
		readDynamicEnergy = 0;
		
		double LatencyCol = 0;
		for (double j=0; j<columnResistance.size(); j++){
			double T_Col = 0;
			T_Col = GetColumnLatency(columnResistance[j]);
			LatencyCol = max(LatencyCol, T_Col);
			if (LatencyCol < 1e-9) {
				LatencyCol = 1e-9;
			} else if (LatencyCol > 10e-9) {
				LatencyCol = 10e-9;
			}
		}


			double latency_minrefcon = ColumnLatency_Table(Rref [Rref.size()-1], param->technode);
			double latency_maxrefcon = ColumnLatency_Table(Rref [1], param->technode) ;
			LatencyCol = max(latency_minrefcon, latency_maxrefcon);
			double refpower = 0;

			// 1.4 update - reference column power
			for (double ii=1; ii<Rref.size()-1; ii++) {
				refpower += GetColumnPower(Rref[ii])/ (param->dumcolshared);
			}

			for (double i=0; i<columnResistance.size(); i++) {
				double P_Col = 0;
				P_Col = GetColumnPower(columnResistance[i]);
				P_Col += refpower;
				if (currentMode) {

					// 1.4 update - power (dynamic)

					double Vread = 0;

					if (param->technode == 130) {Vread=0.58;}

					else if (param->technode == 90) {Vread=0.58;}

					else if (param->technode == 65) {Vread=0.55;}

					else if (param->technode == 45) {Vread=0.51;}

					else if (param->technode == 32) {Vread=0.51;}

					else if (param->technode == 22) {Vread=0.55;}

					else if (param->technode == 14) {Vread=0.277;}

					else if (param->technode == 10) {Vread=0.28;} 

					else if (param->technode == 7) {Vread=0.264;}

					else if (param->technode == 5) {Vread=0.253;}

					else if (param->technode == 3) {Vread=0.248;}

					else if (param->technode == 2) {Vread=0.28;}

					else if (param->technode == 1) {Vread=0.272;}

					double Column_SwitchingE = (param->columncap * 2) * (1/(levelOutput-1)+1/(param->dumcolshared))/2 * pow(Vread,2) // switching at the senseamp-column interface (column cap)
					+ param->reference_energy_peri/(param->dumcolshared) // reference column activation energy
					+ (gatecap_senseamp_N  * 2 + (junctioncap_senseamp_N+gatecap_senseamp_N)/(levelOutput-1) +  (junctioncap_senseamp_N+gatecap_senseamp_N)/(param->dumcolshared)) * pow(Vread,2) // switching at the senseamp-column interface (input/mirror cap)
					+ (gatecap_senseamp_P*2 + gatecap_senseamp_N) * pow(tech.vdd,2)
					+ (gatecap_senseamp_P*3 + gatecap_senseamp_N*3 + (junctioncap_senseamp_P+junctioncap_senseamp_N)*3 + junctioncap_senseamp_P*1  ) * pow(tech.vdd,2) // switching at the senseamp & latch
					+ (gatecap_senseamp_P + gatecap_senseamp_N) * pow(tech.vdd,2) // Interface with the encoding logic
					;

					readDynamicEnergy += Column_SwitchingE * (levelOutput-1);


					readDynamicEnergy += MAX(P_Col*LatencyCol, 0);
			} else {
					// 1.4 update - power (dynamic)

					double Vread = 0;

					if (param->technode == 130) {Vread=0.58;}

					else if (param->technode == 90) {Vread=0.58;}

					else if (param->technode == 65) {Vread=0.55;}

					else if (param->technode == 45) {Vread=0.51;}

					else if (param->technode == 32) {Vread=0.51;}

					else if (param->technode == 22) {Vread=0.55;}

					else if (param->technode == 14) {Vread=0.277;}

					else if (param->technode == 10) {Vread=0.28;} 

					else if (param->technode == 7) {Vread=0.264;}

					else if (param->technode == 5) {Vread=0.253;}

					else if (param->technode == 3) {Vread=0.248;}

					else if (param->technode == 2) {Vread=0.28;}

					else if (param->technode == 1) {Vread=0.272;}

					double Column_SwitchingE = (param->columncap * 2) * (1/(levelOutput-1)+1/(param->dumcolshared))/2 * pow(Vread,2) // switching at the senseamp-column interface (column cap)
					+ param->reference_energy_peri/(param->dumcolshared) // reference column activation energy					
					+ (gatecap_senseamp_N  * 2 + (junctioncap_senseamp_N)/(levelOutput-1) +  (junctioncap_senseamp_N)/(param->dumcolshared)) * pow(Vread,2) // switching at the senseamp-column interface (input/mirror cap)
					// since ref_voltage is assumed to be very small, the energy consumption of the corresponding transistor is negligible
					+ (gatecap_senseamp_P*2 + gatecap_senseamp_N) * pow(tech.vdd,2)
					+ (gatecap_senseamp_P*3 + gatecap_senseamp_N*3 + (junctioncap_senseamp_P+junctioncap_senseamp_N)*3 + junctioncap_senseamp_P*1  ) * pow(tech.vdd,2) // switching at the senseamp & latch
					+ (gatecap_senseamp_P + gatecap_senseamp_N) * pow(tech.vdd,2) // Interface with the encoding logic
					;

					readDynamicEnergy += Column_SwitchingE * (levelOutput-1);

					readDynamicEnergy += MAX(P_Col*1e-9, 0);
			}
		}
		readDynamicEnergy *= numRead;

	}
} 

void MultilevelSenseAmp::PrintProperty(const char* str) {
	FunctionUnit::PrintProperty(str);
}


double MultilevelSenseAmp::GetColumnLatency(double columnRes) {

	// dummy function - this function is no longer used

	double Column_Latency = 0;
	double up_bound = 3, mid_bound = 1.1, low_bound = 0.9;
	double T_max = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if (((double) 1/columnRes == 0) || (columnRes == 0)) {
		Column_Latency = 0;
	} else {
		if (param->deviceroadmap == 1) {  // HP
			Column_Latency = 1e-9;
		} else {                         // LP
			if (param->technode == 130) {
				T_max = (0.2679*log(columnRes/1000)+0.0478)*1e-9;   // T_max = (0.2679*log(R_BL/1000)+0.0478)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.915*pow(ratio,3)-5.3996*pow(ratio,2)+2.4653*ratio+0.3856);  // y = 3.915*x^3-5.3996*x^2+2.4653*x+0.3856;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);  // y = 0.0004*x^4-0.0087*x^3+0.0742*x^2-0.2725*x+1.2211;
						} else if (ratio>up_bound){
							T = T_max * (0.0004*pow(ratio,4)-0.0087*pow(ratio,3)+0.0742*pow(ratio,2)-0.2725*ratio+1.2211);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 90) {
				T_max = (0.0586*log(columnRes/1000)+1.41)*1e-9;   // T_max = (0.0586*log(R_BL/1000)+1.41)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.726*pow(ratio,3)-5.651*pow(ratio,2)+2.8249*ratio+0.3574);    // y = 3.726*x^3-5.651*x^2+2.8249*x+0.3574;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);  // y = 0.0000008*x^4-0.00007*x^3+0.0017*x^2-0.0188*x+0.9835;
						} else if (ratio>up_bound){
							T = T_max * (0.0000008*pow(ratio,4)-0.00007*pow(ratio,3)+0.0017*pow(ratio,2)-0.0188*ratio+0.9835);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 65) {
				T_max = (0.1239*log(columnRes/1000)+0.6642)*1e-9;   // T_max = (0.1239*log(R_BL/1000)+0.6642)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (1.3899*pow(ratio,3)-2.6913*pow(ratio,2)+2.0483*ratio+0.3202);    // y = 1.3899*x^3-2.6913*x^2+2.0483*x+0.3202;
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);   // y = 0.0036*x^4-0.0363*x^3+0.1043*x^2-0.0346*x+1.0512;
						} else if (ratio>up_bound){
							T = T_max * (0.0036*pow(ratio,4)-0.0363*pow(ratio,3)+0.1043*pow(ratio,2)-0.0346*ratio+1.0512);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else if (param->technode == 45 || param->technode == 32) {
				T_max = (0.0714*log(columnRes/1000)+0.7651)*1e-9;    // T_max = (0.0714*log(R_BL/1000)+0.7651)*10^-9;

				for (int i=1; i<levelOutput-1; i++){
					double ratio = Rref[i]/columnRes;
					double T = 0;
					if (ratio >= 20 || ratio <= 0.05) {
						T = 1e-9;
					} else {
						if (ratio <= low_bound){
							T = T_max * (3.7949*pow(ratio,3)-5.6685*pow(ratio,2)+2.6492*ratio+0.4807);    // y = 3.7949*x^3-5.6685*x^2+2.6492*x+0.4807
						} else if (mid_bound <= ratio <= up_bound){
							T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);   // 0.000001*x^4-0.00006*x^3+0.0001*x^2-0.0171*x+1.0057;
						} else if (ratio>up_bound){
							T = T_max * (0.000001*pow(ratio,4)-0.00006*pow(ratio,3)+0.0001*pow(ratio,2)-0.0171*ratio+1.0057);
						} else {
							T = T_max;
						}
					}
					Column_Latency = max(Column_Latency, T);
				}
			} else {   // technode below and equal to 22nm
				Column_Latency = 1e-9;
			}
		}
	}
	return Column_Latency;
}



double MultilevelSenseAmp::GetColumnPower(double columnRes) {
	double Column_Power = 0;
	// in Cadence simulation, we fix Vread to 0.5V, with user-defined Vread (different from 0.5V)
	// we should modify the equivalent columnRes
	columnRes *= 0.5/param->readVoltage;
	if (currentMode) {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				if (param->technode == 130) {
					Column_Power = 19.898*(levelOutput-1)*1e-6;
					Column_Power += 0.17452*exp(-2.367*log10(columnRes));
				} else if (param->technode == 90) {
					Column_Power = 13.09*(levelOutput-1)*1e-6;
					Column_Power += 0.14900*exp(-2.345*log10(columnRes));
				} else if (param->technode == 65) {
					Column_Power = 9.9579*(levelOutput-1)*1e-6;
					Column_Power += 0.1083*exp(-2.321*log10(columnRes));
				} else if (param->technode == 45) {
					Column_Power = 7.7017*(levelOutput-1)*1e-6;
					Column_Power += 0.0754*exp(-2.296*log10(columnRes));
				} else if (param->technode == 32){  
					Column_Power = 3.9648*(levelOutput-1)*1e-6;
					Column_Power += 0.079*exp(-2.313*log10(columnRes));
				} else if (param->technode == 22){   
					Column_Power = 1.8939*(levelOutput-1)*1e-6;
					Column_Power += 0.073*exp(-2.311*log10(columnRes));
				} else if (param->technode == 14){  // dummy values (not supported mode)
					Column_Power = 1.2*(levelOutput-1)*1e-6;
					Column_Power += 0.0584*exp(-2.311*log10(columnRes));
				} else if (param->technode == 10){   // dummy values (not supported mode)
					Column_Power = 0.8*(levelOutput-1)*1e-6;
					Column_Power += 0.0318*exp(-2.311*log10(columnRes));
				} else {   // 7nm // dummy values (not supported mode)
					Column_Power = 0.5*(levelOutput-1)*1e-6;
					Column_Power += 0.0210*exp(-2.311*log10(columnRes));
				}
			} else {                         // LP

				// 1.4 update : leakage power due to the current mirror part

				if (param->technode == 130) {
					if(columnRes<4832){
						Column_Power = -4.12599E-11* columnRes + 1.48536E-06;

					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -2.6617E-07* log(columnRes) + 3.56576E-06;

						
					} else {
						Column_Power = -1.25872E-07* log(columnRes) + 1.81453E-06;

					}						
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 90) {
					if(columnRes<4832){
						Column_Power = -1.67004E-11* columnRes + 8.54958E-07;

					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -1.50719E-07* log(columnRes) + 2.0858E-06	;
					
						
					} else {
						Column_Power = -8.89685E-08* log(columnRes) + 1.3007E-06;
						

					}						
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 65) {
					if(columnRes<4832){
						Column_Power = -6.08647E-12* columnRes + 4.82814E-07;

					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -8.03449E-08* log(columnRes) + 1.16437E-06;
						
						
					} else {
						Column_Power = -6.01171E-08* log(columnRes) + 8.9355E-07;
						

					}						
					Column_Power *= (levelOutput-1);
				} else if (param->technode == 45) {
					if(columnRes<4832){
						Column_Power = -3.96471E-12* columnRes  + 3.44344E-07;
						
					
					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -5.66235E-08* log(columnRes)+ 8.27849E-07;
						

					} else {
						Column_Power = -4.32673E-08* log(columnRes) + 6.46111E-07;
						

					}						
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 32){  
					if(columnRes<4832){
						Column_Power = -6.72783E-12* columnRes + 4.19847E-07;
					
					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -7.28881E-08* log(columnRes) + 1.02677E-06;


					} else {
						Column_Power = -4.35333E-08* log(columnRes)  + 6.4697E-07;
					

					}						
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 22){   
					if(columnRes<4832){
						Column_Power = -2.85373E-11* columnRes + 9.16744E-07;
					
					} 
					else if ( (columnRes>=4832) && (columnRes<379269)) {
						Column_Power = -1.63133E-07* log(columnRes) + 2.1687E-06;					
					} else {
						Column_Power = -5.8712E-08* log(columnRes) + 8.64648E-07;

					}
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 14){

					if(columnRes<7847){
						
					Column_Power =-3.11213E-13* columnRes + 5.34821E-08;
					
					} 
					else {
						
					Column_Power = -9.64978E-09* log(columnRes) + 1.44527E-07;
					}
					Column_Power *= (levelOutput-1);
					
				} else if (param->technode == 10){   

					if(columnRes<20691){
						Column_Power = -3.27225E-13 * columnRes + 5.54082E-08;
					} 
					else {
						Column_Power = -9.99199E-09* log(columnRes) + 1.49539E-07;
					}
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 7){   // 7nm
					if(columnRes<20691){
						Column_Power = -2.62089E-13 * columnRes + 4.7357E-08;
					} 
					else {
						Column_Power = -8.49198E-09 * log(columnRes) + 1.27848E-07;
					}
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 5){   // 5nm
					if(columnRes<20691){
						Column_Power = -1.4179E-13 * columnRes + 3.31434E-08;
					} 
					else {
						Column_Power = -5.76552E-09 * log(columnRes) + 8.89536E-08;
					}
					Column_Power *= (levelOutput-1);
				} else if (param->technode == 3){   // 3nm
					if(columnRes<20691){
						Column_Power = -1.17664E-13 * columnRes + 2.96368E-08;
					} 
					else{
						Column_Power = -5.09202E-09 * log(columnRes) + 7.91852E-08;
					} 
					Column_Power *= (levelOutput-1);

				} else if (param->technode == 2){   // 2nm
					if(columnRes<20691){
						Column_Power = -2.01487E-13 * columnRes + 4.21066E-08;		
					} 
					else {
						Column_Power = -7.43005E-09* log(columnRes) + 1.13414E-07;						
					}
					Column_Power *= (levelOutput-1);
					
				} else {   // 1nm
					if(columnRes<20691){
						Column_Power = -1.68865E-13 * columnRes + 3.80513E-08;
					} 
					else {
						Column_Power = -6.64737E-09* log(columnRes) + 1.02221E-07;								
					}
					Column_Power *= (levelOutput-1);
					
				}		
			}
		}
		
	} else {
		if ((double) 1/columnRes == 0) { 
			Column_Power = 1e-6;
		} else if (columnRes == 0) {
			Column_Power = 0;
		} else {
			if (param->deviceroadmap == 1) {  // HP
				// 1.4 update : leakage power due to the current mirror part
				// -> negligible in the voltage mode case
					Column_Power=0;
			} else {                         // LP
				// 1.4 update : leakage power due to the current mirror part
				// -> negligible in the voltage mode case
					Column_Power=0;
			}
		}
	}
	Column_Power *= (1+1.3e-3*(param->temp-300));
	return Column_Power;
}


double MultilevelSenseAmp:: ColumnLatency_Table(double Res, double tech) {


double latency = 0; 
double x =Res;
double refcap;
double resthreshold; // shows cap dependency above a threshold
double R2; // fitting parameter
double dC; // fitting parameter

			if (currentMode) {
				if (tech == 130) {
					if (Res<1832) {
						latency  = -1.18573E-10* log(x)  + 2.54175E-09; 
					}

					else {
						latency = 3.78659E-14* x + 2.60629E-09; 
					}		
					refcap=26.62E-15;
					dC=2.0E-9;
					R2=0.1E+6;
					resthreshold=1.95E+3;

				}

				else if (tech  == 90) {
					if (Res<4832) {
						latency  = -1.08914E-10* log(x) + 2.43204E-09 ;
					}

					else {
						latency =  2.8688E-14* x + 2.31147E-09 ;
					}
					refcap=18.4E-15;
					dC=1.5E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 65) {
					if (Res<4832) {
						latency  = -8.11436E-11* log(x) + 2.2554E-09 ;
					}

					else {
						latency = 2.12366E-14* x + 2.14339E-09 ;
					}
					refcap=13.3E-15;
					dC=1.2E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 45) {
					if (Res<4832) {
						latency = -1.12976E-10* log(x) + 2.16762E-09;  
					}

					else {
						latency  = 1.54619E-14* x + 1.60263E-09; 
					}
					refcap=9.21E-15;
					dC=0.81E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 32) {
					if (Res<4832) {
						latency = -1.31777E-10* log(x) + 2.03151E-09; 
					}

					else {
						latency  = 1.15509E-14* x + 1.19475E-09;
					}
					refcap=6.55E-15;
					dC=0.55E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 22) {

					if (Res<4832) {
						latency = -1.32118E-10* log(x)+ 1.7552E-09 ;
					}

					else {
						latency = 8.02517E-15 * x + 8.51803E-10 ;
					}
					refcap=4.5E-15;
					dC=0.31E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;					
				}
				
				else if (tech == 14) {

					if (Res<812) {
						
						latency = -7.01209E-11* log(x)  + 1.21771E-09;
					}

					else if ((Res>=812) && (Res<2962)) {
						latency = -2.11644E-11* log(x) + 8.96216E-10 ;
					}

					else {
						latency = 6.18337E-15 *x + 8.17633E-10;
					
					
					}
					refcap=2.86E-15;
					dC=0.29E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech == 10) {

					if (Res<1128) {
						latency = -7.20238E-11* log(x) + 1.18578E-09;
					}
					else if ((Res>=1128) && (Res<4832)) {
						latency = -1.81368E-11* log(x) + 8.16065E-10;
					}
					else {
						latency = 4.72185E-15 * x + 7.28474E-10;
					}
					refcap=2.04E-15;
					dC=0.27E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				} 
				else if (tech  == 7) {

					if (Res<4832) {
						latency = -4.67061E-11* log(x) + 1.03708E-09;
					}

					else {
						latency =  3.72084E-15 * x + 7.02172E-10;
					}
					refcap=1.43E-15;
					dC=0.13E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;

				}
				else if (tech  == 5) {


					if (Res<4832) {
						latency = -4.33009E-11* log(x) + 9.79525E-10;
					}

					else if ((Res>=4832) && (Res<7847)) {
						latency = -3.15954E-12* log(x) + 6.53005E-10;
					}

					else {
						latency =  2.77641E-15 * x + 6.46706E-10;
					}
					refcap=1.02E-15;
					dC=0.09E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech  == 3) {


					if (Res<4832) {
						latency = -3.62939E-11* log(x) + 9.04595E-10;
					}
					else if ((Res>=4832) && (Res<7847)) {
						latency = -8.05167E-12* log(x) + 6.69134E-10;
					}
					else {
						latency = 2.02367E-15 * x + 6.04792E-10;
					}
					refcap=0.61E-15;
					dC=0.05E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech  == 2) {

					if (Res<4832) {
						latency = -3.70119E-11* log(x) + 8.91946E-10;
					}

					else if ((Res>= 4832) && (Res<12742)) {
						latency = -7.45304E-12* log(x) + 6.42667E-10;
						
					}
					else {
						latency = 1.74295E-15 * x + 5.82625E-10;
					}
					refcap=0.4E-15;
					dC=0.034E-9;
					R2=0.1E+6;
					resthreshold=20.06E+3;
				}
				else if (tech  == 1) {

					if (Res<4832) {
						latency = -3.81168E-11 * log(x) + 8.78701E-10;
					}
					else if ((Res>= 4832) && (Res<12742)) {
						latency = -2.01799E-11* log(x) + 7.25185E-10;
					}
					else {
						latency = 1.19486E-15 * x + 5.35253E-10;
					}

					refcap=0.2E-15;
					dC=0.013E-9;
					R2=0.1E+6;
					resthreshold=20.06E+3;
				}
			}

			else {
				if (tech == 130) {
					if (Res<1832) {
						latency = -1.6656E-10* log(x) + 2.86608E-09; 
					}

					else {
						latency = 3.36654E-14* x + 2.33047E-09; 
					}	
					refcap=26.62E-15;
					dC=2.12E-9;
					R2=0.1E+6;
					resthreshold=1.95E+3;				
				}

				else if (tech  == 90) {
					if (Res<1956) {
						latency = -8.13238E-11* log(x) + 1.9981E-09; 
					}

					else {
						latency = 2.76205E-14* x + 1.87593E-09; 
					}
					refcap=18.4E-15;
					dC=1.8E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 65) {
					if (Res<1210) {
						latency = -1.15612E-10* log(x) + 2.08218E-09 ; 
					}
					else if ((Res>=1210) && (Res<3161)) {
						latency = -1.00285E-11* log(x) + 1.35472E-09; 
					}
					else {
						latency = 2.03232E-14* x + 1.61564E-09; 
					}
					refcap=13.3E-15;
					dC=1.3E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 45) {
					if (Res<695) {
						latency = -2.66629E-10* log(x) + 3.07867E-09 ;
					}
					else if ((Res>=695) && (Res<4832)) {
						latency = -6.27624E-11* log(x) + 1.78932E-09; 
					}
					else {
						latency = 1.43578E-14* x  + 1.46251E-09 ; 
					}
					refcap=9.21E-15;
					dC=0.9E-9;
					R2=0.1E+6;
					resthreshold=4.83E+3;
				}

				else if (tech  == 32) {
					if (Res<4832) {
						latency = -1.8746E-10* log(x) + 2.44191E-09 ; 
					}

					else {
						latency = 1.07272E-14* x + 1.05695E-09 ; 
					}
					refcap=6.55E-15;
					dC=0.7E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}

				else if (tech  == 22) {

					if (Res<4832) {
						latency = -2.01894E-10* log(x) + 2.2559E-09 ;
					}

					else {
						latency = 7.45392E-15* x + 6.72844E-10 ;
					}
					refcap=4.5E-15;
					dC=0.47E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				
				else if (tech == 14) {

					if (Res<695) {
						
						latency  = -1.17191E-10* log(x) + 1.51724E-09;
					}

					else if ((Res>=695) && (Res<1832)) {
						latency = -3.56669E-11* log(x) + 1.00049E-09;
					}

					else {
						latency = 5.484E-15*x + 7.4812E-10 ; 
					
					
					}
					refcap=2.86E-15;
					dC=0.34E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech == 10) {

					if (Res<2976) {
						latency = -5.88954E-11* log(x)  + 1.12213E-09;
					}
					else if ((Res>=2976) && (Res<7847)) {
						latency = -1.03546E-11* log(x) + 7.46313E-10;
					}
					else {
						latency = 4.02595E-15* x+ 6.64175E-10 ;
					}
					refcap=2.04E-15;
					dC=0.24E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				} 
				else if (tech  == 7) {

					if (Res<1956) {
						latency = -9.29209E-11* log(x)  + 1.36675E-09;
					}
					else if ((Res>=1956) && (Res<8251)) {
						latency  = -2.32202E-11* log(x)  + 8.55312E-10;
					}
					else {
						latency  = 2.96941E-15* x + 6.3388E-10;
					}
					refcap=1.43E-15;
					dC=0.17E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech  == 5) {


					if (Res<20691.38) {
						latency = -3.58902E-11* log(x) + 9.45003E-10;
					}
					else if ((Res>=20691.38) && (Res<33598.18)) {
						latency = 4.1847E-11* log(x) + 2.03225E-10;
					}
					else {
						latency = 2.17318E-15* x  + 5.90032E-10;
					}
					refcap=1.02E-15;
					dC=0.12E-9;
					R2=0.1E+6;
					resthreshold=12.7E+3;
				}
				else if (tech  == 3) {


					if (Res<20691.38) {
						latency = -4.80094E-11* log(x) + 1.02891E-09;
					}
					else if ((Res>=20691.38) && (Res<33598.18)) {
						latency = 1.46916E-11* log(x) + 4.37023E-10;
					}
					else {
						latency = 1.39879E-15 * x + 5.42026E-10;
					}
					refcap=0.61E-15;
					dC=0.34E-9;
					R2=0.5E+6;
					resthreshold=49.6E+3;
				}
				else if (tech  == 2) {

					if (Res<3161.018) {
						latency = -1.44146E-10 * log(x) + 1.72384E-09; 
					}

					else if ((Res>= 3161.018) && (Res<34798.92)) {
						latency = -3.01414E-11 * log(x)+ 8.19483E-10;
						
					}
					else {
						latency = 6.80209E-16* x + 4.60774E-10;
					}
					refcap=0.4E-15;
					dC=0.23E-9;
					R2=0.5E+6;
					resthreshold=49.6E+3;
				}
				else if (tech  == 1) {

					if (Res<5107) {
						latency = -6.09403E-11* log(x) + 1.06251E-09;
					}
					else if ((Res>= 5107) && (Res<34798.92)) {
						latency = -4.20915E-11* log(x) + 8.98944E-10;
					}
					else {
						latency = 5.44698E-16 * x + 4.38466E-10 ;
					}

					refcap=0.2E-15;
					dC=0.11E-9;
					R2=0.5E+6;
					resthreshold=60.9E+3;
				}

			}
			

	if (x>resthreshold) {
		double columncap = param->columncap + gatecap_senseamp_N * ( Rref.size()-2);
		latency += dC/(R2-resthreshold) * (x-resthreshold) * ((columncap-refcap)/refcap);
	}

	return latency;
}
