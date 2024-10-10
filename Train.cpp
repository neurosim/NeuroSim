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
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <vector>
#include <random>
#include "formula.h"
#include "Param.h"
#include "Array.h"
#include "Mapping.h"
#include "NeuroSim.h"

extern Param *param;

extern std::vector< std::vector<double> > Input;
extern std::vector< std::vector<int> > dInput;
extern std::vector< std::vector<double> > Output;

extern std::vector< std::vector<double> > weight1;
extern std::vector< std::vector<double> > weight2;
extern std::vector< std::vector<double> > deltaWeight1;
extern std::vector< std::vector<double> > deltaWeight2;

extern Technology techIH;
extern Technology techHO;
extern Array *arrayIH;
extern Array *arrayHO;
extern SubArray *subArrayIH;
extern SubArray *subArrayHO;
extern Adder adderIH;
extern Mux muxIH;
extern RowDecoder muxDecoderIH;
extern DFF dffIH;
extern Adder adderHO;
extern Mux muxHO;
extern RowDecoder muxDecoderHO;
extern DFF dffHO;

void Train(const int numTrain, const int epochs) {
	int numBatchReadSynapse;	// # of read synapses in a batch read operation (decide later)
	int numBatchWriteSynapse;	// # of write synapses in a batch write operation (decide later)
	double outN1[param->nHide]; // Net input to the hidden layer [param->nHide]
	double a1[param->nHide];    // Net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	int da1[param->nHide];  // Digitized net output of hidden layer [param->nHide] also the input of hidden layer to output layer
	double outN2[param->nOutput];   // Net input to the output layer [param->nOutput]
	double a2[param->nOutput];  // Net output of output layer [param->nOutput]

	double s1[param->nHide];    // Output delta from input layer to the hidden layer [param->nHide]
	double s2[param->nOutput];  // Output delta from hidden layer to the output layer [param->nOutput]

	for (int t = 0; t < epochs; t++) {
		for (int batchSize = 0; batchSize < numTrain; batchSize++) {

			int i = rand() % param->numMnistTrainImages;  // Randomize sample
			
			// Forward propagation
			/* First layer (input layer to the hidden layer) */
			std::fill_n(outN1, param->nHide, 0);
			std::fill_n(a1, param->nHide, 0);
			if (param->useHardwareInTrainingFF) {   // Hardware
				double sumArrayReadEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double readVoltage = static_cast<eNVM*>(arrayIH->cell[0][0])->readVoltage;
				double readPulseWidth = static_cast<eNVM*>(arrayIH->cell[0][0])->readPulseWidth;
				#pragma omp parallel for reduction(+: sumArrayReadEnergy)
				for (int j=0; j<param->nHide; j++) {
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
						if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * param->nInput; // All WLs open
						}
					} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
						// XXX: To be released
					}
					for (int n=0; n<param->numBitInput; n++) {
						double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayIH->arrayRowSize;  // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							double Isum = 0;    // weighted sum current
							double IsumMax = 0; // Max weighted sum current
							double inputSum = 0;    // Weighted sum current of input vector * weight=1 column
							for (int k=0; k<param->nInput; k++) {
								if ((dInput[i][k]>>n) & 1) {    // if the nth bit of dInput[i][k] is 1
									Isum += arrayIH->ReadCell(j,k);
									inputSum += arrayIH->GetMaxCellReadCurrent(j,k);
									sumArrayReadEnergy += arrayIH->wireCapRow * readVoltage * readVoltage; // Selected BLs (1T1R) or Selected WLs (cross-point)
								}
								IsumMax += arrayIH->GetMaxCellReadCurrent(j,k);
							}
							sumArrayReadEnergy += Isum * readVoltage * readPulseWidth;
							int outputDigits = 2 * CurrentToDigits(Isum, IsumMax) - CurrentToDigits(inputSum, IsumMax);
							outN1[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
						} else {    // SRAM or digital eNVM
							// XXX: To be released
						}
					}
					a1[j] = sigmoid(outN1[j]);
					da1[j] = round_th(a1[j]*(param->numInputLevel-1), param->Hthreshold);
				}
				arrayIH->readEnergy += sumArrayReadEnergy;

				numBatchReadSynapse = (int)ceil((double)param->nHide/param->numColMuxed);
				// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
				for (int j=0; j<param->nHide; j+=numBatchReadSynapse) {
					int numActiveRows = 0;  // Number of selected rows for NeuroSim
					for (int n=0; n<param->numBitInput; n++) {
						for (int k=0; k<param->nInput; k++) {
							if ((dInput[i][k]>>n) & 1) {    // if the nth bit of dInput[i][k] is 1
								numActiveRows++;
							}
						}
					}
					subArrayIH->activityRowRead = (double)numActiveRows/param->nInput/param->numBitInput;
					subArrayIH->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayIH);
					subArrayIH->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
					subArrayIH->readLatency += NeuroSimSubArrayReadLatency(subArrayIH);
					subArrayIH->readLatency += NeuroSimNeuronReadLatency(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH);
				}
			} else {    // Algorithm
				#pragma omp parallel for
				for (int j = 0; j < param->nHide; j++) {
					for (int k = 0; k < param->nInput; k++) {
						outN1[j] += 2 * Input[i][k] * weight1[j][k] - Input[i][k];
					}
					a1[j] = sigmoid(outN1[j]);
				}
			}

			/* Second layer (hidder layer to the output layer) */
			std::fill_n(outN2, param->nOutput, 0);
			std::fill_n(a2, param->nOutput, 0);
			if (param->useHardwareInTrainingFF) {   // Hardware
				double sumArrayReadEnergy = 0;  // Use a temporary variable here since OpenMP does not support reduction on class member
				double readVoltage = static_cast<eNVM*>(arrayHO->cell[0][0])->readVoltage;
				double readPulseWidth = static_cast<eNVM*>(arrayHO->cell[0][0])->readPulseWidth;
				#pragma omp parallel for reduction(+: sumArrayReadEnergy)
				for (int j=0; j<param->nOutput; j++) {
					if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
						if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
							sumArrayReadEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd * param->nHide; // All WLs open
						}
					} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
						// XXX: To be released
					}
					for (int n=0; n<param->numBitInput; n++) {
						double pSumMaxAlgorithm = pow(2, n) / (param->numInputLevel - 1) * arrayHO->arrayRowSize;    // Max algorithm partial weighted sum for the nth vector bit (if both max input value and max weight are 1)
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog NVM
							double Isum = 0;    // weighted sum current
							double IsumMax = 0; // Max weighted sum current
							double a1Sum = 0;   // Weighted sum current of a1 vector * weight=1 column
							for (int k=0; k<param->nHide; k++) {
								if ((da1[k]>>n) & 1) {    // if the nth bit of da1[k] is 1
									Isum += arrayHO->ReadCell(j,k);
									a1Sum += arrayHO->GetMaxCellReadCurrent(j,k);
									sumArrayReadEnergy += arrayHO->wireCapRow * readVoltage * readVoltage; // Selected BLs (1T1R) or Selected WLs (cross-point)
								}
								IsumMax += arrayHO->GetMaxCellReadCurrent(j,k);
							}
							sumArrayReadEnergy += Isum * readVoltage * readPulseWidth;
							int outputDigits = 2 * CurrentToDigits(Isum, IsumMax) - CurrentToDigits(a1Sum, IsumMax);
							outN2[j] += DigitsToAlgorithm(outputDigits, pSumMaxAlgorithm);
						} else {    // SRAM or digital eNVM
							// XXX: To be released
						}
					}
					a2[j] = sigmoid(outN2[j]);
				}
				arrayHO->readEnergy += sumArrayReadEnergy;

				numBatchReadSynapse = (int)ceil((double)param->nOutput/param->numColMuxed);
				// Don't parallelize this loop since there may be update of member variables inside NeuroSim functions
				for (int j=0; j<param->nOutput; j+=numBatchReadSynapse) {
					int numActiveRows = 0;  // Number of selected rows for NeuroSim
					for (int n=0; n<param->numBitInput; n++) {
						for (int k=0; k<param->nHide; k++) {
							if ((da1[k]>>n) & 1) {    // if the nth bit of da1[k] is 1
								numActiveRows++;
							}
						}
					}
					subArrayHO->activityRowRead = (double)numActiveRows/param->nHide/param->numBitInput;
					subArrayHO->readDynamicEnergy += NeuroSimSubArrayReadEnergy(subArrayHO);
					subArrayHO->readDynamicEnergy += NeuroSimNeuronReadEnergy(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO);
					subArrayHO->readLatency += NeuroSimSubArrayReadLatency(subArrayHO);
					subArrayHO->readLatency += NeuroSimNeuronReadLatency(subArrayHO, adderHO, muxHO, muxDecoderHO, dffHO);
				}
			} else {
				#pragma omp parallel for
				for (int j = 0; j < param->nOutput; j++) {
					for (int k = 0; k < param->nHide; k++) {
						outN2[j] += 2 * a1[k] * weight2[j][k] - a1[k];
					}
					a2[j] = sigmoid(outN2[j]);
				}
			}

			// Backpropagation
			/* Second layer (hidder layer to the output layer) */
			for (int j = 0; j < param->nOutput; j++){
				s2[j] = -2 * a2[j] * (1 - a2[j])*(Output[i][j] - a2[j]);
			}

			/* First layer (input layer to the hidden layer) */
			std::fill_n(s1, param->nHide, 0);
			#pragma omp parallel for
			for (int j = 0; j < param->nHide; j++) {
				for (int k = 0; k < param->nOutput; k++) {
					s1[j] += a1[j] * (1 - a1[j]) * (2 * weight2[k][j] - 1) * s2[k];
				}
			}
			
			// Weights update
			/* Update weight of the first layer (input layer to the hidden layer) */
			if (param->useHardwareInTrainingWU) {
				double sumArrayWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumNeuroSimWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double writeVoltageLTP = static_cast<eNVM*>(arrayIH->cell[0][0])->writeVoltageLTP;
				double writeVoltageLTD = static_cast<eNVM*>(arrayIH->cell[0][0])->writeVoltageLTD;
				double writePulseWidthLTP = static_cast<eNVM*>(arrayIH->cell[0][0])->writePulseWidthLTP;
				double writePulseWidthLTD = static_cast<eNVM*>(arrayIH->cell[0][0])->writePulseWidthLTD;
				int maxNumLevelLTP = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxNumLevelLTP;
				int maxNumLevelLTD = static_cast<AnalogNVM*>(arrayIH->cell[0][0])->maxNumLevelLTD;
				numBatchWriteSynapse = (int)ceil((double)arrayIH->arrayColSize / param->numWriteColMuxed);
				#pragma omp parallel for reduction(+: sumArrayWriteEnergy, sumNeuroSimWriteEnergy)
				for (int k = 0; k < param->nInput; k++) {
					for (int j = 0; j < param->nHide; j+=numBatchWriteSynapse) {
						/* Batch write */
						int start = j;
						int end = j + numBatchWriteSynapse - 1;
						if (end >= param->nHide) {
							end = param->nHide - 1;
						}
						for (int jj = start; jj <= end; jj++) { // Selected cells
							deltaWeight1[jj][k] = -param->alpha1 * s1[jj] * Input[i][k];
							arrayIH->WriteCell(jj, k, deltaWeight1[jj][k], param->maxWeight, param->minWeight, true, param->writeEnergyReport);
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
								sumArrayWriteEnergy += static_cast<AnalogNVM*>(arrayIH->cell[jj][k])->writeEnergy;
							} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
								// XXX: To be released
							} else {    // SRAM
								// XXX: To be released
							}
							weight1[jj][k] = arrayIH->ConductanceToWeight(jj, k, param->maxWeight, param->minWeight);
						}
						/* Energy consumption on array caps for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog NVM
							if (param->writeEnergyReport) {
								if (static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected SLs is included in WriteCell()
									sumArrayWriteEnergy += arrayIH->wireGateCapRow * techIH.vdd * techIH.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected BL (LTP phases)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTP * writeVoltageLTP * (param->nHide-numBatchWriteSynapse);   // Unselected SLs (LTP phase)
									// No LTD part because all unselected rows and columns are V=0
								} else {
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP * writeVoltageLTP;    // Selected WL (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTP/2 * writeVoltageLTP/2 * (param->nInput - 1);  // Unselected WLs (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTP/2 * writeVoltageLTP/2 * (param->nHide - 1);   // Unselected BLs (LTP phase)
									sumArrayWriteEnergy += arrayIH->wireCapRow * writeVoltageLTD/2 * writeVoltageLTD/2 * (param->nInput - 1);    // Unselected WLs (LTD phase)
									sumArrayWriteEnergy += arrayIH->wireCapCol * writeVoltageLTD/2 * writeVoltageLTD/2 * (param->nHide - 1); // Unselected BLs (LTD phase)
								}
							}
						} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
							// XXX: To be released
						}
						/* Half-selected cells for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							if (!static_cast<eNVM*>(arrayIH->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
								for (int jj = 0; jj < param->nHide; jj++) { // Half-selected cells in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected cells
									sumArrayWriteEnergy += (writeVoltageLTP/2 * writeVoltageLTP/2 * static_cast<eNVM*>(arrayIH->cell[jj][k])->conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP + writeVoltageLTD/2 * writeVoltageLTD/2 * static_cast<eNVM*>(arrayIH->cell[jj][k])->conductanceAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD);
								}
								for (int kk = 0; kk < param->nInput; kk++) {    // Half-selected cells in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; } // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										sumArrayWriteEnergy += (writeVoltageLTP/2 * writeVoltageLTP/2 * static_cast<eNVM*>(arrayIH->cell[jj][kk])->conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP + writeVoltageLTD/2 * writeVoltageLTD/2 * static_cast<eNVM*>(arrayIH->cell[jj][kk])->conductanceAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD);
									}
								}
							}
						} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayIH->cell[0][0])) { // Digital eNVM
							// XXX: To be released
						}
					}
					/* Calculate the average number of write pulses on the selected row */
					#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
					{
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[0][0])) {  // Analog eNVM
							int sumNumWritePulse = 0;
							for (int j = 0; j < param->nHide; j++) {
								sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayIH->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							}
							subArrayIH->numWritePulse = sumNumWritePulse / param->nHide;
						}
						sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayIH);
					}
				}
				arrayIH->writeEnergy += sumArrayWriteEnergy;
				subArrayIH->writeDynamicEnergy += sumNeuroSimWriteEnergy;
				subArrayIH->writeLatency += NeuroSimSubArrayWriteLatency(subArrayIH);
			} else {
				#pragma omp parallel for
				for (int j = 0; j < param->nHide; j++) {
					for (int k = 0; k < param->nInput; k++) {
						deltaWeight1[j][k] = - param->alpha1 * s1[j] * Input[i][k];
						weight1[j][k] = weight1[j][k] + deltaWeight1[j][k];
						if (weight1[j][k] > param->maxWeight) {
							deltaWeight1[j][k] -= weight1[j][k] - param->maxWeight;
							weight1[j][k] = param->maxWeight;
						} else if (weight1[j][k] < param->minWeight) {
							deltaWeight1[j][k] += param->minWeight - weight1[j][k];
							weight1[j][k] = param->minWeight;
						}
						if (param->useHardwareInTrainingFF) {
							arrayIH->WriteCell(j, k, deltaWeight1[j][k], param->maxWeight, param->minWeight, false, param->writeEnergyReport);
						}
					}
				}
			}
			
			/* Update weight of the second layer (hidden layer to the output layer) */
			if (param->useHardwareInTrainingWU) {
				double sumArrayWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double sumNeuroSimWriteEnergy = 0;   // Use a temporary variable here since OpenMP does not support reduction on class member
				double writeVoltageLTP = static_cast<eNVM*>(arrayHO->cell[0][0])->writeVoltageLTP;
				double writeVoltageLTD = static_cast<eNVM*>(arrayHO->cell[0][0])->writeVoltageLTD;
				double writePulseWidthLTP = static_cast<eNVM*>(arrayHO->cell[0][0])->writePulseWidthLTP;
				double writePulseWidthLTD = static_cast<eNVM*>(arrayHO->cell[0][0])->writePulseWidthLTD;
				int maxNumLevelLTP = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxNumLevelLTP;
				int maxNumLevelLTD = static_cast<AnalogNVM*>(arrayHO->cell[0][0])->maxNumLevelLTD;
				numBatchWriteSynapse = (int)ceil((double)arrayHO->arrayColSize / param->numWriteColMuxed);
				#pragma omp parallel for reduction(+: sumArrayWriteEnergy, sumNeuroSimWriteEnergy)
				for (int k = 0; k < param->nHide; k++) {
					for (int j = 0; j < param->nOutput; j+=numBatchWriteSynapse) {
						/* Batch write */
						int start = j;
						int end = j + numBatchWriteSynapse - 1;
						if (end >= param->nOutput) {
							end = param->nOutput - 1;
						}
						for (int jj = start; jj <= end; jj++) { // Selected cells
							deltaWeight2[jj][k] = -param->alpha2 * s2[jj] * a1[k];
							arrayHO->WriteCell(jj, k, deltaWeight2[jj][k], param->maxWeight, param->minWeight, true, param->writeEnergyReport);
							if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
								sumArrayWriteEnergy += static_cast<eNVM*>(arrayHO->cell[jj][k])->writeEnergy;
							} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
								// XXX: To be released
							} else {    // SRAM
								// XXX: To be released
							}
							weight2[jj][k] = arrayHO->ConductanceToWeight(jj, k, param->maxWeight, param->minWeight);
						}
						/* Energy consumption on array caps for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							if (param->writeEnergyReport) {
								if (static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess) {  // 1T1R
									// The energy on selected SLs is included in WriteCell()
									sumArrayWriteEnergy += arrayHO->wireGateCapRow * techHO.vdd * techHO.vdd * 2;   // Selected WL (*2 means both LTP and LTD phases)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected BL (LTP phases)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTP * writeVoltageLTP * (param->nOutput-numBatchWriteSynapse);   // Unselected SLs (LTP phase)
									// No LTD part because all unselected rows and columns are V=0
								} else {
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP * writeVoltageLTP;   // Selected WL (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTP/2 * writeVoltageLTP/2 * (param->nHide - 1);    // Unselected WLs (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTP/2 * writeVoltageLTP/2 * (param->nOutput - 1); // Unselected BLs (LTP phase)
									sumArrayWriteEnergy += arrayHO->wireCapRow * writeVoltageLTD/2 * writeVoltageLTD/2 * (param->nHide - 1);    // Unselected WLs (LTD phase)
									sumArrayWriteEnergy += arrayHO->wireCapCol * writeVoltageLTD/2 * writeVoltageLTD/2 * (param->nOutput - 1); // Unselected BLs (LTD phase)
								}
							}
						} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
							// XXX: To be released
						}
						/* Half-selected cells for eNVM */
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							if (!static_cast<eNVM*>(arrayHO->cell[0][0])->cmosAccess && param->writeEnergyReport) { // Cross-point
								for (int jj = 0; jj < param->nOutput; jj++) {    // Half-selected cells in the same row
									if (jj >= start && jj <= end) { continue; } // Skip the selected cells
									sumArrayWriteEnergy += (writeVoltageLTP/2 * writeVoltageLTP/2 * static_cast<eNVM*>(arrayHO->cell[jj][k])->conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP + writeVoltageLTD/2 * writeVoltageLTD/2 * static_cast<eNVM*>(arrayHO->cell[jj][k])->conductanceAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD);
								}
								for (int kk = 0; kk < param->nHide; kk++) { // Half-selected cells in other rows
									// Note that here is a bit inaccurate if using OpenMP, because the weight on other rows (threads) are also being updated
									if (kk == k) { continue; }  // Skip the selected row
									for (int jj = start; jj <= end; jj++) {
										sumArrayWriteEnergy += (writeVoltageLTP/2 * writeVoltageLTP/2 * static_cast<eNVM*>(arrayHO->cell[jj][kk])->conductanceAtHalfVwLTP * writePulseWidthLTP * maxNumLevelLTP + writeVoltageLTD/2 * writeVoltageLTD/2 * static_cast<eNVM*>(arrayHO->cell[jj][kk])->conductanceAtHalfVwLTD * writePulseWidthLTD * maxNumLevelLTD);
									}
								}
							}
						} else if (DigitalNVM *temp = dynamic_cast<DigitalNVM*>(arrayHO->cell[0][0])) { // Digital eNVM
							// XXX: To be released
						}
					}
					/* Calculate the average number of write pulses on the selected row */
					#pragma omp critical    // Use critical here since NeuroSim class functions may update its member variables
					{
						if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayHO->cell[0][0])) {  // Analog eNVM
							int sumNumWritePulse = 0;
							for (int j = 0; j < param->nOutput; j++) {
								sumNumWritePulse += abs(static_cast<AnalogNVM*>(arrayHO->cell[j][k])->numPulse);    // Note that LTD has negative pulse number
							}
							subArrayHO->numWritePulse = sumNumWritePulse / param->nOutput;
						}
						sumNeuroSimWriteEnergy += NeuroSimSubArrayWriteEnergy(subArrayHO);
					}
				}
				arrayHO->writeEnergy += sumArrayWriteEnergy;
				subArrayHO->writeDynamicEnergy += sumNeuroSimWriteEnergy;
				subArrayHO->writeLatency += NeuroSimSubArrayWriteLatency(subArrayHO);
			} else {
				#pragma omp parallel for
				for (int j = 0; j < param->nOutput; j++) {
					for (int k = 0; k < param->nHide; k++) {
						deltaWeight2[j][k] = -param->alpha2 * s2[j] * a1[k];
						weight2[j][k] = weight2[j][k] + deltaWeight2[j][k];
						if (weight2[j][k] > param->maxWeight) {
							deltaWeight2[j][k] -= weight2[j][k] - param->maxWeight;
							weight2[j][k] = param->maxWeight;
						} else if (weight2[j][k] < param->minWeight) {
							deltaWeight2[j][k] += param->minWeight - weight2[j][k];
							weight2[j][k] = param->minWeight;
						}
						if (param->useHardwareInTrainingFF) {
							arrayHO->WriteCell(j, k, deltaWeight2[j][k], param->maxWeight, param->minWeight, false, param->writeEnergyReport);
						}
					}
				}
			}
		}
	}
}

