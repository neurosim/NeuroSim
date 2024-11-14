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


#ifndef CONSTANT_H_
#define CONSTANT_H_



#define INV     0
#define NOR     1
#define NAND    2

#define NMOS    0
#define PMOS    1

#define MAX_NMOS_SIZE   100.0
#define MIN_NMOS_SIZE   2.0   

#define MAX_TRANSISTOR_HEIGHT_FINFET  34.0
#define MAX_TRANSISTOR_HEIGHT        28.0 

// 1.4 update : Standard cell height trend - updated

#define MAX_TRANSISTOR_HEIGHT_14nm 41.142 // Samsung
#define MAX_TRANSISTOR_HEIGHT_10nm 33.0 // TSMC
#define MAX_TRANSISTOR_HEIGHT_7nm 34.285 // TSMC
#define MAX_TRANSISTOR_HEIGHT_5nm 36.0 // IRDS 2021
#define MAX_TRANSISTOR_HEIGHT_3nm  48.0 // IRDS 2022
#define MAX_TRANSISTOR_HEIGHT_2nm  57.0 // IRDS 2022
#define MAX_TRANSISTOR_HEIGHT_1nm  80.0 // IRDS 2022


#define MIN_GAP_BET_SAME_TYPE_DIFFS    1.6  // not used 


#define MIN_GAP_BET_P_AND_N_DIFFS 3.5 
#define MIN_POLY_EXT_DIFF       1.0 // Minimum poly extension beyond diffusion region
#define MIN_GAP_BET_FIELD_POLY 1.6 // Field poly means the poly above the field oxide (outside the active region)


// 1.4 update : PN separation, region outside fin (half of the inter-spacing between outermost fins in adjacent standard cells) - updated
// OUTER_HEIGHT_REGION = Cell height - MIN_GAP_BET_P_AND_N_DIFFS (PN gap) - Fin region
#define MIN_GAP_BET_P_AND_N_DIFFS_14nm  9.71 // Fin pitch*3 - Fin width
#define OUTER_HEIGHT_REGION_14nm  10.285
#define MIN_GAP_BET_P_AND_N_DIFFS_10nm 10.0 // Fin pitch*3 - Fin width
#define OUTER_HEIGHT_REGION_10nm 8.2
#define MIN_GAP_BET_P_AND_N_DIFFS_7nm 11.85 // Fin pitch*3 - Fin width
#define OUTER_HEIGHT_REGION_7nm 12.28
#define MIN_GAP_BET_P_AND_N_DIFFS_5nm 12.8 // IRDS 2021
#define OUTER_HEIGHT_REGION_5nm 19.4
#define MIN_GAP_BET_P_AND_N_DIFFS_3nm 15.0 // IRDS 2022
#define OUTER_HEIGHT_REGION_3nm 12.666
#define MIN_GAP_BET_P_AND_N_DIFFS_2nm 20.0 // IRDS 2022
#define OUTER_HEIGHT_REGION_2nm 22.0 
#define MIN_GAP_BET_P_AND_N_DIFFS_1nm 15.0  // IRDS 2022
#define OUTER_HEIGHT_REGION_1nm 40.0 

#define POLY_WIDTH                     1.0
#define POLY_WIDTH_FINFET              1.4
#define MIN_GAP_BET_GATE_POLY          2.8  //1.5
#define MIN_GAP_BET_GATE_POLY_FINFET   3.9


// 1.4 update : CPP trend

#define CPP_14nm 5.571 // Samsung
#define POLY_WIDTH_14nm  1.857 // IRDS 2016
#define CPP_10nm  6.4 // TSMC
#define POLY_WIDTH_10nm  2.2 // IRDS 2017
#define CPP_7nm 8.1428 // TSMC
#define POLY_WIDTH_7nm 3.14 // IRDS 2018
#define CPP_5nm 10.2 // IRDS 2021
#define POLY_WIDTH_5nm 4.0 // IRDS 2021
#define CPP_3nm 16.0 // IRDS 2022
#define POLY_WIDTH_3nm 6.0 // IRDS 2022
#define CPP_2nm 22.5 // IRDS 2022
#define POLY_WIDTH_2nm 7.0 // IRDS 2022
#define CPP_1nm 40.0 // IRDS 2022
#define POLY_WIDTH_1nm 12.0 // IRDS 2022

#define MIN_GAP_BET_CONTACT_POLY       0.7  
#define CONTACT_SIZE                   1.3  
#define MIN_WIDTH_POWER_RAIL           3.4  

// 1.4 update : Metal Pitch trend -> needed for row-decoder layout
// Assume single patterning for 14 nm and beyond and assume Minimum Metal Pitch=M1 pitch=M2 pitch=M3 pitch

#define M2_PITCH                       3.2  // along width
#define M2_PITCH_14nm                  4.57
#define M2_PITCH_10nm                  4.4
#define M2_PITCH_7nm                   5.71
#define M2_PITCH_5nm                   6.0
#define M2_PITCH_3nm                   8.0
#define M2_PITCH_2nm                   10.0
#define M2_PITCH_1nm                   16.0

#define M3_PITCH                       2.8  // along height
#define M3_PITCH_14nm                  4.57
#define M3_PITCH_10nm                  4.4
#define M3_PITCH_7nm                   5.71
#define M3_PITCH_5nm                   6.0
#define M3_PITCH_3nm                   8.0
#define M3_PITCH_2nm                   10.0
#define M3_PITCH_1nm                   16.0

#define AVG_RATIO_LEAK_2INPUT_NAND 0.48
#define AVG_RATIO_LEAK_3INPUT_NAND 0.31
#define AVG_RATIO_LEAK_2INPUT_NOR  0.95
#define AVG_RATIO_LEAK_3INPUT_NOR  0.62

// 1.4 update: updated comment - needed for senseamp

#define W_SENSE_P       7.5
#define W_SENSE_N       3.75
#define W_SENSE_ISO     12.5
#define W_SENSE_EN      5.0
#define W_SENSE_MUX     9.0

#define IR_DROP_TOLERANCE           0.25
#define LINEAR_REGION_RATIO         0.20

#define HEIGHT_WIDTH_RATIO_LIMIT    5.0

#define RATIO_READ_THRESHOLD_VS_VOLTAGE 0.2

#define ROW_MODE    0   // Connect to rows
#define COL_MODE    1   // Connect to columns




#endif /* CONSTANT_H_ */
