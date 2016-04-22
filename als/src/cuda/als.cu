/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * als.cu
 *
 *  Created on: Feb 10, 2015
 *      Author: Wei Tan (wtan@us.ibm.com)
 *  Alternating Least Square for Matrix Factorization on CUDA 7.0+
 *  Code optimized for F = 10, 20, 30,...190, 200 ... (F has to be a multiply of 10).
 *  Tested on cc 3.5, 3.7 and 5.2 platforms.
 */
#include "als.h"
//transform a float array to a double array
__global__ void floatArray2doubleArray(const float * floatArray, double* doubleArray,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		doubleArray[i] = (double) floatArray[i];
	}
}

//transform a double array to a float array
__global__ void doubleArray2floatArray(const double * doubleArray, float* floatArray,
		const int size) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < size) {
		floatArray[i] = (float) doubleArray[i];
	}
}

/* Form the hermitian matrices tt from R (csrRowIndex, csrColIndex)
 * For each row u in R, tt[u]= \sum\nolimits_{r_{uv} \neq 0}(theta_v dot theta_v^T)
 * Each block solves a row; each thread takes a 10*10 block from the F*F matrix
 * More detail can be found from http://learningsys.org/papers/LearningSys_2015_paper_3.pdf
 */
__global__ void
get_hermitianT10(const int batch_offset, double* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* __restrict__ thetaT) {
	extern __shared__ float2 thetaTemp [];

	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		//IMPORTANT: use register to accumulate; should NOT use array that is to be in gmem
		//exlpoit register to accumulate; achieve good ILP
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;

		int N = F/10; // N = 100/10=10; for F = 100 and T = 10
		int effective_block_size = N*(N+1)/2;
		//get the x and y coordinate
		int tile_x = 0;
		int tile_y = 0;

		for ( int i = 0; i < N; i++ ) {
			int end = ((2*N-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * 10;
				tile_y = (N + threadIdx.x - end) * 10;
				break;
			}
		}
		float2 theta;
		int index = blockIdx.x*F*F;
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			//REQ: blockDim.x >= F/2
			if(threadIdx.x < F/2){
				for(int k = 0; k< SCAN_BATCH; k++){
					if(iter*SCAN_BATCH + k < end - start){
						theta.x = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x]);
						theta.y = __ldg(&thetaT[F * csrColIndex[start + iter*SCAN_BATCH + k] + 2*threadIdx.x+1]);
						thetaTemp[k * F/2 + threadIdx.x] = theta;
					}
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[k*F/2 + threadIdx.x], 0, 2*sizeof(float));
				}
			}
			__syncthreads();

			if(threadIdx.x < effective_block_size){//this redundant "if" seems improving kernel performance
				for(int k = 0; k < SCAN_BATCH; k++){
					temp0 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp1 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp2 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp3 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp4 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp5 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp6 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp7 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp8 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp9 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp10 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp11 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp12 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp13 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp14 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp15 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp16 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp17 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp18 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp19 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp20 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp21 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp22 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp23 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp24 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp25 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp26 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp27 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp28 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp29 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp30 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp31 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp32 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp33 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp34 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp35 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp36 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp37 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp38 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp39 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp40 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp41 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp42 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp43 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp44 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp45 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp46 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp47 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp48 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp49 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp50 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp51 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp52 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp53 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp54 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp55 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp56 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp57 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp58 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp59 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp60 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp61 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp62 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp63 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp64 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp65 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp66 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp67 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp68 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp69 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp70 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp71 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp72 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp73 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp74 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp75 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp76 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp77 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp78 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp79 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;


					temp80 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp81 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp82 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp83 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp84 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp85 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp86 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp87 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp88 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp89 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp90 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp91 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp92 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp93 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp94 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp95 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp96 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp97 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp98 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp99 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();

		//copy output to gmem
		if(threadIdx.x < effective_block_size){
			tt[index + tile_x + tile_y*F] = temp0;
			tt[index + tile_x + (tile_y + 1)*F] = temp1;
			tt[index + tile_x + (tile_y + 2)*F] = temp2;
			tt[index + tile_x + (tile_y + 3)*F] = temp3;
			tt[index + tile_x + (tile_y + 4)*F] = temp4;
			tt[index + tile_x + (tile_y + 5)*F] = temp5;
			tt[index + tile_x + (tile_y + 6)*F] = temp6;
			tt[index + tile_x + (tile_y + 7)*F] = temp7;
			tt[index + tile_x + (tile_y + 8)*F] = temp8;
			tt[index + tile_x + (tile_y + 9)*F] = temp9;

			tt[index + tile_x + 1 + tile_y*F] = temp10;
			tt[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
			tt[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
			tt[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
			tt[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
			tt[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
			tt[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
			tt[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
			tt[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
			tt[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

			tt[index + tile_x + 2 + tile_y*F] = temp20;
			tt[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
			tt[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
			tt[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
			tt[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
			tt[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
			tt[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
			tt[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
			tt[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
			tt[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

			tt[index + tile_x + 3 + tile_y*F] = temp30;
			tt[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
			tt[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
			tt[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
			tt[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
			tt[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
			tt[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
			tt[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
			tt[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
			tt[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

			tt[index + tile_x + 4 + tile_y*F] = temp0;
			tt[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
			tt[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
			tt[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
			tt[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
			tt[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
			tt[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
			tt[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
			tt[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
			tt[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

			tt[index + tile_x + 5 + tile_y*F] = temp50;
			tt[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
			tt[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
			tt[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
			tt[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
			tt[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
			tt[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
			tt[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
			tt[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
			tt[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

			tt[index + tile_x + 6 + tile_y*F] = temp60;
			tt[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
			tt[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
			tt[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
			tt[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
			tt[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
			tt[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
			tt[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
			tt[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
			tt[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

			tt[index + tile_x + 7 + tile_y*F] = temp70;
			tt[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
			tt[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
			tt[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
			tt[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
			tt[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
			tt[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
			tt[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
			tt[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
			tt[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

			tt[index + tile_x + 8 + tile_y*F] = temp80;
			tt[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
			tt[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
			tt[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
			tt[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
			tt[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
			tt[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
			tt[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
			tt[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
			tt[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

			tt[index + tile_x + 9 + tile_y*F] = temp90;
			tt[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
			tt[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
			tt[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
			tt[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
			tt[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
			tt[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
			tt[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
			tt[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
			tt[index + tile_x + 9 + (tile_y + 9)*F] = temp99;

			//the hermitian matrix is symmetric
			if(tile_x!=tile_y){
				tt[index + tile_y + 0+ (tile_x + 0)*F]= temp0;
				tt[index + tile_y + 1+ (tile_x + 0)*F]= temp1;
				tt[index + tile_y + 2+ (tile_x + 0)*F]= temp2;
				tt[index + tile_y + 3+ (tile_x + 0)*F]= temp3;
				tt[index + tile_y + 4+ (tile_x + 0)*F]= temp4;
				tt[index + tile_y + 5+ (tile_x + 0)*F]= temp5;
				tt[index + tile_y + 6+ (tile_x + 0)*F]= temp6;
				tt[index + tile_y + 7+ (tile_x + 0)*F]= temp7;
				tt[index + tile_y + 8+ (tile_x + 0)*F]= temp8;
				tt[index + tile_y + 9+ (tile_x + 0)*F]= temp9;


				tt[index + tile_y + 0+ (tile_x + 1)*F]= temp10;
				tt[index + tile_y + 1+ (tile_x + 1)*F]= temp11;
				tt[index + tile_y + 2+ (tile_x + 1)*F]= temp12;
				tt[index + tile_y + 3+ (tile_x + 1)*F]= temp13;
				tt[index + tile_y + 4+ (tile_x + 1)*F]= temp14;
				tt[index + tile_y + 5+ (tile_x + 1)*F]= temp15;
				tt[index + tile_y + 6+ (tile_x + 1)*F]= temp16;
				tt[index + tile_y + 7+ (tile_x + 1)*F]= temp17;
				tt[index + tile_y + 8+ (tile_x + 1)*F]= temp18;
				tt[index + tile_y + 9+ (tile_x + 1)*F]= temp19;


				tt[index + tile_y + 0+ (tile_x + 2)*F]= temp20;
				tt[index + tile_y + 1+ (tile_x + 2)*F]= temp21;
				tt[index + tile_y + 2+ (tile_x + 2)*F]= temp22;
				tt[index + tile_y + 3+ (tile_x + 2)*F]= temp23;
				tt[index + tile_y + 4+ (tile_x + 2)*F]= temp24;
				tt[index + tile_y + 5+ (tile_x + 2)*F]= temp25;
				tt[index + tile_y + 6+ (tile_x + 2)*F]= temp26;
				tt[index + tile_y + 7+ (tile_x + 2)*F]= temp27;
				tt[index + tile_y + 8+ (tile_x + 2)*F]= temp28;
				tt[index + tile_y + 9+ (tile_x + 2)*F]= temp29;


				tt[index + tile_y + 0+ (tile_x + 3)*F]= temp30;
				tt[index + tile_y + 1+ (tile_x + 3)*F]= temp31;
				tt[index + tile_y + 2+ (tile_x + 3)*F]= temp32;
				tt[index + tile_y + 3+ (tile_x + 3)*F]= temp33;
				tt[index + tile_y + 4+ (tile_x + 3)*F]= temp34;
				tt[index + tile_y + 5+ (tile_x + 3)*F]= temp35;
				tt[index + tile_y + 6+ (tile_x + 3)*F]= temp36;
				tt[index + tile_y + 7+ (tile_x + 3)*F]= temp37;
				tt[index + tile_y + 8+ (tile_x + 3)*F]= temp38;
				tt[index + tile_y + 9+ (tile_x + 3)*F]= temp39;


				tt[index + tile_y + 0+ (tile_x + 4)*F]= temp40;
				tt[index + tile_y + 1+ (tile_x + 4)*F]= temp41;
				tt[index + tile_y + 2+ (tile_x + 4)*F]= temp42;
				tt[index + tile_y + 3+ (tile_x + 4)*F]= temp43;
				tt[index + tile_y + 4+ (tile_x + 4)*F]= temp44;
				tt[index + tile_y + 5+ (tile_x + 4)*F]= temp45;
				tt[index + tile_y + 6+ (tile_x + 4)*F]= temp46;
				tt[index + tile_y + 7+ (tile_x + 4)*F]= temp47;
				tt[index + tile_y + 8+ (tile_x + 4)*F]= temp48;
				tt[index + tile_y + 9+ (tile_x + 4)*F]= temp49;


				tt[index + tile_y + 0+ (tile_x + 5)*F]= temp50;
				tt[index + tile_y + 1+ (tile_x + 5)*F]= temp51;
				tt[index + tile_y + 2+ (tile_x + 5)*F]= temp52;
				tt[index + tile_y + 3+ (tile_x + 5)*F]= temp53;
				tt[index + tile_y + 4+ (tile_x + 5)*F]= temp54;
				tt[index + tile_y + 5+ (tile_x + 5)*F]= temp55;
				tt[index + tile_y + 6+ (tile_x + 5)*F]= temp56;
				tt[index + tile_y + 7+ (tile_x + 5)*F]= temp57;
				tt[index + tile_y + 8+ (tile_x + 5)*F]= temp58;
				tt[index + tile_y + 9+ (tile_x + 5)*F]= temp59;


				tt[index + tile_y + 0+ (tile_x + 6)*F]= temp60;
				tt[index + tile_y + 1+ (tile_x + 6)*F]= temp61;
				tt[index + tile_y + 2+ (tile_x + 6)*F]= temp62;
				tt[index + tile_y + 3+ (tile_x + 6)*F]= temp63;
				tt[index + tile_y + 4+ (tile_x + 6)*F]= temp64;
				tt[index + tile_y + 5+ (tile_x + 6)*F]= temp65;
				tt[index + tile_y + 6+ (tile_x + 6)*F]= temp66;
				tt[index + tile_y + 7+ (tile_x + 6)*F]= temp67;
				tt[index + tile_y + 8+ (tile_x + 6)*F]= temp68;
				tt[index + tile_y + 9+ (tile_x + 6)*F]= temp69;


				tt[index + tile_y + 0+ (tile_x + 7)*F]= temp70;
				tt[index + tile_y + 1+ (tile_x + 7)*F]= temp71;
				tt[index + tile_y + 2+ (tile_x + 7)*F]= temp72;
				tt[index + tile_y + 3+ (tile_x + 7)*F]= temp73;
				tt[index + tile_y + 4+ (tile_x + 7)*F]= temp74;
				tt[index + tile_y + 5+ (tile_x + 7)*F]= temp75;
				tt[index + tile_y + 6+ (tile_x + 7)*F]= temp76;
				tt[index + tile_y + 7+ (tile_x + 7)*F]= temp77;
				tt[index + tile_y + 8+ (tile_x + 7)*F]= temp78;
				tt[index + tile_y + 9+ (tile_x + 7)*F]= temp79;


				tt[index + tile_y + 0+ (tile_x + 8)*F]= temp80;
				tt[index + tile_y + 1+ (tile_x + 8)*F]= temp81;
				tt[index + tile_y + 2+ (tile_x + 8)*F]= temp82;
				tt[index + tile_y + 3+ (tile_x + 8)*F]= temp83;
				tt[index + tile_y + 4+ (tile_x + 8)*F]= temp84;
				tt[index + tile_y + 5+ (tile_x + 8)*F]= temp85;
				tt[index + tile_y + 6+ (tile_x + 8)*F]= temp86;
				tt[index + tile_y + 7+ (tile_x + 8)*F]= temp87;
				tt[index + tile_y + 8+ (tile_x + 8)*F]= temp88;
				tt[index + tile_y + 9+ (tile_x + 8)*F]= temp89;


				tt[index + tile_y + 0+ (tile_x + 9)*F]= temp90;
				tt[index + tile_y + 1+ (tile_x + 9)*F]= temp91;
				tt[index + tile_y + 2+ (tile_x + 9)*F]= temp92;
				tt[index + tile_y + 3+ (tile_x + 9)*F]= temp93;
				tt[index + tile_y + 4+ (tile_x + 9)*F]= temp94;
				tt[index + tile_y + 5+ (tile_x + 9)*F]= temp95;
				tt[index + tile_y + 6+ (tile_x + 9)*F]= temp96;
				tt[index + tile_y + 7+ (tile_x + 9)*F]= temp97;
				tt[index + tile_y + 8+ (tile_x + 9)*F]= temp98;
				tt[index + tile_y + 9+ (tile_x + 9)*F]= temp99;
			}
			//add regularization
			if(tile_x == tile_y){
				for(int k = 0; k < 10; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}

	}
}

/* Form the hermitian matrices tt from R (csrRowIndex, csrColIndex)
 * For each row u in R, tt[u]= \sum\nolimits_{r_{uv} \neq 0}(theta_v dot theta_v^T)
 * Each block solves a row; each thread takes a 10*10 block from the F*F matrix
 * More detail can be found from http://learningsys.org/papers/LearningSys_2015_paper_3.pdf
 * This kernel is specifically optimized for F = 100
 */
__global__ void
//__launch_bounds__(64, 6)
get_hermitian100(const int batch_offset, double* tt,
		const int* csrRowIndex, const int* csrColIndex, const float lambda, const int m, const int F,
		const float* __restrict__ thetaT) {
	extern __shared__ float2 thetaTemp[];
	int row = blockIdx.x + batch_offset;
	if (row < m) {
		//this block needs to handle end - start thetaT columns
		int start = csrRowIndex[row];
		int end = csrRowIndex[row + 1];
		//slide through [start, end] by window size SCAN_BATCH
		int iterations = (end - start - 1)/SCAN_BATCH + 1;
		//IMPORTANT: use register to accumulate; should NOT use array that is to be in gmem
		float temp0= 0, temp1= 0, temp2= 0, temp3= 0, temp4= 0, temp5= 0, temp6= 0, temp7= 0, temp8= 0, temp9 = 0;
		float temp10= 0, temp11= 0, temp12= 0, temp13= 0, temp14= 0, temp15= 0, temp16= 0, temp17= 0, temp18= 0, temp19 = 0;
		float temp20= 0, temp21= 0, temp22= 0, temp23= 0, temp24= 0, temp25= 0, temp26= 0, temp27= 0, temp28= 0, temp29 = 0;
		float temp30= 0, temp31= 0, temp32= 0, temp33= 0, temp34= 0, temp35= 0, temp36= 0, temp37= 0, temp38= 0, temp39 = 0;
		float temp40= 0, temp41= 0, temp42= 0, temp43= 0, temp44= 0, temp45= 0, temp46= 0, temp47= 0, temp48= 0, temp49 = 0;
		float temp50= 0, temp51= 0, temp52= 0, temp53= 0, temp54= 0, temp55= 0, temp56= 0, temp57= 0, temp58= 0, temp59 = 0;
		float temp60= 0, temp61= 0, temp62= 0, temp63= 0, temp64= 0, temp65= 0, temp66= 0, temp67= 0, temp68= 0, temp69 = 0;
		float temp70= 0, temp71= 0, temp72= 0, temp73= 0, temp74= 0, temp75= 0, temp76= 0, temp77= 0, temp78= 0, temp79 = 0;
		float temp80= 0, temp81= 0, temp82= 0, temp83= 0, temp84= 0, temp85= 0, temp86= 0, temp87= 0, temp88= 0, temp89 = 0;
		float temp90= 0, temp91= 0, temp92= 0, temp93= 0, temp94= 0, temp95= 0, temp96= 0, temp97= 0, temp98= 0, temp99 = 0;


		//int tile_x = (threadIdx.x/tile) * tile;//start x of this tile
		//int tile_y = (threadIdx.x%tile) * tile;//start y of this tile
		int tile_x = 0;
		int tile_y = 0;

		int tile = F/10;
		for ( int i = 0; i < 10; i++){
			int end = ((20-i)*(i+1))/2;
			if(threadIdx.x < end){
				tile_x = i * tile;
				tile_y = (10 + threadIdx.x - end) * tile;
				break;
			}
		}
		//iteration: copy gmem-->smem; aggregate smem-->register
		for (int iter = 0; iter < iterations; iter ++){
			float2 theta;
			//copy texture --> smem, and sync
			//two layers: warp divergence unless we split at 32
			//32 > SCAN_BATCH
			if(threadIdx.x < 2*32 ){
				//int index = threadIdx.x;
				int index = threadIdx.x - (threadIdx.x/32)*32;	//0 to 31;
				if(index < SCAN_BATCH){
					if(iter*SCAN_BATCH + index < end - start){
						//IMPORTANT: for loop has constant and identical start and end
						if(threadIdx.x < 32){
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k+1]);
								thetaTemp[index * F/2 + k/2] = theta;
							}
						}
						else {
							for (int k = 0; k < 50; k += 2){
								theta.x = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 50]);
								theta.y = __ldg(&thetaT[ F * csrColIndex[start + iter*SCAN_BATCH + index] + k + 51]);
								thetaTemp[index * F/2 + k/2 + 25] = theta;
							}
						}
					}
					//must be the last iteration; no need to check
					//not enough theta to copy, set zero
					else
						memset(&thetaTemp[index*F/2], 0, F*sizeof(float));
				}
			}
			__syncthreads();

			//tile: 10*10
			if(threadIdx.x < 55 ){
				for(int k = 0; k < SCAN_BATCH; k++){
					temp0 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp1 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp2 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp3 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp4 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp5 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp6 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp7 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp8 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp9 += thetaTemp[tile_x/2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp10 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp11 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp12 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp13 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp14 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp15 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp16 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp17 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp18 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp19 += thetaTemp[tile_x/2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp20 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp21 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp22 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp23 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp24 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp25 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp26 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp27 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp28 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp29 += thetaTemp[tile_x/2 +1 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp30 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp31 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp32 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp33 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp34 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp35 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp36 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp37 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp38 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp39 += thetaTemp[tile_x/2 +1 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp40 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp41 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp42 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp43 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp44 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp45 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp46 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp47 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp48 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp49 += thetaTemp[tile_x/2 +2 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp50 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp51 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp52 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp53 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp54 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp55 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp56 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp57 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp58 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp59 += thetaTemp[tile_x/2 +2 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp60 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp61 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp62 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp63 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp64 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp65 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp66 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp67 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp68 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp69 += thetaTemp[tile_x/2 +3 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp70 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp71 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp72 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp73 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp74 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp75 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp76 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp77 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp78 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp79 += thetaTemp[tile_x/2 +3 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;


					temp80 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].x;
					temp81 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 + k*F/2].y;
					temp82 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp83 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp84 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp85 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp86 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp87 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp88 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp89 += thetaTemp[tile_x/2 +4 + k*F/2].x * thetaTemp[tile_y/2 +4 + k*F/2].y;

					temp90 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].x;
					temp91 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 + k*F/2].y;
					temp92 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].x;
					temp93 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +1 + k*F/2].y;
					temp94 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].x;
					temp95 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +2 + k*F/2].y;
					temp96 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].x;
					temp97 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +3 + k*F/2].y;
					temp98 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].x;
					temp99 += thetaTemp[tile_x/2 +4 + k*F/2].y * thetaTemp[tile_y/2 +4 + k*F/2].y;
				}
			}
		}
		//end of iteration in copying from smem and aggregating in register
		__syncthreads();
		if(threadIdx.x < 55 ){
			//copy output to gmem
			int index = blockIdx.x*F*F;
			tt[index + tile_x + tile_y*F] = temp0;
			tt[index + tile_x + (tile_y + 1)*F] = temp1;
			tt[index + tile_x + (tile_y + 2)*F] = temp2;
			tt[index + tile_x + (tile_y + 3)*F] = temp3;
			tt[index + tile_x + (tile_y + 4)*F] = temp4;
			tt[index + tile_x + (tile_y + 5)*F] = temp5;
			tt[index + tile_x + (tile_y + 6)*F] = temp6;
			tt[index + tile_x + (tile_y + 7)*F] = temp7;
			tt[index + tile_x + (tile_y + 8)*F] = temp8;
			tt[index + tile_x + (tile_y + 9)*F] = temp9;

			tt[index + tile_x + 1 + tile_y*F] = temp10;
			tt[index + tile_x + 1 + (tile_y + 1)*F] = temp11;
			tt[index + tile_x + 1 + (tile_y + 2)*F] = temp12;
			tt[index + tile_x + 1 + (tile_y + 3)*F] = temp13;
			tt[index + tile_x + 1 + (tile_y + 4)*F] = temp14;
			tt[index + tile_x + 1 + (tile_y + 5)*F] = temp15;
			tt[index + tile_x + 1 + (tile_y + 6)*F] = temp16;
			tt[index + tile_x + 1 + (tile_y + 7)*F] = temp17;
			tt[index + tile_x + 1 + (tile_y + 8)*F] = temp18;
			tt[index + tile_x + 1 + (tile_y + 9)*F] = temp19;

			tt[index + tile_x + 2 + tile_y*F] = temp20;
			tt[index + tile_x + 2 + (tile_y + 1)*F] = temp21;
			tt[index + tile_x + 2 + (tile_y + 2)*F] = temp22;
			tt[index + tile_x + 2 + (tile_y + 3)*F] = temp23;
			tt[index + tile_x + 2 + (tile_y + 4)*F] = temp24;
			tt[index + tile_x + 2 + (tile_y + 5)*F] = temp25;
			tt[index + tile_x + 2 + (tile_y + 6)*F] = temp26;
			tt[index + tile_x + 2 + (tile_y + 7)*F] = temp27;
			tt[index + tile_x + 2 + (tile_y + 8)*F] = temp28;
			tt[index + tile_x + 2 + (tile_y + 9)*F] = temp29;

			tt[index + tile_x + 3 + tile_y*F] = temp30;
			tt[index + tile_x + 3 + (tile_y + 1)*F] = temp31;
			tt[index + tile_x + 3 + (tile_y + 2)*F] = temp32;
			tt[index + tile_x + 3 + (tile_y + 3)*F] = temp33;
			tt[index + tile_x + 3 + (tile_y + 4)*F] = temp34;
			tt[index + tile_x + 3 + (tile_y + 5)*F] = temp35;
			tt[index + tile_x + 3 + (tile_y + 6)*F] = temp36;
			tt[index + tile_x + 3 + (tile_y + 7)*F] = temp37;
			tt[index + tile_x + 3 + (tile_y + 8)*F] = temp38;
			tt[index + tile_x + 3 + (tile_y + 9)*F] = temp39;

			tt[index + tile_x + 4 + tile_y*F] = temp0;
			tt[index + tile_x + 4 + (tile_y + 1)*F] = temp41;
			tt[index + tile_x + 4 + (tile_y + 2)*F] = temp42;
			tt[index + tile_x + 4 + (tile_y + 3)*F] = temp43;
			tt[index + tile_x + 4 + (tile_y + 4)*F] = temp44;
			tt[index + tile_x + 4 + (tile_y + 5)*F] = temp45;
			tt[index + tile_x + 4 + (tile_y + 6)*F] = temp46;
			tt[index + tile_x + 4 + (tile_y + 7)*F] = temp47;
			tt[index + tile_x + 4 + (tile_y + 8)*F] = temp48;
			tt[index + tile_x + 4 + (tile_y + 9)*F] = temp49;

			tt[index + tile_x + 5 + tile_y*F] = temp50;
			tt[index + tile_x + 5 + (tile_y + 1)*F] = temp51;
			tt[index + tile_x + 5 + (tile_y + 2)*F] = temp52;
			tt[index + tile_x + 5 + (tile_y + 3)*F] = temp53;
			tt[index + tile_x + 5 + (tile_y + 4)*F] = temp54;
			tt[index + tile_x + 5 + (tile_y + 5)*F] = temp55;
			tt[index + tile_x + 5 + (tile_y + 6)*F] = temp56;
			tt[index + tile_x + 5 + (tile_y + 7)*F] = temp57;
			tt[index + tile_x + 5 + (tile_y + 8)*F] = temp58;
			tt[index + tile_x + 5 + (tile_y + 9)*F] = temp59;

			tt[index + tile_x + 6 + tile_y*F] = temp60;
			tt[index + tile_x + 6 + (tile_y + 1)*F] = temp61;
			tt[index + tile_x + 6 + (tile_y + 2)*F] = temp62;
			tt[index + tile_x + 6 + (tile_y + 3)*F] = temp63;
			tt[index + tile_x + 6 + (tile_y + 4)*F] = temp64;
			tt[index + tile_x + 6 + (tile_y + 5)*F] = temp65;
			tt[index + tile_x + 6 + (tile_y + 6)*F] = temp66;
			tt[index + tile_x + 6 + (tile_y + 7)*F] = temp67;
			tt[index + tile_x + 6 + (tile_y + 8)*F] = temp68;
			tt[index + tile_x + 6 + (tile_y + 9)*F] = temp69;

			tt[index + tile_x + 7 + tile_y*F] = temp70;
			tt[index + tile_x + 7 + (tile_y + 1)*F] = temp71;
			tt[index + tile_x + 7 + (tile_y + 2)*F] = temp72;
			tt[index + tile_x + 7 + (tile_y + 3)*F] = temp73;
			tt[index + tile_x + 7 + (tile_y + 4)*F] = temp74;
			tt[index + tile_x + 7 + (tile_y + 5)*F] = temp75;
			tt[index + tile_x + 7 + (tile_y + 6)*F] = temp76;
			tt[index + tile_x + 7 + (tile_y + 7)*F] = temp77;
			tt[index + tile_x + 7 + (tile_y + 8)*F] = temp78;
			tt[index + tile_x + 7 + (tile_y + 9)*F] = temp79;

			tt[index + tile_x + 8 + tile_y*F] = temp80;
			tt[index + tile_x + 8 + (tile_y + 1)*F] = temp81;
			tt[index + tile_x + 8 + (tile_y + 2)*F] = temp82;
			tt[index + tile_x + 8 + (tile_y + 3)*F] = temp83;
			tt[index + tile_x + 8 + (tile_y + 4)*F] = temp84;
			tt[index + tile_x + 8 + (tile_y + 5)*F] = temp85;
			tt[index + tile_x + 8 + (tile_y + 6)*F] = temp86;
			tt[index + tile_x + 8 + (tile_y + 7)*F] = temp87;
			tt[index + tile_x + 8 + (tile_y + 8)*F] = temp88;
			tt[index + tile_x + 8 + (tile_y + 9)*F] = temp89;

			tt[index + tile_x + 9 + tile_y*F] = temp90;
			tt[index + tile_x + 9 + (tile_y + 1)*F] = temp91;
			tt[index + tile_x + 9 + (tile_y + 2)*F] = temp92;
			tt[index + tile_x + 9 + (tile_y + 3)*F] = temp93;
			tt[index + tile_x + 9 + (tile_y + 4)*F] = temp94;
			tt[index + tile_x + 9 + (tile_y + 5)*F] = temp95;
			tt[index + tile_x + 9 + (tile_y + 6)*F] = temp96;
			tt[index + tile_x + 9 + (tile_y + 7)*F] = temp97;
			tt[index + tile_x + 9 + (tile_y + 8)*F] = temp98;
			tt[index + tile_x + 9 + (tile_y + 9)*F] = temp99;

			//symmetric
			if(tile_x!=tile_y){
				tt[index + tile_y + 0+ (tile_x + 0)*F]= temp0;
				tt[index + tile_y + 1+ (tile_x + 0)*F]= temp1;
				tt[index + tile_y + 2+ (tile_x + 0)*F]= temp2;
				tt[index + tile_y + 3+ (tile_x + 0)*F]= temp3;
				tt[index + tile_y + 4+ (tile_x + 0)*F]= temp4;
				tt[index + tile_y + 5+ (tile_x + 0)*F]= temp5;
				tt[index + tile_y + 6+ (tile_x + 0)*F]= temp6;
				tt[index + tile_y + 7+ (tile_x + 0)*F]= temp7;
				tt[index + tile_y + 8+ (tile_x + 0)*F]= temp8;
				tt[index + tile_y + 9+ (tile_x + 0)*F]= temp9;


				tt[index + tile_y + 0+ (tile_x + 1)*F]= temp10;
				tt[index + tile_y + 1+ (tile_x + 1)*F]= temp11;
				tt[index + tile_y + 2+ (tile_x + 1)*F]= temp12;
				tt[index + tile_y + 3+ (tile_x + 1)*F]= temp13;
				tt[index + tile_y + 4+ (tile_x + 1)*F]= temp14;
				tt[index + tile_y + 5+ (tile_x + 1)*F]= temp15;
				tt[index + tile_y + 6+ (tile_x + 1)*F]= temp16;
				tt[index + tile_y + 7+ (tile_x + 1)*F]= temp17;
				tt[index + tile_y + 8+ (tile_x + 1)*F]= temp18;
				tt[index + tile_y + 9+ (tile_x + 1)*F]= temp19;


				tt[index + tile_y + 0+ (tile_x + 2)*F]= temp20;
				tt[index + tile_y + 1+ (tile_x + 2)*F]= temp21;
				tt[index + tile_y + 2+ (tile_x + 2)*F]= temp22;
				tt[index + tile_y + 3+ (tile_x + 2)*F]= temp23;
				tt[index + tile_y + 4+ (tile_x + 2)*F]= temp24;
				tt[index + tile_y + 5+ (tile_x + 2)*F]= temp25;
				tt[index + tile_y + 6+ (tile_x + 2)*F]= temp26;
				tt[index + tile_y + 7+ (tile_x + 2)*F]= temp27;
				tt[index + tile_y + 8+ (tile_x + 2)*F]= temp28;
				tt[index + tile_y + 9+ (tile_x + 2)*F]= temp29;


				tt[index + tile_y + 0+ (tile_x + 3)*F]= temp30;
				tt[index + tile_y + 1+ (tile_x + 3)*F]= temp31;
				tt[index + tile_y + 2+ (tile_x + 3)*F]= temp32;
				tt[index + tile_y + 3+ (tile_x + 3)*F]= temp33;
				tt[index + tile_y + 4+ (tile_x + 3)*F]= temp34;
				tt[index + tile_y + 5+ (tile_x + 3)*F]= temp35;
				tt[index + tile_y + 6+ (tile_x + 3)*F]= temp36;
				tt[index + tile_y + 7+ (tile_x + 3)*F]= temp37;
				tt[index + tile_y + 8+ (tile_x + 3)*F]= temp38;
				tt[index + tile_y + 9+ (tile_x + 3)*F]= temp39;


				tt[index + tile_y + 0+ (tile_x + 4)*F]= temp40;
				tt[index + tile_y + 1+ (tile_x + 4)*F]= temp41;
				tt[index + tile_y + 2+ (tile_x + 4)*F]= temp42;
				tt[index + tile_y + 3+ (tile_x + 4)*F]= temp43;
				tt[index + tile_y + 4+ (tile_x + 4)*F]= temp44;
				tt[index + tile_y + 5+ (tile_x + 4)*F]= temp45;
				tt[index + tile_y + 6+ (tile_x + 4)*F]= temp46;
				tt[index + tile_y + 7+ (tile_x + 4)*F]= temp47;
				tt[index + tile_y + 8+ (tile_x + 4)*F]= temp48;
				tt[index + tile_y + 9+ (tile_x + 4)*F]= temp49;


				tt[index + tile_y + 0+ (tile_x + 5)*F]= temp50;
				tt[index + tile_y + 1+ (tile_x + 5)*F]= temp51;
				tt[index + tile_y + 2+ (tile_x + 5)*F]= temp52;
				tt[index + tile_y + 3+ (tile_x + 5)*F]= temp53;
				tt[index + tile_y + 4+ (tile_x + 5)*F]= temp54;
				tt[index + tile_y + 5+ (tile_x + 5)*F]= temp55;
				tt[index + tile_y + 6+ (tile_x + 5)*F]= temp56;
				tt[index + tile_y + 7+ (tile_x + 5)*F]= temp57;
				tt[index + tile_y + 8+ (tile_x + 5)*F]= temp58;
				tt[index + tile_y + 9+ (tile_x + 5)*F]= temp59;


				tt[index + tile_y + 0+ (tile_x + 6)*F]= temp60;
				tt[index + tile_y + 1+ (tile_x + 6)*F]= temp61;
				tt[index + tile_y + 2+ (tile_x + 6)*F]= temp62;
				tt[index + tile_y + 3+ (tile_x + 6)*F]= temp63;
				tt[index + tile_y + 4+ (tile_x + 6)*F]= temp64;
				tt[index + tile_y + 5+ (tile_x + 6)*F]= temp65;
				tt[index + tile_y + 6+ (tile_x + 6)*F]= temp66;
				tt[index + tile_y + 7+ (tile_x + 6)*F]= temp67;
				tt[index + tile_y + 8+ (tile_x + 6)*F]= temp68;
				tt[index + tile_y + 9+ (tile_x + 6)*F]= temp69;


				tt[index + tile_y + 0+ (tile_x + 7)*F]= temp70;
				tt[index + tile_y + 1+ (tile_x + 7)*F]= temp71;
				tt[index + tile_y + 2+ (tile_x + 7)*F]= temp72;
				tt[index + tile_y + 3+ (tile_x + 7)*F]= temp73;
				tt[index + tile_y + 4+ (tile_x + 7)*F]= temp74;
				tt[index + tile_y + 5+ (tile_x + 7)*F]= temp75;
				tt[index + tile_y + 6+ (tile_x + 7)*F]= temp76;
				tt[index + tile_y + 7+ (tile_x + 7)*F]= temp77;
				tt[index + tile_y + 8+ (tile_x + 7)*F]= temp78;
				tt[index + tile_y + 9+ (tile_x + 7)*F]= temp79;


				tt[index + tile_y + 0+ (tile_x + 8)*F]= temp80;
				tt[index + tile_y + 1+ (tile_x + 8)*F]= temp81;
				tt[index + tile_y + 2+ (tile_x + 8)*F]= temp82;
				tt[index + tile_y + 3+ (tile_x + 8)*F]= temp83;
				tt[index + tile_y + 4+ (tile_x + 8)*F]= temp84;
				tt[index + tile_y + 5+ (tile_x + 8)*F]= temp85;
				tt[index + tile_y + 6+ (tile_x + 8)*F]= temp86;
				tt[index + tile_y + 7+ (tile_x + 8)*F]= temp87;
				tt[index + tile_y + 8+ (tile_x + 8)*F]= temp88;
				tt[index + tile_y + 9+ (tile_x + 8)*F]= temp89;


				tt[index + tile_y + 0+ (tile_x + 9)*F]= temp90;
				tt[index + tile_y + 1+ (tile_x + 9)*F]= temp91;
				tt[index + tile_y + 2+ (tile_x + 9)*F]= temp92;
				tt[index + tile_y + 3+ (tile_x + 9)*F]= temp93;
				tt[index + tile_y + 4+ (tile_x + 9)*F]= temp94;
				tt[index + tile_y + 5+ (tile_x + 9)*F]= temp95;
				tt[index + tile_y + 6+ (tile_x + 9)*F]= temp96;
				tt[index + tile_y + 7+ (tile_x + 9)*F]= temp97;
				tt[index + tile_y + 8+ (tile_x + 9)*F]= temp98;
				tt[index + tile_y + 9+ (tile_x + 9)*F]= temp99;
			}
			//add regularization
			if(tile_x == tile_y){
				for(int k = 0; k < tile; k++)
					tt[index + (tile_x+k)*(1+F)] += (end - start) * lambda;
			}
		}
	}
}

void loadCSRSparseMatrixBin(const char* dataFile, const char* rowFile, const char* colFile,
		float* data, int* row, int* col, const int m, const long nnz) {
    printf("\n loading CSR...\n");
	FILE *dFile = fopen(dataFile,"rb");
	FILE *rFile = fopen(rowFile,"rb");
	FILE *cFile = fopen(colFile,"rb");
	if (!rFile||!dFile||!dFile)
	{
		printf("Unable to open file!");
		return;
	}

	fread(&row[0], 4*(m+1) ,1, rFile);
	fread(&col[0], 4*nnz ,1, cFile);
	fread(&data[0], 4*nnz ,1, dFile);

	fclose(rFile);
	fclose(dFile);
	fclose(cFile);
}

/* To solve many Ax = b;
 * As are the tt from the get_hermitianT10 kernel
 * Use LU decomposition from cublas to solve
 */
int updateX(cudaStream_t cuda_stream, const int batch_size, const int batch_offset, float * ythetaT, double * tt, float * XT,
		cublasHandle_t handle, const int m, const int n, const int f, const int nnz,
		double** devPtrTTHost, double **devPtrYthetaTHost){	
	//pointers needed by batch op
	double **devPtrTT = 0;
	int *INFO;
	for (int k = 0; k < batch_size; k++) {
		devPtrTTHost[k] = &tt[k * f * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrTT, batch_size * sizeof(*devPtrTT)));
	cudacall(cudaMemcpyAsync(devPtrTT, devPtrTTHost, batch_size * sizeof(*devPtrTT),cudaMemcpyHostToDevice, cuda_stream));
	cudacall( cudaMalloc(&INFO, batch_size * sizeof(int) ));
	
	cublasSetStream(handle, cuda_stream); 
	//LU decomposition, without pivoting
	cublascall(cublasDgetrfBatched(handle, f, devPtrTT, f, NULL, INFO, batch_size));
	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();
	//printf("*******solve: tt * XT = ythetaT use cublas, with CUDA 7.\n");

	double * ythetaT_d;
	cudacall(cudaMalloc((void** ) &ythetaT_d, batch_size * f* sizeof(double)));
	floatArray2doubleArray<<<(batch_size*f-1)/1024 + 1, 1024, 0, cuda_stream>>>(&ythetaT[batch_offset*f],ythetaT_d, batch_size*f);
	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();

	double **devPtrYthetaT = 0;

	for (int k = 0; k < batch_size; k++) {
		devPtrYthetaTHost[k] = &ythetaT_d[k * f];
	}
	cudacall(cudaMalloc((void** ) &devPtrYthetaT, batch_size * sizeof(*devPtrYthetaT)));
	cudacall(cudaMemcpyAsync(devPtrYthetaT, devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaT), cudaMemcpyHostToDevice, cuda_stream));

	int * info2 = (int *) malloc(sizeof(int));
	cublasSetStream(handle, cuda_stream); 
	cublascall( cublasDgetrsBatched(handle, CUBLAS_OP_N, f, 1,
			(const double ** ) devPtrTT, f, NULL, devPtrYthetaT, f, info2, batch_size) );

	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();

	doubleArray2floatArray<<<(batch_size*f-1)/1024 + 1, 1024, 0, cuda_stream>>>(ythetaT_d, &ythetaT[batch_offset*f], batch_size*f);
	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();


	cudacall( cudaMemcpyAsync(&XT[batch_offset * f], &ythetaT[batch_offset * f],
			batch_size * f * sizeof(float), cudaMemcpyDeviceToDevice, cuda_stream) );
	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();

	cudacall(cudaFree(devPtrTT));
	cudacall(cudaFree(INFO));
	cudacall(cudaFree(devPtrYthetaT));
	cudacall(cudaFree(ythetaT_d));
	return 0;

}


//from thetaT to X, do a half ALS iteration
void doALSWithCSR(cudaStream_t cuda_stream, int* csrRowIndex, int* csrColIndex, float* csrVal,
		float* thetaTHost, float* XTHost,
		const int m, const int n, const int f, const long nnz, const float lambda,
		const int X_BATCH)
{
	double elapsed = 0.0;
	struct timeval tv;
	struct timeval start_tv;
	gettimeofday(&start_tv, NULL);
	//device pointers
	float * thetaT = 0;
	double * tt = 0;
	float * XT = 0;
	//dimension: F*N
	cudacall(cudaMalloc((void** ) &thetaT, f * n * sizeof(thetaT[0])));
	//dimension: M*F
	cudacall(cudaMalloc((void** ) &XT, f * m * sizeof(XT[0])));

	//printf("*******start copying memory to GPU...\n");
	cudacall(cudaMemcpyAsync(thetaT, thetaTHost, (size_t ) (n * f * sizeof(thetaT[0])), cudaMemcpyHostToDevice, cuda_stream));
	cudacall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	//64-bit smem access
	//http://acceleware.com/blog/maximizing-shared-memory-bandwidth-nvidia-kepler-gpus
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//initialize cublas, cusparse
	cublasHandle_t handle;
	cublascall(cublasCreate(&handle));
	cusparseHandle_t cushandle = 0;
	cusparsecall(cusparseCreate(&cushandle));
	cusparseMatDescr_t descr;
	cusparsecall( cusparseCreateMatDescr(&descr));
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
	using namespace std;

	//printf("\tgenerate: Y*theta using cusparse.\n");
	float * ytheta = 0;
	float * ythetaT = 0;
	cudacall(cudaMalloc((void** ) &ytheta, f * m * sizeof(ytheta[0])));
	cudacall(cudaMalloc((void** ) &ythetaT, f * m * sizeof(ythetaT[0])));

	const float alpha = 1.0f;
	const float beta = 0.0f;
	cusparseSetStream(cushandle, cuda_stream); 
	cusparsecall (cusparseScsrmm2(cushandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			CUSPARSE_OPERATION_TRANSPOSE, m, f, n, nnz, &alpha, descr, csrVal,
			csrRowIndex, csrColIndex, thetaT, f, &beta, ytheta, m) );
	//printf("*******transpose ytheta use cublas.\n");
	//ytheta: m*f; need ythetaT = (ytheta).T = f*m
	cublasSetStream(handle, cuda_stream); 
	cublascall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, f, m, &alpha,
			(const float * ) ytheta, m, &beta, ythetaT, f, ythetaT, f));
	cudaStreamSynchronize(cuda_stream);
	cudaCheckError();
	cudacall(cudaFree(ytheta));
	cudacall(cudaFree(csrVal));

	for(int batch_id = 0; batch_id< X_BATCH; batch_id ++){
		//printf("*******batch %d / %d.*******\n", batch_id, X_BATCH);
		int batch_size = 0;
		if(batch_id != X_BATCH - 1)
			batch_size = m/X_BATCH;
		else
			batch_size = m - batch_id*(m/X_BATCH);
		int batch_offset = batch_id * (m/X_BATCH);
		cudacall(cudaMalloc((void** ) &tt, f * f * batch_size * sizeof(double)));
		
		int block_dim = f/T10*(f/T10+1)/2;
		if (block_dim < f/2) block_dim = f/2;

		if(f == 100)
			get_hermitian100<<<batch_size, 64, SCAN_BATCH * f/2*sizeof(float2), cuda_stream>>>
				(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
		else
			get_hermitianT10<<<batch_size, block_dim, SCAN_BATCH * f/2*sizeof(float2), cuda_stream>>>
				(batch_offset, tt, csrRowIndex, csrColIndex, lambda, m, f, thetaT);
		cudaStreamSynchronize(cuda_stream);
		cudaCheckError();

		double ** devPtrTTHost = 0;
		cudacall(cudaMallocHost( (void** ) &devPtrTTHost, batch_size * sizeof(*devPtrTTHost) ) );
		double **devPtrYthetaTHost = 0;
		cudacall(cudaMallocHost( (void** ) &devPtrYthetaTHost, batch_size * sizeof(*devPtrYthetaTHost) ) );

		//printf("\tinvoke updateX with batch_size: %d, batch_offset: %d..\n", batch_size, batch_offset);
		updateX(cuda_stream, batch_size, batch_offset, ythetaT, tt, XT, handle, m, n, f, nnz,
				devPtrTTHost, devPtrYthetaTHost);
		cudaStreamSynchronize(cuda_stream);
		cudaCheckError();
		
		//printf("\tupdateX run seconds: %f \n", seconds() - t0);
		cudacall(cudaFree(tt));
		cudacall(cudaFreeHost(devPtrTTHost));
		cudacall(cudaFreeHost(devPtrYthetaTHost));
	}
	cudacall(cudaFree(csrRowIndex));
	cudacall(cudaFree(csrColIndex));
	cudacall(cudaFree(ythetaT));

	//copy feature vectors back to host
	cudacall(cudaMemcpy(XTHost, XT, (size_t ) (m * f * sizeof(XT[0])), cudaMemcpyDeviceToHost));
	cudacall(cudaFree(thetaT));
	cudacall(cudaFree(XT));
	//cudacall(cudaDeviceReset());
	gettimeofday(&tv, NULL);
	elapsed = (tv.tv_sec - start_tv.tv_sec) + (tv.tv_usec - start_tv.tv_usec) / 1000000.0;
	printf("    doALSWithCSR() runs %.3f seconds, gridSize: %d, blockSize %d.\n", elapsed, m, f);

}
