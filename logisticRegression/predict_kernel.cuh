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

#ifndef _PRD_CUDA_H_
#define _PRD_CUDA_H_

__global__ void
score_kernel(
		const int numSamples,
		const double *d_margin,
		const double intercept,
		double *d_score)
{

   int iam = threadIdx.x;
   int bid = blockIdx.x;
   int threads = blockDim.x;
   int gid = bid * threads + iam;

   double margin=0.0, score=0.0;


   if (gid < numSamples){

        margin = d_margin[gid] + intercept;
        score = 1.0 / (1.0 + exp(-margin));
      
        d_score[gid] = score;
   }



}


#endif
