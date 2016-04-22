/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDA_CHECKING_H_
#define CUDA_CHECKING_H_

#include <string>
#include <cuda.h>
#include <cublas_v2.h>



extern const char *cublasGetErrorString(cublasStatus_t e);

#define checkCublasErrors(err)    __cublasCheckError( err, __FILE__, __LINE__ )
inline void __cublasCheckError( cublasStatus_t err, const char *file, const int line )
{
#ifdef CUBLAS_ERROR_CHECK
    if ( CUBLAS_STATUS_SUCCESS != err )
    {
        fprintf( stderr, "CUBLAS call failed at %s:%i : %s\n",
                 file, line, cublasGetErrorString( err ) );
        exit( -1 );
    }
#endif
}

#endif

