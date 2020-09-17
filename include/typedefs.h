/*
 * Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#if defined(__GNUC__) || defined(__clang__)

#pragma push_macro("FORCE_INLINE")
#pragma push_macro("ALIGN_STRUCT")
#define FORCE_INLINE static inline __attribute__((always_inline))
#define ALIGN_STRUCT(x) __attribute__((aligned(x)))

#else

#error "Macro name collisions may happens with unknown compiler"
#ifdef FORCE_INLINE
#undef FORCE_INLINE
#endif

#define FORCE_INLINE static inline
#ifndef ALIGN_STRUCT
#define ALIGN_STRUCT(x) __declspec(align(x))
#endif

#endif

#define likely(x)     __builtin_expect(!!(x), 1)
#define unlikely(x)   __builtin_expect(!!(x), 0)

#ifndef __int32
#define __int32 int
#endif

#ifndef __int64
#define __int64 long long
#endif

#ifndef __mmask64
#define __mmask64 unsigned long long
#endif

#ifndef __mmask16
#define __mmask16 unsigned short
#endif

#ifndef __mmask8
#define __mmask8 unsigned char
#endif

/* Constants for use with _mm_prefetch.  */
enum _mm_hint {
    _MM_HINT_NTA = 0,  /* load data to L1 and L2 cache, mark it as NTA */
    _MM_HINT_T0 = 1,   /* load data to L1 and L2 cache */
    _MM_HINT_T1 = 2,   /* load data to L2 cache only */
    _MM_HINT_T2 = 3,   /* load data to L2 cache only, mark it as NTA */
    _MM_HINT_ENTA = 4, /* exclusive version of _MM_HINT_NTA */
    _MM_HINT_ET0 = 5,  /* exclusive version of _MM_HINT_T0 */
    _MM_HINT_ET1 = 6,  /* exclusive version of _MM_HINT_T1 */
    _MM_HINT_ET2 = 7   /* exclusive version of _MM_HINT_T2 */
};

#endif //TYPEDEFS_H
