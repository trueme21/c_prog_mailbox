#ifndef STRIDY_H
#define STRIDY_H

#include <stdio.h>
#include <iostream>

template<typename T>
inline T* get2df(T* p, const unsigned int x, unsigned int y, const unsigned int stride) {
        return (T*)(p + x * stride) + y;
}

template<typename T>
inline T* get2df(T* p, int x, int y, const unsigned int stride) {
        return (T*)(p + x * stride) + y;
}

template<typename T>
inline T* get2df(T* p, unsigned int x, int y, const unsigned int stride) {
        return (T*)(p + x * stride) + y;
}

template<typename T>
inline T* get2df(T* p, long x, int y, const unsigned int stride) {
        return (T*)(p + ((int) x) * stride) + y;
}



template<typename T>
inline T* get2df(T* p, long x, long y, const unsigned int stride) {
        return (T*)(p + ((int)x) * stride) + ((int)y);
}

#endif
