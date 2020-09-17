#ifndef DEBUG_H
#define DEBUG_H


template<typename T>
void printVector(T* vector, int size) {
    for(int i = 0; i < size ; ++i)
        std::cout << vector[i] << " ";
    std::cout << std::endl;
}

#endif
