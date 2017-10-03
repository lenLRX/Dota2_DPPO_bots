#ifndef __SIMULATORIMP_H__
#define __SIMULATORIMP_H__

#include "simulator.h"

class cppSimulatorImp
{
public:
    cppSimulatorImp() = delete;
    cppSimulatorImp(cppSimulatorObject* obj);
    int get_time();
private:
    float tick_time;
    float tick_per_second;
};

#endif//__SIMULATORIMP_H__