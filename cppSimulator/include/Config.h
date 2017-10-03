#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <Python.h>

class Config {
public:
    static const double tick_per_second;
    static const double map_div;
    static const PyObject* rad_init_pos;
    static const PyObject* dire_init_pos;
    static const double velocity;
    static const double bound_length;
};

#endif//__CONFIG_H__